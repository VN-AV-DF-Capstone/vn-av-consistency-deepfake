"""VAD + scene boundaries + single-face quality, inspired by the Capstone cutter.

Quality acceptance does not assert audio-visual synchrony. Every clip retains source times.
"""

import csv
import io
import math
import time
from collections import Counter
from itertools import pairwise
from pathlib import Path

from vn_av_data.common.runtime import atomic_bytes, fingerprint, read_json, run, sha, write_json
from vn_av_data.data.acquisition import source_path
from vn_av_data.data.manifest import read_manifest
from vn_av_data.data.media import CLIP_FPS, MAX_SHORT_SIDE, ffmpeg, probe, quality_issue


def write_csv(path, rows, fields=None):
    stream = io.StringIO(newline="")
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, stream.getvalue().encode("utf-8-sig"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def media_origin(path):
    import av

    info = probe(path)
    with av.open(str(path)) as container:
        starts = [info[k] for k in ("video_start_s", "audio_start_s") if info[k] is not None]
        origin = (
            float(container.start_time / av.time_base)
            if container.start_time is not None
            else min(starts, default=0.0)
        )
    return origin, info


def scan_visual(path, cfg, origin, tracker=None):
    import av
    import cv2

    from vn_av_data.features.face import FaceTracker

    tracker = tracker or FaceTracker(cfg["face_model"], min_size=cfg["min_face_size"])
    samples, scenes, previous_hist, next_sample = [], [], None, 0.0
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            if frame.pts is None:
                raise ValueError("Video frame without timestamp")
            time = float(frame.pts * frame.time_base) - origin
            image = frame.to_ndarray(format="bgr24")
            tiny = cv2.resize(image, (96, 54))
            hist = cv2.calcHist([tiny], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
            cv2.normalize(hist, hist)
            if (
                previous_hist is not None
                and cv2.compareHist(previous_hist, hist, cv2.HISTCMP_CORREL)
                < cfg["scene_correlation"]
            ):
                scenes.append(max(0, time))
                tracker.previous = None
            previous_hist = hist
            if time + 1e-6 < next_sample:
                continue
            scale = min(1, cfg["scan_max_side"] / max(image.shape[:2]))
            image = cv2.resize(
                image, (round(image.shape[1] * scale), round(image.shape[0] * scale))
            )
            info = tracker.inspect(image)
            samples.append({"time_s": max(0, time), **info})
            next_sample = time + 1 / cfg["sample_fps"]
    return samples, scenes


def every_frame_faces(path, cfg, origin, start, end, tracker=None):
    """Mọi frame trong cửa sổ có đúng một mặt hợp lệ (luật của generator: Wav2Lip/MuseTalk dò
    mặt ở mọi frame). Bước quét 4 fps không bảo đảm điều này nên cửa sổ đã qua quét được kiểm lại.
    """
    import av
    import cv2

    from vn_av_data.features.face import FaceTracker

    tracker = tracker or FaceTracker(cfg["face_model"], min_size=cfg["min_face_size"])
    tracker.previous = None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(max(0.0, origin + start - 1) / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Video frame without timestamp")
            time = float(frame.pts * frame.time_base) - origin
            if time < start - 1e-6:
                continue
            if time >= end - 1e-6:
                break
            image = frame.to_ndarray(format="bgr24")
            scale = min(1, cfg["scan_max_side"] / max(image.shape[:2]))
            image = cv2.resize(
                image, (round(image.shape[1] * scale), round(image.shape[0] * scale))
            )
            if not tracker.inspect(image)["valid"]:
                return False
    return True


def plan_clips(speech, scenes, duration, cfg):
    minimum, maximum, overlap = cfg["min_seconds"], cfg["max_seconds"], cfg["overlap_seconds"]
    if not (0 < minimum <= maximum <= 60 and 0 <= overlap < minimum):
        raise ValueError("Need 0 <= overlap < min <= max <= 60 seconds")
    boundaries = sorted({0.0, duration, *(s for s in scenes if 0 < s < duration)})
    result = []
    for start, end in speech:
        for left, right in pairwise(boundaries):
            a, b = max(start, left), min(end, right)
            a, b = math.ceil(a * 25 - 1e-6) / 25, math.floor(b * 25 + 1e-6) / 25
            while b - a >= minimum - 1e-6:
                stop = min(b, a + maximum)
                # Absorb a short tail by moving the final full window backwards.
                if 0 < b - stop < minimum - overlap:
                    stop = b
                    a = max(a, stop - maximum)
                result.append((round(a, 6), round(stop, 6)))
                if stop == b:
                    break
                a = stop - overlap
    return sorted(set(result))


def cut_media(source, output, start, end, origin):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.mp4")
    # Both streams subtract the SAME timestamp. Never independently reset STARTPTS.
    absolute = origin + start
    stop = origin + end
    filters = (
        f"[0:v:0]trim=start={absolute:.6f}:end={stop:.6f},"
        f"setpts=PTS-({absolute:.6f})/TB,"
        # Chuẩn hoá clip: 25 fps CFR (bỏ frame, nguồn đã ≥25 fps) và cạnh ngắn ≤1080, giữ tỉ lệ.
        f"fps={CLIP_FPS},"
        f"scale=w='if(gte(iw,ih),-2,min(trunc(iw/2)*2,{MAX_SHORT_SIDE}))'"
        f":h='if(gte(iw,ih),min(trunc(ih/2)*2,{MAX_SHORT_SIDE}),-2)'[v];"
        f"[0:a:0]atrim=start={absolute:.6f}:end={stop:.6f},"
        f"asetpts=PTS-({absolute:.6f})/TB[a]"
    )
    try:
        run(
            [
                ffmpeg(),
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-copyts",
                "-ss",
                str(start),
                "-i",
                str(source),
                "-filter_complex",
                filters,
                "-map",
                "[v]",
                "-map",
                "[a]",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-fps_mode",
                "cfr",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-ar",
                "48000",
                "-avoid_negative_ts",
                "disabled",
                "-movflags",
                "+faststart",
                partial,
            ]
        )
        info = probe(partial)
        if not info["duration_s"] or abs(info["duration_s"] - (end - start)) > 0.2:
            raise ValueError("Encoded clip duration disagrees with requested source interval")
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)


def curate_sources(
    manifest, output, cfg, speech_detector=None, visual_scanner=None, frame_checker=None
):
    from vn_av_data.data.source_io import read_rows

    progress = Path(manifest).resolve().parent / "logs/download_results.csv"
    if progress.exists() and any(
        r.get("status") not in {"downloaded", "rejected"} for r in read_rows(progress)
    ):
        raise ValueError(
            "Download batch still contains pending/failed sources; finish step 02 first"
        )
    from vn_av_data.data.vad import detect_speech

    cfg = dict(cfg["curation"])
    if not 0 < cfg["sample_fps"] <= 25 or not 0 <= cfg["min_face_ratio"] <= 1:
        raise ValueError("Invalid visual sampling/gate configuration")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(manifest)
    if not rows or len({r["source_id"] for r in rows}) != len(rows):
        raise ValueError("Need unique source IDs")
    # Chữ ký không gồm danh sách nguồn: tải thêm video thì cắt tiếp trong cùng thư mục,
    # chỉ đổi luật cắt/code/model mới bị chặn để clip cũ và mới không lẫn luật.
    signature = fingerprint(
        {
            "config": cfg,
            "implementation": [
                sha(__file__),
                sha(Path(__file__).with_name("vad.py")),
                sha(Path(__file__).parents[1] / "features/face.py"),
            ],
            "models": {key: sha(cfg[key]) for key in ("face_model", "vad_model")},
        }
    )
    # clips/ + review.csv là dữ liệu đi cùng nhau; logs/ chỉ để đọc và để cắt tiếp khi bị ngắt.
    logs = output / "logs"
    lock = logs / "cut-run.json"
    if lock.exists() and read_json(lock)["signature"] != signature:
        raise ValueError("Cut settings/code changed; delete this cut directory to recut the part")
    write_json(lock, {"signature": signature, "config": cfg})
    journal = logs / "sources.json"
    finished = read_json(journal) if journal.exists() else {}
    candidates, rejected, errors = [], [], []
    for number, source in enumerate(rows, 1):
        sid = source["source_id"]
        head = f"[{number}/{len(rows)}] {sid}"
        path = source_path(source, manifest)
        if sha(path) != source["sha256"]:
            raise ValueError(f"Source changed: {path}")
        if sid in finished:
            done = finished[sid]
            for clip in done["candidates"]:
                target = output / clip["file_path"]
                if not target.is_file() or sha(target) != clip["sha256"]:
                    raise ValueError(f"Published clip missing/changed: {target}; use a new cut run")
            candidates.extend(done["candidates"])
            rejected.extend(done["rejected"])
            print(
                f"{head}: đã cắt ở lượt trước ({len(done['candidates'])} clip), bỏ qua", flush=True
            )
            continue
        try:
            began = time.monotonic()
            origin, info = media_origin(path)
            duration = info["duration_s"]
            # Thông số nguồn ghi vào nhật ký để thống kê (notebook/W&B) cả khi nguồn bị loại.
            media = {
                "duration_s": duration,
                "fps": info["fps"],
                "width": info["width"],
                "height": info["height"],
                "codec": info.get("video_codec"),
                "bitrate_kbps": info.get("video_bitrate_kbps") or info.get("total_bitrate_kbps"),
            }
            print(
                f"{head}: {(duration or 0) / 60:.1f} phút, {info['fps'] or 0:.2f} fps, "
                f"{info['width']}x{info['height']}",
                flush=True,
            )
            if not duration or not 0 < duration <= cfg["max_source_hours"] * 3600:
                raise ValueError("Source duration missing or exceeds configured limit")
            issue = quality_issue(info)
            if issue:
                # Nguồn nạp ngoài bước tải (index) vẫn phải qua cùng ngưỡng fps/độ phân giải.
                print(f"  loại nguồn: {issue}", flush=True)
                finished[sid] = {
                    "candidates": [],
                    "rejected": [],
                    "source_rejected": issue,
                    "media": media,
                }
                write_json(journal, finished)
                continue
            # Gộp khoảng lặng ngắn (merge_silence_seconds) để câu nói không bị tách vụn.
            speech = (speech_detector or detect_speech)(
                path, cfg["vad_model"], origin, duration, cfg.get("merge_silence_seconds", 0.3)
            )
            speech_s = sum(end - start for start, end in speech)
            print(
                f"  VAD: {speech_s:.0f}s tiếng nói ({speech_s / duration:.0%}), "
                f"{len(speech)} đoạn | {time.monotonic() - began:.0f}s",
                flush=True,
            )
            samples, scenes = (visual_scanner or scan_visual)(path, cfg, origin)
            valid = sum(s["valid"] for s in samples) / max(1, len(samples))
            print(
                f"  Mặt: {len(samples)} mẫu, {valid:.0%} đạt; {len(scenes)} lần chuyển cảnh "
                f"| {time.monotonic() - began:.0f}s",
                flush=True,
            )
            accepted_source, rejected_source = [], []
            for start, end in plan_clips(speech, scenes, duration, cfg):
                selected = [s for s in samples if start <= s["time_s"] < end]
                ratio = sum(s["valid"] for s in selected) / max(1, len(selected))
                coverage = len(selected) / max(1, (end - start) * cfg["sample_fps"])
                clip_id = "clip_" + fingerprint([sid, start, end])[:24]
                base = {
                    "clip_id": clip_id,
                    "source_id": sid,
                    "url": source.get("url", ""),
                    "source_sha256": source["sha256"],
                    "speaker_id": source.get("speaker_id") or "",
                    "dataset": source.get("dataset", "youtube"),
                    "channel": source.get("channel", ""),
                    "source_start_s": start,
                    "source_end_s": end,
                    # Giữ thông số nguồn để phân tích ảnh hưởng fps/độ phân giải về sau.
                    "source_fps": info["fps"],
                    "source_width": info["width"],
                    "source_height": info["height"],
                    "source_codec": info.get("video_codec"),
                    "source_bitrate_kbps": info.get("video_bitrate_kbps")
                    or info.get("total_bitrate_kbps"),
                    "face_ratio": ratio,
                    "visual_sample_coverage": min(1.0, coverage),
                }
                if coverage < 0.8:  # Thiếu mẫu hình (frame hỏng/khoảng trống), chưa xét mặt.
                    rejected_source.append({**base, "reason": "coverage"})
                    continue
                if ratio < cfg["min_face_ratio"]:
                    rejected_source.append({**base, "reason": "face"})
                    continue
                if cfg.get("every_frame_face") and not (frame_checker or every_frame_faces)(
                    path, cfg, origin, start, end
                ):
                    rejected_source.append({**base, "reason": "face_frame"})
                    continue
                target = output / "clips" / (clip_id + ".mp4")
                # Nguồn chưa ghi vào sources.json = lần trước bị ngắt giữa chừng: cắt lại clip dở.
                target.unlink(missing_ok=True)
                cut_media(path, target, start, end, origin)
                accepted_source.append(
                    {
                        **base,
                        "duration_s": end - start,
                        "file_path": target.relative_to(output).as_posix(),
                        "sha256": sha(target),
                        "quality": "pass",
                        "decision": "uncertain",
                        "sync_status": "unverified",
                    }
                )
            elapsed = time.monotonic() - began
            clip_s = sum(c["duration_s"] for c in accepted_source)
            reasons = Counter(r["reason"] for r in rejected_source)
            finished[sid] = {
                "candidates": accepted_source,
                "rejected": rejected_source,
                "speech_regions": speech,
                "scene_boundaries": scenes,
                "media": media,
                "elapsed_s": round(elapsed, 1),
            }
            write_json(journal, finished)
            print(
                f"  {len(accepted_source) + len(rejected_source)} cửa sổ: giữ "
                f"{len(accepted_source)} clip ({clip_s:.0f}s), loại do mặt {reasons['face']} "
                f"(+{reasons['face_frame']} khi kiểm mọi frame), "
                f"do thiếu mẫu {reasons['coverage']} | {elapsed:.0f}s",
                flush=True,
            )
            candidates.extend(accepted_source)
            rejected.extend(rejected_source)
        except Exception as exc:  # noqa: BLE001 -- record a source failure and continue the batch
            print(f"  lỗi: {exc}", flush=True)
            errors.append({"source_id": sid, "error": str(exc)})
        write_json(logs / "errors.json", errors)
    if candidates:
        write_csv(logs / "candidates.csv", candidates)
        # review.csv chứa quyết định duyệt: giữ nguyên dòng cũ, chỉ nối clip mới cắt.
        review = output / "review.csv"
        reviewed = read_csv(review) if review.exists() else []
        seen = {r["clip_id"] for r in reviewed}
        write_csv(review, reviewed + [c for c in candidates if c["clip_id"] not in seen])
    write_json(logs / "rejected.json", rejected)
    summary = {
        "sources": len(rows),
        "candidates": len(candidates),
        "rejected": len(rejected),
        "errors": len(errors),
        "output": str(output),
        "note": "Quality pass is not verified synchrony; review before importing clean labels",
    }
    write_json(logs / "summary.json", summary)
    if errors:
        raise RuntimeError(
            f"{len(errors)} source failures; see {logs / 'errors.json'}; rerun to resume"
        )
    return summary
