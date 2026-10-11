"""Persistent full/partial AI lip-sync edits with frame-exact manipulation intervals."""

import csv
import io
import queue
import random
import shutil
import tempfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

from vn_av_data.contract import validate_bundle

from vn_av_df import worker
from vn_av_df.common.runtime import atomic_bytes, fingerprint, read_json, sha, write_json
from vn_av_df.data.groups import read_manifest, write_manifest
from vn_av_df.data.media import decode, probe
from vn_av_df.data.render import encode
from vn_av_df.dataset import SCHEMA, validate_rows
from vn_av_df.generators import generator_kind

AUDIO_MODES = ("source", "donor")


def fake_audio_modes(options, split):
    """source: vẽ miệng theo đúng tiếng gốc (chỉ artifact); donor: tiếng thật khác của cùng người.

    Cấu hình cũ chỉ có audio_mode chung cho mọi split vẫn được đọc như trước.
    """
    if "fake_audio_modes_by_split" in options:
        modes = options["fake_audio_modes_by_split"].get(split)
    else:
        modes = [options.get("audio_mode", "source")]
    if not modes or len(set(modes)) != len(modes) or set(modes) - set(AUDIO_MODES):
        raise ValueError("Fake audio modes must be a nonempty unique subset of source/donor")
    return list(modes)


def csv_write(path, rows):
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, text.getvalue().encode("utf-8-sig"))


def make_plan(cfg):
    """Khóa parent/donor/generator theo split đã gán ở 05_export; giữ audio gốc."""
    dest = Path(cfg["plan"])
    if dest.exists():
        raise FileExistsError("Plan exists; reuse it to resume or choose a new run name")
    rows, info = validate_bundle(cfg["clean_dataset"], probe=probe)
    if any("split" not in r for r in rows):
        raise ValueError("Clean part has no split; run 05_export again (split is assigned there)")
    split_info = read_json(Path(cfg["clean_dataset"]) / "dataset_info.json").get("split", {})
    history = len(split_info.get("covered_parts", [1])) > 1
    rows = [{**r, "sample_id": r["clip_id"]} for r in rows]
    count = int(cfg["generation"]["clips_per_split"])
    if count < 0:
        raise ValueError("clips_per_split is 0 for all, otherwise positive")
    selected = []
    for split in ("train", "validation", "test"):
        candidates = [r for r in rows if r["split"] == split and r["duration_s"] >= 5]
        random.Random(cfg["seed"]).shuffle(candidates)
        selected.extend(candidates[:count] if count else candidates)
    jobs, skipped = [], []
    for row in selected:
        # Same-speaker donors stay in split and do not overlap the same source interval.
        donors = [
            d
            for d in rows
            if d["split"] == row["split"]
            and d["clip_id"] != row["clip_id"]
            and d.get("speaker_id")
            and d["speaker_id"] == row.get("speaker_id")
            and d["duration_s"] >= row["duration_s"]
            and (
                d["source_id"] != row["source_id"]
                or d["source_end_s"] <= row["source_start_s"]
                or row["source_end_s"] <= d["source_start_s"]
            )
        ]
        modes = fake_audio_modes(cfg["generation"], row["split"])
        sham_donor = sorted(donors, key=lambda r: r["clip_id"])[0] if donors else None
        if "donor" in modes and not donors:
            # Bỏ cả parent để mỗi parent giữ đủ các ô 2×2 đã chọn; ghi lại để báo cáo.
            skipped.append(row["clip_id"])
            continue
        donor = (
            random.Random(cfg["seed"] + len(jobs)).choice(
                sorted(donors, key=lambda r: r["clip_id"])
            )
            if "donor" in modes
            else row
        )
        generators = (
            cfg["generation"].get("generators_by_split", {}).get(row["split"], ["wav2lip_gan"])
        )
        if not generators or len(set(generators)) != len(generators):
            raise ValueError("Need unique generators per split")
        if generator_kind(generators[0]) != "ai":
            raise ValueError("First generator of a split must be a real AI generator (emits real)")
        mechanisms = [g for g in generators if generator_kind(g) == "mech"]
        # Mỗi parent một cơ chế MechFake (xoay vòng theo clip) được trích đủ AV-HuBERT để train
        # head hợp điểm; cơ chế còn lại chỉ cần DINOv2 (hộp miệng lấy từ real của parent).
        fusion = (
            mechanisms[int(fingerprint(row["clip_id"])[:8], 16) % len(mechanisms)]
            if mechanisms
            else None
        )
        first, parent_jobs = True, []
        for name in generators:
            # MechFake và chẩn đoán luôn giữ tiếng gốc.
            for mode in modes if generator_kind(name) == "ai" else ["source"]:
                job = {
                    "original": row,
                    "donor": donor if mode == "donor" else row,
                    "generator": name,
                    "audio_mode": mode,
                    "emit_real": first,
                    "sham_donor": sham_donor
                    if first and cfg["generation"].get("include_sham", False)
                    else None,
                }
                if generator_kind(name) == "mech":
                    job["fusion_subset"] = name == fusion
                parent_jobs.append(job)
                first = False
        real_sample = fingerprint(parent_jobs[0])[:20] + "_real"
        for job in parent_jobs[1:]:
            if generator_kind(job["generator"]) != "ai":
                job["real_sample_id"] = real_sample
        jobs.extend(parent_jobs)
    if not jobs or (
        not history and {j["original"]["split"] for j in jobs} != {"train", "validation", "test"}
    ):
        raise ValueError("No eligible source/donor pair in a split; add longer independent clips")
    plan = {
        "schema": "vn-av-df-plan-v1",
        "data_part": cfg.get("data_part", 1),
        "split": split_info,
        "manifest_sha256": info["manifest_sha256"],
        "seed": cfg["seed"],
        "assignments": rows,
        "jobs": jobs,
        "config": cfg["generation"],
        "split_counts": dict(Counter(j["original"]["split"] for j in jobs)),
        "skipped_without_donor": skipped,
    }
    write_json(dest, plan)
    total = sum(len(job_kinds(j)) for j in jobs)
    return {
        "pairs": len(jobs),
        "videos": total,
        "splits": plan["split_counts"],
        "skipped_without_donor": len(skipped),
    }


def job_kinds(job):
    """Video sinh ra của một job: chẩn đoán chỉ bản xử lý cả clip; còn lại full + partial."""
    if generator_kind(job.get("generator", "wav2lip_gan")) == "diag":
        return ["full"]
    kinds = (["real"] if job.get("emit_real", True) else []) + ["full", "partial"]
    return kinds + (["sham"] if job.get("sham_donor") else [])


def partial_frames(config, rng, n):
    """Độ dài đoạn partial (frame 25 Hz) lấy ngẫu nhiên liên tục trong partial_range_s."""
    low, high = config["partial_range_s"]
    if not 0 < low <= high:
        raise ValueError("partial_range_s must be [min, max] seconds with 0 < min <= max")
    return max(1, min(n - 2, round(rng.uniform(low, high) * 25)))


def compose_partial(original, fake, span):
    """Identical decoded timeline and dimensions; no stretched/looped donors."""
    if (
        original["frames"].shape != fake["frames"].shape
        or original["pcm"].shape != fake["pcm"].shape
    ):
        raise ValueError("Generated video must preserve source geometry and timeline")
    a, b = span
    n = len(original["frames"])
    if not 0 <= a < b <= n:
        raise ValueError("Partial bounds must be valid frame indices")
    frames, pcm = original["frames"].copy(), original["pcm"].copy()
    frames[a:b] = fake["frames"][a:b]
    pcm[a * 1920 : b * 1920] = fake["pcm"][a * 1920 : b * 1920]
    return frames, pcm, [[a / 25, b / 25]]


def validate_plan_settings(cfg, plan):
    """Không âm thầm đổi cấu hình sinh của plan đã khóa khi chạy lại notebook."""
    if (
        plan["config"] != cfg["generation"]
        or plan["seed"] != cfg["seed"]
        or plan.get("data_part", 1) != cfg.get("data_part", 1)
    ):
        raise ValueError("Generation settings differ from frozen plan; use a new plan/output")


# Lỗi generator không thấy mặt ở một số frame (bước cắt chỉ đòi YuNet thấy mặt ở ≥90% frame
# lấy mẫu 4 fps; Wav2Lip/MuseTalk đòi mọi frame).
FACE_ERRORS = ("Face not detected", "Missing MuseTalk face", "Invalid face bbox")
# Bỏ liên tiếp chừng này cặp thì dừng: lỗi hệ thống (hết VRAM, worker hỏng), không phải clip khó.
MAX_SKIP_STREAK = 10


def skip_reason(exc):
    """Lý do bỏ cặp, hoặc None nếu phải dừng.

    Bỏ qua: lỗi generator báo cho riêng một cặp ("Worker failed", worker vẫn sống).
    Dừng: worker chết/timeout và lỗi ở code pipeline (decode, encode, nhãn).
    """
    text = str(exc)
    face = next((m for m in FACE_ERRORS if m in text), None)
    if face:
        return "face", face
    if text.startswith("Worker failed: "):
        return "generator_error", text.removeprefix("Worker failed: ").splitlines()[0][:200]
    return None


def generate(cfg, generator=None):
    """Sinh bằng worker thực; giữ audio chung cho V-only, provenance và resume nghiêm ngặt.

    cfg["generation_generators"] (vd. ["wav2lip_gan"]) giới hạn generator của phiên này: cặp
    của generator khác để phiên sau (PREVIOUS_GENERATION), nên mỗi phiên chỉ cài một worker.
    """
    from vn_av_df.generators import generator_adapter

    plan = read_json(cfg["plan"])
    validate_plan_settings(cfg, plan)
    root, out = Path(cfg["clean_dataset"]), Path(cfg["generated_dataset"])
    originals, info = validate_bundle(root, probe=probe)
    if (
        plan.get("schema") != "vn-av-df-plan-v1"
        or plan["manifest_sha256"] != info["manifest_sha256"]
    ):
        raise ValueError("Clean dataset differs from plan")
    assigned = [{**r, "sample_id": r["clip_id"]} for r in originals]
    if "assignments" in plan and assigned != plan["assignments"]:
        raise ValueError("Changed split assignments in plan")
    locked = {r["clip_id"]: r for r in assigned}
    for job in plan["jobs"]:
        if (
            any(locked.get(job[k]["clip_id"]) != job[k] for k in ("original", "donor"))
            or job["original"]["split"] != job["donor"]["split"]
        ):
            raise ValueError("Changed source, donor or split")
        if job.get("sham_donor") and (
            locked.get(job["sham_donor"]["clip_id"]) != job["sham_donor"]
            or job["sham_donor"]["split"] != job["original"]["split"]
        ):
            raise ValueError("Changed sham donor or split")
    planned = {j.get("generator", "wav2lip_gan") for j in plan["jobs"]}
    active = set(cfg.get("generation_generators") or planned)
    if active - planned:
        raise ValueError(f"Generators not in plan: {sorted(active - planned)}")
    adapters, provenance = {}, {}
    for name in sorted(active):
        if generator is None:
            adapter = generator_adapter(name)
            adapters[name], provenance[name] = adapter.synthesize, adapter.provenance(cfg)
        else:
            adapters[name] = generator
            provenance[name] = {"generator": "test_fixture", "version": "test-only"}
    # Provenance từng generator kiểm riêng: các phiên (mỗi phiên một generator) chung output.
    signature = fingerprint(
        {
            "plan": plan,
            "implementation": sha(__file__),
            "render": sha(Path(__file__).parent / "data/render.py"),
        }
    )
    lock = out / "generation.json"
    run_log = {"signature": signature, "provenance": {}, "plan": plan, "sessions": []}
    if lock.exists():
        run_log = {**run_log, **read_json(lock)}
        if run_log["signature"] != signature:
            raise ValueError("Changed generation run; choose new output")
    elif out.exists() and any(out.iterdir()):
        raise ValueError("Output has no generation lock")
    for name in active:
        if run_log["provenance"].get(name, provenance[name]) != provenance[name]:
            raise ValueError(f"{name} differs from earlier sessions of this output")
    run_log["provenance"].update(provenance)
    write_json(lock, run_log)
    # Biên bản chia split của part sạch đi kèm part đã sinh (ZIP).
    if (root / "split-lock.json").is_file():
        shutil.copy2(root / "split-lock.json", out / "split-lock.json")
    # GPU sinh song song (mỗi GPU một worker); không cấu hình = tuần tự như trước.
    gpus = list(cfg.get("generation_gpus") or [None])
    # Hạn giờ (epoch giây): quá hạn thì không nhận cặp mới; phần đã sinh giữ để phiên sau làm tiếp.
    deadline = cfg.get("generation_deadline")
    total = len(plan["jobs"])
    pending = sum(
        not (out / "records" / f"{fingerprint(job)[:20]}.json").exists()
        and job.get("generator", "wav2lip_gan") in active
        for job in plan["jobs"]
    )
    print(
        f"Generate {total} cặp ({total - pending} đã có hoặc để phiên khác), phiên này "
        f"{pending} cặp của {sorted(active)} trên {len(gpus)} GPU: "
        f"theo split {plan['split_counts']}, "
        f"{len(plan.get('skipped_without_donor', []))} parent bỏ vì không có donor",
        flush=True,
    )
    # Thư mục tạm còn sót khi phiên trước bị dừng đột ngột.
    for leftover in [*out.glob("tmp*"), *(out / "clips").glob("tmp*")]:
        if leftover.is_dir():
            shutil.rmtree(leftover, ignore_errors=True)
    began = time.monotonic()
    progress = {"done": 0, "skipped": 0, "streak": 0}
    seconds = defaultdict(list)  # Thời gian từng cặp theo generator, ghi vào generation.json.
    slots = queue.Queue()
    for gpu in gpus:
        slots.put(gpu)
    guard = threading.Lock()

    def make_pair(number, job):
        r, d = job["original"], job["donor"]
        name = job.get("generator", "wav2lip_gan")
        head = (
            f"Pair {number + 1}/{total} [{r['split']}] {name}/{job.get('audio_mode', 'source')}: "
            f"parent {r['clip_id']} ({r.get('speaker_id') or '-'})"
        )
        key = fingerprint(job)[:20]
        record = out / "records" / f"{key}.json"
        if record.exists():
            previous = read_json(record)
            if isinstance(previous, dict):  # Cặp đã bỏ qua (không thấy mặt): không sinh lại.
                return []
            if any(sha(out / p["video"]) != p["sha256"] for p in previous):
                raise ValueError("Generated file changed")
            return previous
        if name not in active:
            return None  # Cặp của generator để phiên khác (chưa nạp provenance/adapter).
        gen_info = provenance[name]
        gpu = slots.get()
        if deadline is not None and time.time() > deadline:
            slots.put(gpu)
            return None
        try:
            tick = time.monotonic()
            kinds = job_kinds(job)
            dests = {kind: out / "clips" / f"{key}_{kind}.mp4" for kind in kinds}
            # Có video mà chưa có record = cặp bị ngắt giữa chừng (hết giờ phiên): xoá, sinh lại.
            for stale in dests.values():
                if stale.exists():
                    stale.unlink()
                    print(f"{head}: xoá output dở của lượt trước {stale.name}", flush=True)
            original = decode(
                root / r["video"], max_side=plan["config"]["max_side"], sample_rate=48000
            )
            with tempfile.TemporaryDirectory(dir=out) as temp:
                target = Path(temp) / "generated.mkv"
                # Worker thật nhận thêm gpu (sinh song song); fixture test giữ chữ ký cũ.
                synth = adapters[name] if generator else partial(adapters[name], gpu=gpu)
                try:
                    synth(
                        cfg,
                        root / r["video"],
                        root / d["video"],
                        target,
                        len(original["frames"]),
                        original["frames"].shape[2],
                        original["frames"].shape[1],
                    )
                except Exception as exc:
                    reason = skip_reason(exc)
                    if reason is None:
                        raise
                    kind, reason = reason
                    # Không ghi video nào của cặp (kể cả real) để real/fake vẫn đi theo cặp.
                    skipped = {
                        "skipped": reason,
                        "kind": kind,
                        "generator": name,
                        "audio_mode": job.get("audio_mode", "source"),
                        "split": r["split"],
                        "parent_clip_id": r["clip_id"],
                        "donor_clip_id": d["clip_id"],
                        "speaker_id": r.get("speaker_id"),
                    }
                    write_json(record, skipped)
                    with guard:
                        progress["done"] += 1
                        progress["skipped"] += 1
                        progress["streak"] += 1
                        done, streak = progress["done"], progress["streak"]
                        seconds[name].append(time.monotonic() - tick)
                    print(f"{head}: BỎ QUA ({reason}) | mới {done}/{pending}", flush=True)
                    if streak >= MAX_SKIP_STREAK:
                        raise RuntimeError(
                            f"{streak} cặp liên tiếp bị bỏ, lỗi cuối: {reason}; kiểm tra worker"
                        ) from exc
                    return []
                fake = decode(target, max_side=plan["config"]["max_side"], sample_rate=48000)
            mode = job.get("audio_mode", plan["config"].get("audio_mode", "source"))
            # Không lấy PCM generator (đã qua 16kHz/codec) làm dấu phân biệt fake.
            if mode == "source":
                fake["pcm"] = original["pcm"].copy()
            else:
                donor_audio = decode(
                    root / d["video"], max_side=plan["config"]["max_side"], sample_rate=48000
                )["pcm"]
                if len(donor_audio) < len(original["pcm"]):
                    raise ValueError("Donor audio shorter than parent")
                fake["pcm"] = donor_audio[: len(original["pcm"])].copy()
            n = len(original["frames"])
            rng = random.Random(plan["seed"] + number)
            # Độ dài ngẫu nhiên liên tục: model không học thuộc vài độ dài cố định.
            length = partial_frames(plan["config"], rng, n)
            a = rng.randrange(1, n - length)
            partial_video, partial_audio, spans = compose_partial(original, fake, (a, a + length))
            sham_audio = original["pcm"].copy()
            if job.get("sham_donor"):
                # Ghép âm thanh thu thật, hình không AI: đối chứng dấu vết splice và lệch A/V.
                genuine = decode(
                    root / job["sham_donor"]["video"],
                    max_side=plan["config"]["max_side"],
                    sample_rate=48000,
                )["pcm"]
                if len(genuine) < n * 1920:
                    raise ValueError("Sham donor too short")
                sham_audio[a * 1920 : (a + length) * 1920] = genuine[a * 1920 : (a + length) * 1920]
            generated = []
            family = generator_kind(name)
            for kind, frames, pcm, intervals in (
                ("real", original["frames"], original["pcm"], []),
                ("full", fake["frames"], fake["pcm"], [[0, n / 25]]),
                ("partial", partial_video, partial_audio, spans),
                ("sham", original["frames"], sham_audio, []),
            ):
                if kind not in dests:
                    continue
                # Chẩn đoán E6: xử lý không AI, nhãn 0, không có đoạn fake.
                is_fake = kind in {"full", "partial"} and family != "diag"
                diagnostic = family == "diag" and kind == "full"
                if diagnostic:
                    intervals = []
                encode(frames, pcm, dests[kind], plan["config"]["crf"])
                if abs(probe(dests[kind])["duration_s"] - n / 25) > 0.1:
                    raise ValueError("Encoded duration differs")
                generated.append(
                    {
                        "sample_id": f"{key}_{kind}",
                        "video": dests[kind].relative_to(out).as_posix(),
                        "sha256": sha(dests[kind]),
                        "duration_s": n / 25,
                        "label": int(is_fake),
                        "fake_intervals": intervals,
                        "split": r["split"],
                        "source_id": r["source_id"],
                        "source_clip_id": r["clip_id"],
                        "speaker_id": r.get("speaker_id"),
                        **{
                            field: r[field]
                            for field in (
                                "global_speaker_ids",
                                "source_sha256",
                                "duplicate_group_id",
                            )
                            if r.get(field)
                        },
                        "audio_source_id": job["sham_donor"]["source_id"]
                        if kind == "sham"
                        else d["source_id"]
                        if kind != "real"
                        else r["source_id"],
                        "parent_ids": sorted(
                            {
                                r["source_id"],
                                d["source_id"],
                                *(([job["sham_donor"]["source_id"]]) if kind == "sham" else []),
                            }
                        ),
                        "speaker_ids": list(
                            filter(None, {r.get("speaker_id"), d.get("speaker_id")})
                        ),
                        "generator": gen_info["generator"] if is_fake else "none",
                        "generator_version": gen_info["version"]
                        if is_fake
                        else "matched-encode-v2",
                        "synthetic_audio": False,
                        "synthetic_visual": is_fake,
                        "audio_fake_intervals": [],
                        "visual_fake_intervals": intervals,
                        "audio_mode": mode if is_fake else None,
                        # Lệch tiếng-miệng đã biết: sham = đoạn ghép; fake chưa đo được nên null.
                        "av_mismatch_intervals": spans
                        if kind == "sham"
                        else []
                        if kind == "real"
                        else None,
                        "audio_edit_kind": "splice"
                        if kind == "sham"
                        else "none"
                        if mode == "source" or kind == "real"
                        else "donor_replace"
                        if kind == "full"
                        else "splice",
                        "processing_profile": "common-pcm48k-h264-aac-crf"
                        + str(plan["config"]["crf"]),
                        "control_type": "conventional_audio_splice" if kind == "sham" else None,
                        "sync_status": "reviewed_match" if kind == "real" else "unknown",
                        "generator_checkpoint_sha256": gen_info.get("checkpoint_sha256")
                        if is_fake
                        else None,
                        "variant": kind,
                        "review_status": "pending",
                        "group_id": key,
                    }
                )
                row = generated[-1]
                if family != "ai" and (is_fake or diagnostic):
                    row["generator"] = name
                if diagnostic:
                    row.update(
                        generator_version=gen_info["version"],
                        variant="diagnostic",
                        diagnostic=name.removeprefix("diag_"),
                        av_mismatch_intervals=[],
                    )
                if family == "mech":
                    row.update(
                        # Chỉ M5 làm miệng lệch tiếng; cơ chế khác giữ đúng hình dạng miệng.
                        av_mismatch_intervals=intervals if name == "mech_temporal" else [],
                        feature_scope="full" if job.get("fusion_subset") else "artifact",
                        box_sample_id=job["real_sample_id"],
                    )
            write_json(record, generated)
        finally:
            slots.put(gpu)
        with guard:
            progress["done"] += 1
            progress["streak"] = 0
            done, spent = progress["done"], time.monotonic() - began
            seconds[name].append(time.monotonic() - tick)
        print(
            f"{head}, donor {d['clip_id'] if mode == 'donor' else '(tiếng gốc)'}; "
            f"{'/'.join(dests)}; partial {length / 25:.2f}s tại {a / 25:.2f}s | "
            f"{time.monotonic() - tick:.0f}s{'' if gpu is None else f' (GPU {gpu})'}, "
            f"mới {done}/{pending}, đã chạy {spent / 60:.1f} phút, "
            f"còn ~{spent / done * (pending - done) / 60:.1f} phút",
            flush=True,
        )
        return generated

    results = [None] * total
    try:
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = {pool.submit(make_pair, i, job): i for i, job in enumerate(plan["jobs"])}
            try:
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
            except BaseException:
                # Một cặp lỗi: không nhận cặp mới, chờ cặp đang chạy xong rồi báo lỗi.
                for future in futures:
                    future.cancel()
                raise
    finally:
        worker.close_all()
        # Chi phí thật của phiên: dùng cho bảng chi phí/dự tính GPU của báo cáo.
        run_log["sessions"].append(
            {
                "started_unix": round(time.time() - (time.monotonic() - began)),
                "elapsed_s": round(time.monotonic() - began, 1),
                "gpus": len(gpus),
                "generators": sorted(active),
                "pairs_done": progress["done"],
                "pairs_skipped": progress["skipped"],
                "seconds_per_pair": {
                    k: round(sum(v) / len(v), 1) for k, v in sorted(seconds.items())
                },
            }
        )
        write_json(lock, run_log)
    missing = sum(result is None for result in results)
    skipped = [
        record
        for record in map(read_json, sorted((out / "records").glob("*.json")))
        if isinstance(record, dict)
    ]
    if skipped:
        reasons = Counter(record["skipped"] for record in skipped)
        print(f"Bỏ qua {len(skipped)}/{total} cặp: {dict(reasons)}", flush=True)
    if missing:
        # Chưa đủ cặp: không ghi candidates/review (review.csv chỉ tạo một lần khi đủ).
        left = Counter(
            job.get("generator", "wav2lip_gan")
            for job, result in zip(plan["jobs"], results)
            if result is None
        )
        return {
            "status": "partial",
            "pairs_done": total - missing,
            "pairs_remaining": missing,
            "remaining_by_generator": dict(left),
            "pairs_skipped": len(skipped),
            "next": "Save Version, attach output này làm PREVIOUS_GENERATION, chọn GENERATORS "
            "theo remaining_by_generator rồi chạy tiếp",
        }
    rows = [row for result in results for row in result]
    # MechFake lấy hộp miệng từ real của parent: cặp AI của parent bị bỏ thì bỏ luôn MechFake.
    produced = {row["sample_id"] for row in rows}
    orphans = [r for r in rows if r.get("box_sample_id") and r["box_sample_id"] not in produced]
    if orphans:
        print(f"Bỏ {len(orphans)} mẫu MechFake vì parent không có real (cặp AI bị bỏ)", flush=True)
        rows = [r for r in rows if r not in orphans]
    write_manifest(out / "candidates.jsonl", rows)
    write_json(out / "skipped_pairs.json", skipped)
    if not (out / "review.csv").exists():
        # Mặc định keep: review chỉ để đánh reject/uncertain mẫu lỗi.
        csv_write(
            out / "review.csv", [{"sample_id": r["sample_id"], "decision": "keep"} for r in rows]
        )
    return {
        "status": "complete",
        "samples": len(rows),
        "pairs_skipped": len(skipped),
        "next": "Run review then finalize; pending media is not training data",
    }


def finalize(cfg):
    """Chỉ keep được đi vào dataset; chốt xong không sửa nhãn/split tại chỗ."""
    root = Path(cfg["generated_dataset"])
    rows = read_manifest(root / "candidates.jsonl")
    with (root / "review.csv").open(encoding="utf-8-sig", newline="") as f:
        decisions = list(csv.DictReader(f))
    if len({r["sample_id"] for r in decisions}) != len(decisions) or {
        r["sample_id"] for r in decisions
    } != {r["sample_id"] for r in rows}:
        raise ValueError("Review IDs differ from candidates")
    review = {r["sample_id"]: r["decision"] for r in decisions}
    if set(review.values()) - {"keep", "reject", "uncertain", "pending"}:
        raise ValueError("Invalid review decision")
    kept = [{**r, "review_status": "keep"} for r in rows if review[r["sample_id"]] == "keep"]
    validate_rows(kept, root, verify_media=True)
    generation = read_json(root / "generation.json")
    # Part kế thừa split của part trước có thể chỉ thuộc train; kiểm đủ ở dataset gộp.
    plan = generation["plan"]
    covered = plan.get("split", {}).get("covered_parts", [1])
    history = bool(plan.get("prior_assignments")) or len(covered) > 1
    splits = {r["split"] for r in kept} if history else {"train", "validation", "test"}
    for split in splits:
        if {r["label"] for r in kept if r["split"] == split} != {0, 1}:
            raise ValueError(f"Review needs real and fake in {split}")
    manifest = root / "manifest.jsonl"
    if manifest.exists():
        if read_manifest(manifest) == kept:
            return read_json(root / "dataset_info.json")
        raise FileExistsError("Finalized dataset immutable; use a new dataset version")
    write_manifest(manifest, kept)
    info = {
        "schema_version": SCHEMA,
        "samples": len(kept),
        "manifest_sha256": sha(manifest),
        "review_sha256": sha(root / "review.csv"),
    }
    write_json(root / "dataset_info.json", info)
    return info
