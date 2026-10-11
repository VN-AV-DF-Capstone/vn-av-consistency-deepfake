from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vn_av_df.common.runtime import read_json, sha, write_json
from vn_av_df.data.groups import write_manifest
from vn_av_df.data.media import decode, probe
from vn_av_df.data.render import encode
from vn_av_df.dataset import load_dataset
from vn_av_df.generation import finalize, generate, make_plan
from vn_av_df.web import demo_app, review_app

# Tỷ lệ của fixture giữ 80/10/10 như khi các test này được viết.
FIXTURE_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}


def seal_clean(root, rows, part=1, history=()):
    """Như 05_export: gán split theo nhóm, ghi split-lock và dataset_info của part sạch."""
    from vn_av_data.data.split import assign_splits

    rows, registry = assign_splits(rows, part, 42, FIXTURE_RATIOS, history)
    write_manifest(root / "manifest.jsonl", rows)
    write_json(root / "split-lock.json", registry)
    write_json(
        root / "dataset_info.json",
        {
            "schema_version": "vn-av-dataset-v1",
            "dataset_id": f"fixture_part{part}",
            "clips": len(rows),
            "manifest_sha256": sha(root / "manifest.jsonl"),
            "split": {k: registry[k] for k in ("data_part", "covered_parts", "counts")},
        },
    )
    return rows


def clean_bundle(root, identities):
    """Bundle sạch fixture: mỗi clip một màu/tần số riêng; identities = [(source, speaker)]."""
    rows = []
    for i, (source, speaker) in enumerate(identities):
        frames = np.full((130, 48, 64, 3), 20 + i * 25, np.uint8)
        pcm = (0.15 * np.sin(np.arange(130 * 1920) * 2 * np.pi * (300 + i * 70) / 48000)).astype(
            "float32"
        )
        path = root / "clips" / f"c{i}.mp4"
        encode(frames, pcm, path, 18)
        duration = probe(path)["duration_s"]
        rows.append(
            {
                "clip_id": f"c{i}",
                "video": f"clips/c{i}.mp4",
                "sha256": sha(path),
                "source_id": source,
                "speaker_id": speaker,
                "duration_s": duration,
                "source_start_s": 0,
                "source_end_s": duration,
                "review_decision": "keep",
                "sync_status": "reviewed_match",
                "relation_annotations": {},
            }
        )
    return seal_clean(root, rows)


def paint_generator(cfg, video, audio, output, n, w, h):
    """Generator fixture: vẽ một khối trắng; không chứng minh generator thật chạy được."""
    d = decode(video, max_side=640, sample_rate=48000)
    d["frames"][:, 15:35, 15:45] = 255
    encode(d["frames"], d["pcm"], output, 18)


def test_donor_source_design_labels_audio_and_skip(tmp_path):
    from vn_av_df.dataset import condition
    from vn_av_df.generation import csv_write

    root = tmp_path / "clean"
    # c6 cùng nguồn với c0 nhưng không có clip cùng người: không có donor → bị bỏ qua.
    identities = [(f"s{i}", f"p{i // 2}") for i in range(6)] + [("s0", "p9")]
    clean_bundle(root, identities)
    cfg = {
        "seed": 42,
        "clean_dataset": str(root),
        "plan": str(tmp_path / "plan.json"),
        "generated_dataset": str(tmp_path / "generated"),
        "generation": {
            "clips_per_split": 0,
            "partial_range_s": [0.4, 0.4],
            "crf": 18,
            "max_side": 640,
            "include_sham": True,
            "fake_audio_modes_by_split": {
                "train": ["donor"],
                "validation": ["donor", "source"],
                "test": ["donor", "source"],
            },
        },
    }
    assert make_plan(cfg)["skipped_without_donor"] == 1
    plan = read_json(cfg["plan"])
    assert plan["skipped_without_donor"] == ["c6"]
    for job in plan["jobs"]:
        expected = ["donor"] if job["original"]["split"] == "train" else ["donor", "source"]
        assert job["audio_mode"] in expected
        if job["audio_mode"] == "source":
            assert job["donor"] == job["original"] and not job["emit_real"]
        else:
            assert job["donor"]["speaker_id"] == job["original"]["speaker_id"]
            assert job["donor"]["clip_id"] != job["original"]["clip_id"]
    generate(cfg, paint_generator)
    out = Path(cfg["generated_dataset"])
    candidates = [r["sample_id"] for r in load_candidates(out)]
    csv_write(out / "review.csv", [{"sample_id": s, "decision": "keep"} for s in candidates])
    finalize(cfg)
    bundle = load_dataset(out, verify_media=True)
    cells = {condition(r) for r in bundle}
    assert cells == {"real", "sham", "test_fixture/donor", "test_fixture/source"}
    assert all(r["audio_mode"] == "donor" for r in bundle if r["split"] == "train" and r["label"])
    for r in bundle:
        if r["variant"] == "real":
            assert r["av_mismatch_intervals"] == [] and r["audio_mode"] is None
        elif r["variant"] == "sham":
            assert r["label"] == 0 and r["av_mismatch_intervals"][0][1] - r[
                "av_mismatch_intervals"
            ][0][0] == pytest.approx(0.4)
        else:
            assert r["av_mismatch_intervals"] is None
    # donor thay tiếng (thật, cùng người); source giữ đúng PCM gốc.
    pcm = {r["sample_id"]: decode(out / r["video"])["pcm"] for r in bundle}
    for r in bundle:
        if r["variant"] != "full":
            continue
        real = next(
            x
            for x in bundle
            if x["variant"] == "real" and x["source_clip_id"] == r["source_clip_id"]
        )
        same = np.allclose(pcm[r["sample_id"]], pcm[real["sample_id"]], atol=1e-4)
        assert same == (r["audio_mode"] == "source")


def test_mechfake_and_diagnostic_rows_and_random_partials(tmp_path):
    from vn_av_df.common.runtime import fingerprint
    from vn_av_df.dataset import condition

    root = tmp_path / "clean"
    clean_bundle(root, [(f"s{i}", f"p{i // 2}") for i in range(6)])
    mech = ["mech_blend", "mech_vae"]
    cfg = {
        "seed": 42,
        "clean_dataset": str(root),
        "plan": str(tmp_path / "plan.json"),
        "generated_dataset": str(tmp_path / "generated"),
        "generation": {
            "clips_per_split": 1,
            "partial_range_s": [0.2, 3.0],
            "crf": 18,
            "max_side": 640,
            "include_sham": True,
            "generators_by_split": {
                "train": ["wav2lip_gan", *mech],
                "validation": ["wav2lip_gan", *mech],
                "test": ["wav2lip_gan", "diag_paste"],
            },
        },
    }
    with pytest.raises(ValueError, match="First generator"):
        make_plan(
            {**cfg, "generation": {**cfg["generation"], "generators_by_split": {"train": mech}}}
        )
    make_plan(cfg)
    jobs = read_json(cfg["plan"])["jobs"]
    for parent in {j["original"]["clip_id"] for j in jobs}:
        mine = [j for j in jobs if j["original"]["clip_id"] == parent]
        real = fingerprint(mine[0])[:20] + "_real"
        assert mine[0]["emit_real"] and all(j["real_sample_id"] == real for j in mine[1:])
        flags = [j["fusion_subset"] for j in mine if j["generator"] in mech]
        assert flags == [] or sum(flags) == 1  # Đúng một cơ chế mỗi parent được trích đủ.
    generate(cfg, paint_generator)
    finalize(cfg)
    bundle = load_dataset(cfg["generated_dataset"], verify_media=True)
    ids = {r["sample_id"]: r for r in bundle}
    fakes = [r for r in bundle if r["generator"] in mech]
    assert fakes and all(r["label"] == 1 and r["audio_mode"] == "source" for r in fakes)
    assert all(ids[r["box_sample_id"]]["variant"] == "real" for r in fakes)
    assert {r["feature_scope"] for r in fakes} == {"full", "artifact"}
    assert all(r["split"] != "test" for r in fakes)
    diagnostics = [r for r in bundle if r.get("diagnostic")]
    assert diagnostics and all(
        r["label"] == 0
        and r["split"] == "test"
        and r["fake_intervals"] == []
        and condition(r) == "diag/paste"
        for r in diagnostics
    )
    lengths = [b - a for r in bundle if r["variant"] == "partial" for a, b in r["fake_intervals"]]
    assert lengths and all(0.2 - 1e-6 <= x <= 3.0 + 1e-6 for x in lengths)


def test_generation_stops_at_deadline_and_resumes_in_parallel(tmp_path):
    import time

    from vn_av_df.common.runtime import fingerprint

    root = tmp_path / "clean"
    clean_bundle(root, [(f"s{i}", f"p{i // 2}") for i in range(6)])
    cfg = {
        "seed": 42,
        "clean_dataset": str(root),
        "plan": str(tmp_path / "plan.json"),
        "generated_dataset": str(tmp_path / "generated"),
        "generation": {
            "clips_per_split": 1,
            "partial_range_s": [0.4, 0.4],
            "crf": 18,
            "max_side": 640,
        },
    }
    make_plan(cfg)
    jobs = read_json(cfg["plan"])["jobs"]
    out = Path(cfg["generated_dataset"])
    # Hết giờ trước khi bắt đầu: không cặp nào, không ghi candidates/review.
    partial = generate({**cfg, "generation_deadline": time.time() - 1}, paint_generator)
    assert partial["status"] == "partial" and partial["pairs_remaining"] == len(jobs)
    assert not (out / "candidates.jsonl").exists() and not (out / "review.csv").exists()
    # Phiên trước bị ngắt giữa cặp: video dở (chưa có record) và thư mục tạm còn sót.
    key = fingerprint(jobs[0])[:20]
    (out / "clips").mkdir(parents=True, exist_ok=True)
    (out / "clips" / f"{key}_full.mp4").write_bytes(b"incomplete")
    (out / "tmpleftover").mkdir()
    # Phiên sau: chạy tiếp, 2 slot song song, đủ mọi cặp theo đúng thứ tự plan.
    done = generate({**cfg, "generation_gpus": [0, 1]}, paint_generator)
    assert done["status"] == "complete" and done["samples"] == 9
    assert not (out / "tmpleftover").exists()
    rows = load_candidates(out)
    assert [r["group_id"] for r in rows[::3]] == [fingerprint(j)[:20] for j in jobs]
    assert all(sha(out / r["video"]) == r["sha256"] for r in rows)


def test_generation_skips_pairs_without_face_but_stops_on_other_errors(tmp_path):
    from vn_av_df.common.runtime import fingerprint

    root = tmp_path / "clean"
    clean_bundle(root, [(f"s{i}", f"p{i // 2}") for i in range(6)])
    cfg = {
        "seed": 42,
        "clean_dataset": str(root),
        "plan": str(tmp_path / "plan.json"),
        "generated_dataset": str(tmp_path / "generated"),
        "generation": {
            "clips_per_split": 1,
            "partial_range_s": [0.4, 0.4],
            "crf": 18,
            "max_side": 640,
        },
    }
    make_plan(cfg)
    jobs = read_json(cfg["plan"])["jobs"]
    first = Path(jobs[0]["original"]["video"]).name
    out = Path(cfg["generated_dataset"])

    def broken(cfg, video, audio, output, n, w, h):
        if Path(video).name == first:
            raise RuntimeError("Worker exited (1): crashed")
        return paint_generator(cfg, video, audio, output, n, w, h)

    # Worker chết (không phải lỗi riêng một cặp): vẫn dừng như cũ.
    with pytest.raises(RuntimeError, match="Worker exited"):
        generate(cfg, broken)

    def faceless(cfg, video, audio, output, n, w, h):
        if Path(video).name == first:
            raise RuntimeError("Worker failed: ValueError: Face not detected! Ensure ...")
        return paint_generator(cfg, video, audio, output, n, w, h)

    done = generate(cfg, faceless)
    assert done["status"] == "complete" and done["pairs_skipped"] == 1
    key = fingerprint(jobs[0])[:20]
    # Cặp bị bỏ không có video nào (kể cả real); các cặp khác đủ.
    assert not list((out / "clips").glob(f"{key}_*"))
    rows = load_candidates(out)
    assert key not in {r["group_id"] for r in rows} and len(rows) == 3 * (len(jobs) - 1)
    skipped = read_json(out / "skipped_pairs.json")
    assert [s["skipped"] for s in skipped] == ["Face not detected"]
    assert skipped[0]["parent_clip_id"] == jobs[0]["original"]["clip_id"]
    # Chạy lại: cặp đã bỏ không sinh lại, kết quả giữ nguyên.
    again = generate(cfg, paint_generator)
    assert again["pairs_skipped"] == 1 and load_candidates(out) == rows


def test_generation_skips_generator_errors_but_stops_on_a_failing_streak(tmp_path, monkeypatch):
    from vn_av_df import generation

    root = tmp_path / "clean"
    clean_bundle(root, [(f"s{i}", f"p{i // 2}") for i in range(6)])
    cfg = {
        "seed": 42,
        "clean_dataset": str(root),
        "plan": str(tmp_path / "plan.json"),
        "generated_dataset": str(tmp_path / "generated"),
        "generation": {
            "clips_per_split": 1,
            "partial_range_s": [0.4, 0.4],
            "crf": 18,
            "max_side": 640,
        },
    }
    make_plan(cfg)
    out = Path(cfg["generated_dataset"])

    def whisper(cfg, video, audio, output, n, w, h):
        raise RuntimeError("Worker failed: ValueError: Whisper produced fewer frames\nTraceback")

    # Mọi cặp lỗi generator liên tiếp: đủ ngưỡng thì dừng (lỗi hệ thống, không phải clip khó).
    monkeypatch.setattr(generation, "MAX_SKIP_STREAK", 2)
    with pytest.raises(RuntimeError, match="2 cặp liên tiếp"):
        generate(cfg, whisper)
    # Một cặp lỗi generator (output mới): bỏ cặp đó, vẫn hoàn tất; lý do lấy dòng đầu của lỗi.
    monkeypatch.setattr(generation, "MAX_SKIP_STREAK", 99)
    out = tmp_path / "generated_one"
    first = Path(read_json(cfg["plan"])["jobs"][0]["original"]["video"]).name

    def whisper_once(cfg, video, audio, output, n, w, h):
        if Path(video).name == first:
            whisper(cfg, video, audio, output, n, w, h)
        return paint_generator(cfg, video, audio, output, n, w, h)

    done = generate({**cfg, "generated_dataset": str(out)}, whisper_once)
    assert done["status"] == "complete" and done["pairs_skipped"] == 1 and done["samples"] == 6
    skipped = read_json(out / "skipped_pairs.json")
    assert {(s["kind"], s["skipped"]) for s in skipped} == {
        ("generator_error", "ValueError: Whisper produced fewer frames")
    }


def load_candidates(folder):
    from vn_av_df.data.groups import read_manifest

    return read_manifest(folder / "candidates.jsonl")


@pytest.mark.parametrize("multigenerator", [False, True])
def test_real_media_generation_review_export_and_resume(tmp_path, multigenerator):
    root = tmp_path / "clean"
    rows = clean_bundle(root, [(f"s{i}", f"p{i // 2}") for i in range(6)])
    cfg = {
        "seed": 42,
        "clean_dataset": str(root),
        "plan": str(tmp_path / "plan.json"),
        "generated_dataset": str(tmp_path / "generated"),
        "generation": {
            "clips_per_split": 1,
            "partial_range_s": [0.4, 0.4],
            "crf": 18,
            "max_side": 640,
        },
    }
    if multigenerator:
        cfg["generation"].update(
            include_sham=True,
            audio_mode="source",
            generators_by_split={
                "train": ["wav2lip_gan"],
                "validation": ["wav2lip_gan"],
                "test": ["wav2lip_gan", "musetalk_1_5"],
            },
        )
    expected = 14 if multigenerator else 9
    assert make_plan(cfg)["videos"] == expected
    jobs = read_json(cfg["plan"])["jobs"]
    assert all(j["original"]["split"] == "test" for j in jobs if j["generator"] == "musetalk_1_5")

    generator = paint_generator
    assert generate(cfg, generator)["samples"] == expected
    with TestClient(review_app(cfg)) as client:
        items = client.get("/items").json()
        # Mặc định keep: review chỉ đánh mẫu lỗi; đổi qua lại vẫn lưu đúng.
        assert {item["decision"] for item in items} == {"keep"}
        assert client.get("/media/" + items[0]["sample_id"]).status_code == 200
        for decision in ("reject", "keep"):
            sid = items[0]["sample_id"]
            assert client.post("/decision/" + sid, json={"decision": decision}).status_code == 200
    first = finalize(cfg)
    assert first["samples"] == expected
    bundle = load_dataset(cfg["generated_dataset"], verify_media=True)
    assert {r["label"] for r in bundle} == {0, 1}
    assert all(
        r["fake_intervals"][0][1] - r["fake_intervals"][0][0] == pytest.approx(0.4)
        for r in bundle
        if r["variant"] == "partial"
    )
    if multigenerator:
        assert sum(r["variant"] == "real" for r in bundle) == 3
        assert sum(r["variant"] == "sham" for r in bundle) == 3
        assert all(r["synthetic_audio"] is False for r in bundle)
        assert all(
            r["label"] == 0 and r["fake_intervals"] == [] for r in bundle if r["variant"] == "sham"
        )
        # Cùng PCM cho real/full/partial: không để resampling audio chỉ xuất hiện ở fake.
        real = next(r for r in bundle if r["variant"] == "real")
        original_pcm = decode(Path(cfg["generated_dataset"]) / real["video"])["pcm"]
        for row in bundle:
            if row["source_clip_id"] == real["source_clip_id"] and row["variant"] in {
                "full",
                "partial",
            }:
                np.testing.assert_allclose(
                    decode(Path(cfg["generated_dataset"]) / row["video"])["pcm"], original_pcm
                )
    assert generate(cfg, generator)["samples"] == expected
    assert finalize(cfg) == first
    with TestClient(review_app(cfg)) as client:
        assert (
            client.post(
                "/decision/" + bundle[0]["sample_id"], json={"decision": "reject"}
            ).status_code
            == 409
        )

    clean_split = {r["clip_id"]: r["split"] for r in rows}
    if not multigenerator:
        # Part 2 only needs its new media and part 1's cumulative JSON split registry.
        from vn_av_df.dataset import training_dataset

        root2 = tmp_path / "clean2"
        d = decode(root / rows[0]["video"], max_side=640, sample_rate=48000)
        d["frames"][:, :8, :8] = 150
        new_video = root2 / "clips/c6.mp4"
        encode(d["frames"], d["pcm"], new_video, 18)
        added = {
            **rows[0],
            "clip_id": "c6",
            "source_id": "s6",
            "video": "clips/c6.mp4",
            "sha256": sha(new_video),
            "duration_s": probe(new_video)["duration_s"],
        }
        added.pop("split")
        # Part 2 kế thừa split part 1 qua split-lock của part sạch (bật ở 05_export).
        seal_clean(root2, [added], part=2, history=[root / "split-lock.json"])
        cfg2 = {
            **cfg,
            "data_part": 2,
            "clean_dataset": str(root2),
            "generated_dataset": str(tmp_path / "generated2"),
            "plan": str(tmp_path / "plan2.json"),
        }
        assert make_plan(cfg2)["videos"] == 3
        assert generate(cfg2, generator)["samples"] == 3
        with TestClient(review_app(cfg2)) as client:
            for item in client.get("/items").json():
                assert (
                    client.post(
                        "/decision/" + item["sample_id"], json={"decision": "keep"}
                    ).status_code
                    == 200
                )
        assert finalize(cfg2)["samples"] == 3
        second = load_dataset(cfg2["generated_dataset"], verify_media=True)
        assert {r["split"] for r in second} == {clean_split["c0"]}
        combined, receipt = training_dataset(
            {
                "dataset": str(tmp_path),
                "dataset_parts": ["generated", "generated2"],
            },
            verify_media=True,
        )
        assert len(combined) == expected + 3 and len(receipt["parts"]) == 2
        history2 = read_json(Path(cfg2["generated_dataset"]) / "split-lock.json")
        assert history2["data_part"] == 2 and len(history2["assignments"]) == 7


def test_demo_public_result_has_only_scores_and_intervals(tmp_path):
    import time

    class Fixture:
        def __init__(self, cfg):
            assert cfg["demo_details"] is True  # Frontend cần điểm theo ô để vẽ timeline.

        def analyze(self, path, output):
            result = {
                "video_score": 0.8,
                "intervals": [{"start_sec": 1.0, "end_sec": 2.0, "score": 0.9}],
            }
            write_json(output, result)
            return result

    with TestClient(demo_app({"demo_output": str(tmp_path)}, Fixture)) as client:
        assert client.get("/api/health").json()["ready"] is False
        assert client.get("/api/research").json()["runs"] is None
        job = client.post("/api/jobs", files={"video": ("clip.mp4", b"fixture")}).json()
        for _ in range(100):
            state = client.get("/api/jobs/" + job["id"]).json()
            if state["status"] == "complete":
                break
            time.sleep(0.01)
        assert set(state["result"]) == {"video_score", "intervals"}
        assert client.get("/api/jobs/" + job["id"] + "/report").json() == state["result"]
