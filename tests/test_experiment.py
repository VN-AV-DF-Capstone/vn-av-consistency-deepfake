from pathlib import Path

import numpy as np
import pytest
import torch

from vn_av_df.common.runtime import fingerprint, read_json, save_npz, sha, write_json
from vn_av_df.data.groups import write_manifest
from vn_av_df.dataset import SCHEMA
from vn_av_df.experiment import evaluate, load_model, train_one, train_selected
from vn_av_df.models import TemporalDetector


def test_masks_block_temporal_leakage_and_gradients():
    torch.manual_seed(2)
    model = TemporalDetector(8, 8, "tcn", hidden=8, dropout=0).eval()
    a = torch.randn(12, 8)
    v = torch.randn(12, 8)
    valid = torch.ones(12, dtype=torch.bool)
    valid[5] = False
    one = model(a, v, valid)
    a2 = a.clone()
    a2[6:] += 100
    two = model(a2, v, valid)
    assert torch.allclose(one[:5], two[:5])
    one[valid].sum().backward()
    assert model.classifier.weight.grad.abs().sum() > 0
    assert torch.isfinite(model(a, v, torch.zeros_like(valid))).all()


@pytest.fixture
def project(tmp_path):
    rows = []
    cache = tmp_path / "cache"
    dataset = tmp_path / "dataset"
    rng = np.random.default_rng(4)
    for split in ("train", "validation", "test"):
        for label in (0, 1):
            sid = f"{split}_{label}"
            digest = fingerprint(sid)
            row = {
                "sample_id": sid,
                "video": sid + ".mp4",
                "sha256": digest,
                "duration_s": 4.0,
                "label": label,
                "fake_intervals": [[1, 3]] if label else [],
                "split": split,
                "source_id": split,
                "speaker_id": split,
                "review_status": "keep",
                "generator": "wav2lip_gan" if label else "none",
                "generator_version": "fixture",
            }
            rows.append(row)
            path = cache / (sid + ".npz")
            save_npz(
                path,
                audio=rng.normal(size=(20, 8)).astype("float32"),
                visual=rng.normal(size=(20, 8)).astype("float32"),
                times_s=np.arange(20) * 0.2,
                audio_valid=np.ones(20, bool),
                visual_valid=np.ones(20, bool),
            )
            write_json(
                path.with_suffix(".json"),
                {
                    "feature_sha256": sha(path),
                    "feature_signature": "test-only",
                    "source_fingerprint": fingerprint(
                        {"assets": {"video": digest}, "variant": {"kind": "clean"}}
                    ),
                    "step_s": 0.2,
                    "window_s": 2.0,
                    "duration_s": 4.0,
                },
            )
    write_manifest(dataset / "manifest.jsonl", rows)
    write_json(
        dataset / "dataset_info.json",
        {
            "schema_version": SCHEMA,
            "samples": len(rows),
            "manifest_sha256": sha(dataset / "manifest.jsonl"),
        },
    )
    return {
        "dataset": str(dataset),
        "cache": str(cache),
        "runs": str(tmp_path / "runs"),
        "device": "cpu",
        "encoder": {},
        "architectures": ["tcn"],
        "seeds": [42, 43],
        "held_out_generators": [],
        "training": {
            "epochs": 1,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "hidden": 8,
            "dropout": 0.0,
            "patience": 0,
            "top_fraction": 0.2,
        },
    }, rows


def test_train_test_thresholds_lock_and_binary_output(project, tmp_path):
    cfg, rows = project
    report = train_selected(cfg)
    assert len(report["runs"]) == 2
    before = {r["checkpoint"]: load_model(r["checkpoint"])[1]["thresholds"] for r in report["runs"]}
    results = evaluate(cfg)
    assert len(results) == 2
    assert all("by_generator" in r for r in results)
    assert before == {p: load_model(p)[1]["thresholds"] for p in before}
    assert read_json(Path(cfg["runs"]) / "evaluation.json")["runs"][0]["architecture"] == "tcn"
    index_path = Path(cfg["runs"]) / "training.json"
    write_json(index_path, {**report, "note": "changed after test"})
    with pytest.raises(ValueError, match="Test protocol changed"):
        evaluate(cfg)
    write_json(index_path, report)
    cfg["training"]["epochs"] = 2
    train_one(cfg, "tcn", 42, resume=True)
    assert len(read_json(Path(cfg["runs"]) / "tcn_seed42/history.json")) == 2
    from vn_av_df.inference import Analyzer

    class FixtureEncoder:
        signature = "test-only"

        def extract(self, row, output):
            source = Path(cfg["cache"]) / "test_0.npz"
            Path(output).write_bytes(source.read_bytes())
            return {"step_s": 0.2, "duration_s": 4.0}

    cfg["demo_method"] = "tcn"
    result = Analyzer(cfg, FixtureEncoder()).analyze("fixture.mp4")
    assert set(result) == {"video_score", "intervals"}
    assert 0 <= result["video_score"] <= 1
    cfg["training"]["hidden"] = 12
    with pytest.raises(ValueError, match="Resume"):
        train_one(cfg, "tcn", 42, resume=True)


def test_fast_threshold_matches_exhaustive_search():
    from vn_av_df.metrics import binary_metrics, choose_threshold

    rng = np.random.default_rng(3)
    for _ in range(25):
        n = int(rng.integers(4, 300))
        labels = rng.integers(0, 2, n)
        labels[:2] = [0, 1]
        scores = np.round(rng.random(n), int(rng.integers(1, 4)))  # Có nhiều điểm hòa.
        candidates = np.unique(np.r_[0.5, scores, np.nextafter(scores, np.inf)])
        candidates = candidates[(candidates >= 0) & (candidates <= 1)]

        def key(t):
            m = binary_metrics(labels, scores, t)
            return m["f1"], -m["false_alarm_rate"]

        best = max(key(t) for t in candidates)
        chosen = choose_threshold(labels.tolist(), scores.tolist())
        assert key(chosen) == pytest.approx(best)
        assert chosen == max(t for t in candidates if key(t) == pytest.approx(best))


def test_resume_after_early_stop_keeps_selected_checkpoint(project):
    cfg, _ = project
    cfg["training"].update(epochs=2, patience=3)
    best = train_one(cfg, "tcn", 42)
    state = torch.load(best.parent / "last.pt", weights_only=True)
    state["stale"] = 3  # Như thể phiên trước đã early stop.
    torch.save(state, best.parent / "last.pt")
    before = sha(best)
    cfg["training"]["epochs"] = 5
    train_one(cfg, "tcn", 42, resume=True)
    assert len(read_json(best.parent / "history.json")) == 2
    assert sha(best) == before


def test_training_never_reads_test_cache(project):
    cfg, rows = project
    for r in rows:
        if r["split"] == "test":
            (Path(cfg["cache"]) / (r["sample_id"] + ".npz")).unlink()
    train_one(cfg, "tcn", 42)


@pytest.mark.parametrize("methods", [["tcn"]])
def test_selected_training_reports_without_automatic_comparison(project, methods, monkeypatch):
    from vn_av_df.actions import execute
    from vn_av_df.inference import chosen_checkpoint

    cfg, rows = project
    cfg["architectures"] = methods
    cfg["seeds"] = [42]
    execute("train", cfg)
    folder = Path(cfg["runs"])
    index = read_json(folder / "training.json")
    assert index["status"] == "complete"
    assert [r["architecture"] for r in index["runs"]] == methods
    assert not (folder / "test-lock.json").exists()
    for name in methods:
        detector = folder / f"{name}_seed42"
        predictions = read_json(detector / "predictions-validation.json")
        assert {p["sample_id"] for p in predictions} == {
            r["sample_id"] for r in rows if r["split"] == "validation"
        }
        assert all("label" in p and "fake_intervals" in p for p in predictions)
        assert read_json(detector / "best.json")["epoch"] == 1
    cfg["demo_method"] = methods[0]
    assert chosen_checkpoint(cfg) == folder / f"{methods[0]}_seed42/best.pt"
    if len(methods) == 1:
        monkeypatch.setenv("MPLBACKEND", "Agg")
        from vn_av_df.reporting import training_reports

        reports = training_reports(folder)
        assert len(reports) == 1 and reports[0]["summary"]["split"] == "validation"
        assert all(Path(p).read_bytes().startswith(b"\x89PNG") for p in reports[0]["images"])
        assert not (folder / "test-lock.json").exists()  # Rendering never calls test.
        evaluate(cfg)
        assert (folder / "evaluation.json").exists()
        assert training_reports(folder, split="test")[0]["summary"]["split"] == "test"


def test_training_selection_cannot_silently_overwrite_run_index(project):
    from vn_av_df.experiment import train_selected

    cfg, _ = project
    cfg["architectures"], cfg["seeds"] = ["tcn"], [42]
    train_selected(cfg)
    cfg["seeds"] = [42, 43]
    with pytest.raises(ValueError, match="selection changed"):
        train_selected(cfg, resume=True)


def test_train_part1_then_union_requires_new_run_and_records_parts(project):
    """An actual tiny CPU training run verifies selection reaches every model/test path."""
    cfg, rows = project
    from vn_av_df.dataset import training_dataset

    root = Path(cfg["dataset"])

    def part(name, items):
        folder = root / name
        write_manifest(folder / "manifest.jsonl", items)
        write_json(
            folder / "dataset_info.json",
            {
                "schema_version": SCHEMA,
                "samples": len(items),
                "manifest_sha256": sha(folder / "manifest.jsonl"),
            },
        )

    part("vn-av-df-data-part1", rows)
    cfg.update(dataset_parts=["vn-av-df-data-part1"], architectures=["tcn"])
    train_selected(cfg)
    assert "vn-av-df-data-part1" in evaluate(cfg)[0]["by_part"]
    extra = []
    for r in rows:
        sid = "p2_" + r["sample_id"]
        digest = fingerprint(sid)
        extra.append(
            {
                **r,
                "sample_id": sid,
                "sha256": digest,
                "source_id": "p2_" + r["source_id"],
                "speaker_id": "p2_" + r["speaker_id"],
            }
        )
        cache = Path(cfg["cache"])
        target = cache / (sid + ".npz")
        target.write_bytes((cache / (r["sample_id"] + ".npz")).read_bytes())
        meta = read_json(cache / (r["sample_id"] + ".json"))
        meta["source_fingerprint"] = fingerprint(
            {"assets": {"video": digest}, "variant": {"kind": "clean"}}
        )
        write_json(target.with_suffix(".json"), meta)
    part("vn-av-df-data-part2", extra)
    cfg["dataset_parts"] = "all"
    assert len(training_dataset(cfg)[0]) == 12
    with pytest.raises(ValueError, match="Resume"):
        train_one(cfg, "tcn", 42, resume=True)
    with pytest.raises(ValueError, match="Test protocol changed"):
        evaluate(cfg)
    cfg["runs"] += "-parts1-2"
    report = train_selected(cfg)
    assert len(report["dataset_selection"]["parts"]) == 2
    result = evaluate(cfg)[0]
    assert set(result["by_part"]) == {"vn-av-df-data-part1", "vn-av-df-data-part2"}
    assert read_json(Path(cfg["runs"]) / "dataset-selection.json")["samples"] == 12


def test_old_checkpoint_rejected(tmp_path):
    p = tmp_path / "old.pt"
    torch.save({"format": "fate-two-heads-v1"}, p)
    with pytest.raises(ValueError, match="incompatible"):
        load_model(p)


def test_artifact_checkpoint_relocated_from_kaggle(tmp_path):
    from vn_av_df.experiment import resolve_checkpoint

    p = tmp_path / "tcn_seed42/best.pt"
    p.parent.mkdir()
    p.write_bytes(b"path-only")
    assert (
        resolve_checkpoint(
            {"runs": str(tmp_path)}, "/kaggle/working/project/runs/run/tcn_seed42/best.pt"
        )
        == p
    )


def test_native_features_pooled_for_tcn_but_not_native_models(tmp_path):
    from vn_av_df.features.registry import feature_arrays

    path = tmp_path / "cache.npz"
    save_npz(
        path,
        audio=np.ones((13, 4)),
        visual=np.ones((13, 6)),
        times_s=np.arange(13) * 0.04,
        audio_valid=np.array([True] * 12 + [False]),
        visual_valid=np.ones(13, bool),
    )
    arrays, times, meta = feature_arrays(path, {"step_s": 0.04, "output_stride": 5}, "tcn")
    assert arrays[0].shape == (3, 4)
    assert not arrays[2][-1]
    assert meta["step_s"] == 0.2
    arrays, times, meta = feature_arrays(path, {"step_s": 0.04, "output_stride": 5}, "syncartifact")
    assert arrays[0].shape == (13, 4) and len(times) == 3


@pytest.mark.parametrize("devices,copies", [(None, 1), (["cuda:0", "cuda:1"], 2)])
def test_prepare_extracts_every_sample_once_across_devices(tmp_path, monkeypatch, devices, copies):
    from vn_av_df import experiment

    # 3 parent, mỗi parent vài biến thể; một mẫu không có source_clip_id.
    rows = [
        {
            "sample_id": f"p{i // 3}_{i % 3}",
            "video": f"clips/{i}.mp4",
            "source_clip_id": f"p{i // 3}",
        }
        for i in range(9)
    ] + [{"sample_id": "lone", "video": "clips/lone.mp4"}]
    seen, made, closed = [], [], []

    class Encoder:
        signature = "sig"

        def __init__(self, cfg):
            self.device = cfg.get("device")
            made.append(self.device)

        def extract(self, row, output):
            seen.append((Path(output).stem, id(self)))

        def close(self):
            closed.append(id(self))

    monkeypatch.setattr(experiment, "training_dataset", lambda cfg, verify_media: (rows, {}))
    monkeypatch.setattr(experiment, "make_encoder", Encoder)
    monkeypatch.setattr(experiment, "media_path", lambda root, row: Path(root) / row["video"])
    cfg = {
        "dataset": str(tmp_path),
        "cache": str(tmp_path / "cache" / "avhubert"),
        "encoder": {"device": "cuda", "workers_per_device": copies},
        "prepare_devices": devices,
    }
    report = experiment.prepare(cfg)
    # workers_per_device encoder mỗi device; mỗi mẫu trích đúng một lần, không bỏ sót.
    expected = [d for d in (devices or ["cuda"]) for _ in range(copies)]
    assert sorted(made) == sorted(expected) and report["samples"] == len(rows)
    assert sorted(stem for stem, _ in seen) == sorted(r["sample_id"] for r in rows)
    # Cùng parent → cùng encoder (dùng lại landmark); mọi encoder được đóng khi xong.
    for parent in ("p0", "p1", "p2"):
        assert len({e for stem, e in seen if stem.startswith(parent + "_")}) == 1
    assert len(closed) == len(made)
    info = read_json(tmp_path / "cache" / "avhubert" / "extraction.json")
    assert info["feature_signature"] == "sig"


def test_avhubert_reuses_one_worker_and_landmarks_of_identical_frames(tmp_path, monkeypatch):
    from vn_av_df.features import avhubert, avhubert_worker

    # Phía worker: frame giống hệt nhau chỉ dò dlib một lần; frame khác hoặc không đúng một mặt
    # vẫn dò/đánh dấu đúng.
    calls = []

    class Face:
        def part(self, j):
            return type("P", (), {"x": j, "y": 2 * j})()

    def detector(gray, upsample):
        calls.append(int(gray.sum()))
        return [object()] if gray.sum() else []

    state = {
        "detector": detector,
        "predictor": lambda gray, box: Face(),
        "landmarks": __import__("collections").OrderedDict(),
    }
    frame = np.full((4, 4, 3), 7, np.uint8)
    first = avhubert_worker.face_landmarks(state, frame)
    again = avhubert_worker.face_landmarks(state, frame.copy())
    assert first.shape == (68, 2) and np.array_equal(first, again) and len(calls) == 1
    assert avhubert_worker.face_landmarks(state, np.zeros_like(frame)) is None and len(calls) == 2

    # Phía gọi: một worker cho nhiều mẫu, mở lại khi worker chết.
    started, requests = [], []

    class FakeWorker:
        def __init__(self, command):
            started.append(command)
            self.alive = True

        def call(self, request, timeout):
            requests.append(request)
            out = Path(request["output"])
            out.write_bytes(b"x")
            write_json(out.with_suffix(".json"), {"ok": True})

        def close(self):
            self.alive = False

    monkeypatch.setattr(avhubert.worker, "Worker", FakeWorker)
    encoder = avhubert.AVHubertVideoEncoder.__new__(avhubert.AVHubertVideoEncoder)
    encoder.cfg, encoder.signature, encoder.worker = {"python": "py"}, "sig", None
    video = tmp_path / "v.mp4"
    video.write_bytes(b"video")
    for i in range(3):
        assert encoder.extract({"video": str(video)}, tmp_path / f"s{i}.npz") == {"ok": True}
    assert len(started) == 1 and "--serve" in started[0] and len(requests) == 3
    encoder.worker.alive = False
    encoder.extract({"video": str(video)}, tmp_path / "s9.npz")
    assert len(started) == 2
    encoder.close()
    assert encoder.worker is None


@pytest.mark.parametrize(
    "only,expected",
    [(None, ["avhubert", "dinov2"]), (["dinov2"], ["dinov2"]), (["avhubert"], ["avhubert"])],
)
def test_prepare_and_setup_only_touch_selected_encoders(tmp_path, monkeypatch, only, expected):
    from vn_av_df import assets, experiment

    p2 = {"encoder": "avhubert", "artifact_encoder": "dinov2"}
    cfg = {
        "cache": str(tmp_path / "cache"),
        "encoder": {},
        "encoders": {k: {"kind": k} for k in ("avhubert", "dinov2")},
        "methods": {
            "tcn": {"encoder": "avhubert", "architecture": "tcn"},
            "p2": {**p2, "architecture": "syncartifact"},
        },
        "architectures": ["tcn", "p2"],
        "prepare_encoders": only,
    }
    for module, name in ((experiment, "prepare"), (assets, "setup_assets")):
        original, seen = getattr(module, name), []

        def fake(stage, *args, original=original, seen=seen):
            if stage.get("methods"):
                return original(stage, *args)
            seen.append(stage["encoder"]["kind"])
            return {}

        monkeypatch.setattr(module, name, fake)
        fake(cfg)
        # DINOv2 luôn sau AV-HuBERT; encoder không chọn không được trích/tải.
        assert seen == expected


def test_attach_caches_links_read_only_and_copies_writable(tmp_path):
    from vn_av_df.experiment import attach_caches

    try:  # Windows không quyền admin/Developer Mode không tạo được symlink; Kaggle (Linux) thì có.
        (tmp_path / "probe").symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip("Hệ thống không cho tạo symlink")
    first, second = tmp_path / "in1", tmp_path / "in2"
    for folder, key in ((first, "avhubert"), (second, "dinov2")):
        (folder / key).mkdir(parents=True)
        (folder / key / "a.json").write_text("{}")
    (second / "environment").mkdir()
    target = tmp_path / "cache"
    found = attach_caches([first, second], target, writable=["dinov2"])
    assert sorted(found) == ["avhubert", "dinov2"]
    assert (target / "avhubert").is_symlink() and (target / "avhubert/a.json").is_file()
    assert not (target / "dinov2").is_symlink() and (target / "dinov2/a.json").is_file()
    assert not (target / "environment").exists()
    # Chạy lại cell: giữ nguyên bản đã gom.
    (target / "dinov2/b.npz").write_bytes(b"x")
    attach_caches([first, second], target, writable=["dinov2"])
    assert (target / "dinov2/b.npz").is_file()
    with pytest.raises(ValueError, match="avhubert"):
        attach_caches([first, first], tmp_path / "other")
