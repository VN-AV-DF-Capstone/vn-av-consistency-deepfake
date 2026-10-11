"""P2 SyncArtifact trên cache fixture: kiểm protocol, không chứng minh độ chính xác thật."""

import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from vn_av_df.common.runtime import fingerprint, read_json, save_npz, sha, write_json
from vn_av_df.data.groups import write_manifest
from vn_av_df.dataset import SCHEMA
from vn_av_df.experiment import evaluate, load_model, train_one, train_selected
from vn_av_df.features.registry import combined_signature
from vn_av_df.models import build_model

FRAMES = 100  # 4 s ở 25 Hz, ô output 0,2 s.
VARIANTS = {
    # variant: (label, control_type, audio_mode, fake_intervals, av_mismatch_intervals)
    "real": (0, None, None, [], []),
    "sham": (0, "conventional_audio_splice", None, [], [[1.0, 2.0]]),
    "donor": (1, None, "donor", [[1.0, 3.0]], None),
    "source": (1, None, "source", [[0.0, 4.0]], None),
}


@pytest.fixture
def project(tmp_path):
    rows = []
    cache = tmp_path / "cache"
    dataset = tmp_path / "dataset"
    rng = np.random.default_rng(7)
    for split in ("train", "validation", "test"):
        for parent in range(2):
            for variant, (label, control, mode, spans, mismatch) in VARIANTS.items():
                sid = f"{split}_{parent}_{variant}"
                digest = fingerprint(sid)
                rows.append(
                    {
                        "sample_id": sid,
                        "video": sid + ".mp4",
                        "sha256": digest,
                        "duration_s": 4.0,
                        "label": label,
                        "fake_intervals": spans,
                        "av_mismatch_intervals": mismatch,
                        "control_type": control,
                        "audio_mode": mode,
                        "variant": variant if variant in ("real", "sham") else "full",
                        "split": split,
                        "source_id": f"{split}_{parent}",
                        "source_clip_id": f"{split}_{parent}",
                        "speaker_id": split,
                        "review_status": "keep",
                        "generator": "wav2lip_gan" if label else "none",
                        "generator_version": "fixture",
                    }
                )
                source = fingerprint({"assets": {"video": digest}, "variant": {"kind": "clean"}})
                boxes = np.tile(np.array([[32, 40, 20]], np.float32), (FRAMES, 1))
                avh = cache / "avhubert" / (sid + ".npz")
                save_npz(
                    avh,
                    audio=rng.normal(size=(FRAMES, 8)).astype("float32"),
                    visual=(rng.normal(size=(FRAMES, 8)) + label).astype("float32"),
                    times_s=np.arange(FRAMES) * 0.04,
                    audio_valid=np.ones(FRAMES, bool),
                    visual_valid=np.ones(FRAMES, bool),
                    mouth_boxes=boxes,
                )
                write_json(
                    avh.with_suffix(".json"),
                    dict(
                        feature_sha256=sha(avh),
                        feature_signature="avh-test",
                        source_fingerprint=source,
                        step_s=0.04,
                        output_stride=5,
                        duration_s=4.0,
                    ),
                )
                dino = cache / "dinov2" / (sid + ".npz")
                save_npz(
                    dino,
                    artifact=(rng.normal(size=(FRAMES, 6)) + 2 * label).astype("float16"),
                    times_s=np.arange(FRAMES) * 0.04,
                    artifact_valid=np.ones(FRAMES, bool),
                )
                write_json(
                    dino.with_suffix(".json"),
                    dict(
                        feature_sha256=sha(dino),
                        feature_signature="dino-test",
                        source_fingerprint=source,
                        box_sha256=sha(avh),
                        step_s=0.04,
                    ),
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
    p2 = {"encoder": "avhubert", "artifact_encoder": "dinov2"}
    cfg = {
        "dataset": str(dataset),
        "cache": str(cache),
        "runs": str(tmp_path / "runs"),
        "device": "cpu",
        "encoder": {},
        "encoders": {"avhubert": {"kind": "avhubert"}, "dinov2": {"kind": "dinov2"}},
        "methods": {
            "p2": {**p2, "architecture": "syncartifact"},
        },
        "architectures": ["p2"],
        "seeds": [42],
        "held_out_generators": [],
        "training": {
            "epochs": 1,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "hidden": 8,
            "dropout": 0.0,
            "patience": 0,
            "top_fraction": 0.2,
            "artifact_aux_weight": 0.5,
        },
        "reconstruction": {"epochs": 2, "lr": 0.001, "patience": 0},
        "sync": {"epochs": 2, "lr": 0.001, "patience": 0, "shift_frames": [2, 4]},
    }
    return cfg, rows


def test_p2_stages_protocol_evaluation_and_report(project, monkeypatch):
    cfg, rows = project
    train_selected(cfg)
    runs = Path(cfg["runs"])
    sync = read_json(runs / "p2_seed42/sync.json")
    labels = {r["sample_id"]: r["label"] for r in rows}
    # Stage S chỉ thấy real/sham; stage A chỉ thấy real gốc.
    assert {labels[s] for s in sync["train_ids"] + sync["validation_ids"]} == {0}
    assert sync["sham_train"] == 2
    recon = read_json(runs / "p2_seed42/reconstruction.json")
    assert all(s.endswith("_real") for s in recon["train_ids"] + recon["validation_ids"])
    model, state = load_model(runs / "p2_seed42/best.pt")
    assert state["feature_signature"] == combined_signature("avh-test", "dino-test")
    assert bool(model.sync_ready) and state["sync_report"]["best_validation_loss"] >= 0
    model.train()
    assert not model.sync_head.training
    a, v, art = torch.randn(FRAMES, 8), torch.randn(FRAMES, 8), torch.randn(FRAMES, 6)
    model(a, v, torch.ones(FRAMES, dtype=torch.bool), art).sum().backward()
    for module in (model.reconstructor, *model.sync_modules()):
        assert all(p.grad is None for p in module.parameters())
    assert model.classifier.weight.grad is not None
    results = {r["architecture"]: r for r in evaluate(cfg)}
    cells = {"real", "sham", "wav2lip_gan/donor", "wav2lip_gan/source"}
    assert set(results["p2"]["by_condition"]) == cells
    assert set(results["p2"]["branches"]["scores"]) == {"sync", "artifact"}
    assert set(results["p2"]["branches"]["scores"]["sync"]) == cells - {"real"}
    predictions = read_json(runs / "p2_seed42/predictions-test.json")
    assert all(len(p["branch_scores"]["sync"]) == len(p["scores"]) for p in predictions)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    from vn_av_df.reporting import detector_report

    report = detector_report(runs / "p2_seed42", "test")
    assert report["summary"]["sync"]["best_validation_loss"] >= 0
    assert set(report["summary"]["by_condition"]) == cells
    assert any(Path(p).name == "sync.png" for p in report["images"])
    cfg["training"]["epochs"] = 2
    train_one(cfg, "p2", 42, resume=True)
    assert len(read_json(runs / "p2_seed42/history.json")) == 2
    cfg["sync"]["shift_frames"] = [3, 5]
    with pytest.raises(ValueError, match="Resume"):
        train_one(cfg, "p2", 42, resume=True)


def test_p2_logs_every_epoch_to_wandb_while_training(project, monkeypatch):
    cfg, _ = project
    cfg["architectures"] = ["p2"]
    cfg["wandb"] = {"project": "test", "group": "run01"}
    logged, finished = [], []

    class Run:
        id, project, entity, summary = "abc123", "test", None, {}

        def define_metric(self, *args, **kwargs):
            pass

        def log(self, row):
            logged.append(row)

        def finish(self):
            finished.append(self.summary.get("best_epoch"))

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=lambda **kw: Run()))
    train_selected(cfg)
    # Ghi ngay sau từng epoch, đúng thứ tự stage A → S → detector, mỗi stage trục epoch riêng.
    assert [list(row)[0] for row in logged] == ["stageA/epoch"] * 2 + ["stageS/epoch"] * 2 + [
        "detector/epoch"
    ]
    assert "stageS/validation_auc" in logged[2] and "detector/validation_loss" in logged[4]
    assert read_json(Path(cfg["runs"]) / "p2_seed42/wandb.json")["id"] == "abc123"
    assert finished == [1]  # Đóng run sau khi train, kèm summary best epoch/resources.


def test_p2_inference_reports_branch_evidence(project):
    cfg, _ = project
    cfg["architectures"] = ["p2"]
    checkpoint = train_one(cfg, "p2", 42)
    from vn_av_df.inference import Analyzer

    class Fixture:
        def __init__(self, signature, folder, meta):
            self.signature, self.folder, self.meta = signature, folder, meta

        def extract(self, row, output):
            source = Path(cfg["cache"]) / self.folder / "test_0_donor.npz"
            Path(output).write_bytes(source.read_bytes())
            return self.meta

    primary = Fixture("avh-test", "avhubert", {"step_s": 0.04, "output_stride": 5, "duration_s": 4})
    artifact = Fixture("dino-test", "dinov2", {"step_s": 0.04})
    scoped = {**cfg, "checkpoint": str(checkpoint)}
    result = Analyzer(scoped, primary, artifact).analyze("fixture.mp4")
    assert set(result) == {"video_score", "intervals", "evidence"}
    assert set(result["evidence"]) == {"sync", "artifact"}
    assert all(0 <= x <= 1 for x in result["evidence"].values())
    # Trường mà demo/src/main.tsx đọc để vẽ timeline AI và hai timeline nhánh.
    detailed = Analyzer({**scoped, "demo_details": True}, primary, artifact).analyze("f.mp4")
    for key in ("scores", "times_s", "valid", "thresholds", "evidence_scores"):
        assert key in detailed
    assert set(detailed["evidence_scores"]) == {"sync", "artifact"}
    assert all(len(v) == len(detailed["scores"]) for v in detailed["evidence_scores"].values())
    with pytest.raises(ValueError, match="Encoder/preprocessing differs"):
        Analyzer(scoped, primary, Fixture("other", "dinov2", {"step_s": 0.04}))


def test_p2_no_information_crosses_gap():
    model = build_model(8, 8, "syncartifact", 8, 0, native_stride=5, artifact_dim=6).eval()
    model.reconstruction_ready.fill_(True)
    model.sync_ready.fill_(True)
    a, v, art = torch.randn(100, 8), torch.randn(100, 8), torch.randn(100, 6)
    mask = torch.ones(100, dtype=torch.bool)
    mask[50:55] = False
    first = model(a, v, mask, art)
    evidence = {k: x.clone() for k, x in model.outputs.items() if x is not None}
    a[55:] += 100
    v[55:] -= 100
    art[55:] *= -3
    second = model(a, v, mask, art)
    assert first.shape == (20,) and first[10] == 0
    assert torch.allclose(first[:10], second[:10])
    for name, value in evidence.items():
        assert torch.allclose(value[:10], model.outputs[name][:10])


def test_p2_requires_sync_stage_and_artifact_features():
    model = build_model(8, 8, "syncartifact", 8, 0, native_stride=5, artifact_dim=6)
    valid = torch.ones(10, dtype=torch.bool)
    model.reconstruction_ready.fill_(True)
    with pytest.raises(ValueError, match="sync stage"):
        model(torch.randn(10, 8), torch.randn(10, 8), valid, torch.randn(10, 6))
    model.sync_ready.fill_(True)
    with pytest.raises(ValueError, match="artifact"):
        model(torch.randn(10, 8), torch.randn(10, 8), valid)
    with pytest.raises(ValueError, match="artifact"):
        build_model(8, 8, "syncartifact", 8, 0, native_stride=5)


def test_shifted_real_labels_only_moved_frames():
    from vn_av_df.syncartifact import shifted

    audio = torch.arange(40, dtype=torch.float32)[:, None].repeat(1, 3)
    valid = torch.ones(40, dtype=torch.bool)
    for seed in range(20):
        (moved, _, mask), target = shifted((audio, audio, valid), random.Random(seed), (2, 4), 5)
        changed = (moved[:, 0] != audio[:, 0]) & mask
        assert torch.equal(changed, (target > 0) & mask)
        offsets = (moved[:, 0] - audio[:, 0])[changed].abs()
        assert ((offsets >= 2) & (offsets <= 4)).all()
        # Frame nguồn ngoài clip không được coi là quan sát.
        assert mask.sum() >= 40 - 4


def test_dinov2_mouth_encoder_uses_avhubert_boxes(tmp_path):
    from transformers import Dinov2Config, Dinov2Model

    from vn_av_df.data.render import encode
    from vn_av_df.features.dinov2 import DinoMouthEncoder

    video = tmp_path / "clip.mp4"
    frames = np.random.default_rng(1).integers(0, 255, (30, 64, 80, 3), dtype=np.uint8)
    encode(frames, np.zeros(30 * 1920, np.float32), video, 18)
    boxes = np.tile(np.array([[40, 36, 24]], np.float32), (30, 1))
    boxes[5:8] = 0  # Không có hộp miệng: frame phải bị mask.
    cache = tmp_path / "avhubert/clip.npz"
    save_npz(
        cache,
        audio_valid=np.ones(30, bool),
        visual_valid=np.ones(30, bool),
        mouth_boxes=boxes,
    )
    source = fingerprint({"assets": {"video": sha(video)}, "variant": {"kind": "clean"}})
    write_json(
        cache.with_suffix(".json"),
        dict(feature_sha256=sha(cache), source_fingerprint=source, output_stride=5),
    )
    backbone = Dinov2Model(
        Dinov2Config(
            hidden_size=12,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=24,
            image_size=28,
            patch_size=14,
        )
    )
    encoder = DinoMouthEncoder({"crop_size": 28, "box_cache": str(cache.parent)}, backbone)
    output = tmp_path / "dinov2/clip.npz"
    meta = encoder.extract({"video": str(video)}, output)
    with np.load(output) as z:
        assert z["artifact"].shape == (30, 24) and z["artifact"].dtype == np.float16
        assert not z["artifact_valid"][5:8].any() and z["artifact_valid"][8:].all()
        assert not z["artifact"][5:8].any()
    assert meta["box_sha256"] == sha(cache) and meta["step_s"] == 0.04
    assert encoder.extract({"video": str(video)}, output) == meta  # Cache hợp lệ được dùng lại.
    other = tmp_path / "other.mp4"
    encode(frames[::-1].copy(), np.zeros(30 * 1920, np.float32), other, 18)
    with pytest.raises(ValueError, match="another video"):
        encoder.extract({"video": str(other), "boxes": str(cache)}, tmp_path / "x.npz")
