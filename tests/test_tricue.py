"""TriCue (v002) trên cache fixture: kiểm quy ước và protocol, không chứng minh độ chính xác thật."""

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from vn_av_df.common.runtime import fingerprint, read_json, save_npz, sha, write_json
from vn_av_df.data.groups import write_manifest
from vn_av_df.dataset import SCHEMA
from vn_av_df.experiment import evaluate, load_model, train_selected
from vn_av_df.tricue import TriCue, lead_profiles, offset_sample

FRAMES = 200  # 8 s ở 25 Hz.
OPTIONS = {"max_offset": 3, "lead_window": 4, "lead_lags": 2}


def ready(model):
    for name in ("sync", "lead", "artifact", "fusion"):
        getattr(model, f"{name}_ready").fill_(True)
    return model


def test_sync_profile_peaks_at_the_applied_audio_shift():
    model = TriCue(4, 4, "tricue", 8, 0, 5, 6, branches=["sync"], **OPTIONS)
    visual = torch.randn(40, 4)
    valid = torch.ones(40, dtype=torch.bool)
    for shift in (-2, 0, 3):
        # Tiếng tại t mang nội dung của hình t+shift → đỉnh ở lớp độ lệch shift.
        audio = torch.roll(visual, -shift, 0)
        profile = model.sync_profile(audio, visual, valid)
        inner = profile[5:-5].argmax(-1) - OPTIONS["max_offset"]
        assert (inner == shift).all()


def test_tricue_no_information_crosses_gap():
    model = ready(TriCue(4, 4, "tricue", 8, 0, 5, 6, **OPTIONS)).eval()
    a, v, art = torch.randn(100, 4), torch.randn(100, 4), torch.randn(100, 6)
    lip = torch.zeros(100, 3)
    mask = torch.ones(100, dtype=torch.bool)
    mask[50:55] = False
    first = model(a, v, mask, art, lip)
    evidence = {k: x.clone() for k, x in model.outputs.items()}
    a[55:] += 100
    v[55:] -= 100
    art[55:] *= -3
    second = model(a, v, mask, art, lip)
    assert first.shape == (20,) and first[10] == 0
    assert torch.allclose(first[:10], second[:10])
    for name, value in evidence.items():
        assert torch.allclose(value[:10], model.outputs[name][:10])
    assert set(evidence) == {"sync", "lead", "artifact", "mismatch"}


def test_stages_must_run_before_use():
    model = TriCue(4, 4, "tricue", 8, 0, 5, 6, **OPTIONS)
    tensors = (torch.randn(10, 4), torch.randn(10, 4), torch.ones(10, dtype=torch.bool))
    with pytest.raises(ValueError, match="sync stage"):
        model(*tensors, torch.randn(10, 6), torch.zeros(10, 3))
    artifact = TriCue(0, 0, "artifact", 8, 0, 5, 6)
    assert artifact.branch_names == ["artifact"] and bool(artifact.fusion_ready)


def test_offset_targets_mark_shifted_cells_sham_and_boundaries():
    model = TriCue(4, 4, "tricue", 8, 0, 5, 6, branches=["sync"], **OPTIONS)
    k = OPTIONS["max_offset"]
    tensors = (
        torch.arange(40.0)[:, None].repeat(1, 4),
        torch.zeros(40, 4),
        torch.ones(40, dtype=torch.bool),
    )
    real = {"label": 0, "av_mismatch_intervals": []}
    for seed in range(10):
        (audio, _, valid), target, label = offset_sample(
            model, real, tensors, random.Random(seed), (2, 3), 1.0
        )
        assert label == 1
        moved = target[(target >= 0) & (target != k)]
        assert len(moved) and ((moved - k).abs() >= 2).all() and ((moved - k).abs() <= 3).all()
    sham = {
        "label": 0,
        "control_type": "conventional_audio_splice",
        "av_mismatch_intervals": [[0.2, 0.8]],
    }
    _, target, label = offset_sample(model, sham, tensors, random.Random(0), (2, 3), 1.0)
    assert label == 1 and target[1:4].tolist() == [2 * k + 1] * 3 and target[0] == k


def test_lead_profiles_find_onsets_and_lip_lead():
    n, lead = 200, 2
    envelope = np.full(n, -6.0, np.float32)
    for start in range(20, n - 20, 25):
        envelope[start : start + 12] = np.linspace(-1, -2, 12)
    aperture = np.roll(np.exp(envelope), -lead)  # Môi mở trước tiếng `lead` frame.
    lip = np.stack([aperture, envelope, np.ones(n, np.float32)], 1)
    found = lead_profiles(lip, np.ones(n, bool), 4, 3)
    assert len(found) >= 5
    peaks = [int(np.argmax(profile)) - 3 for _, profile in found]
    assert np.median(peaks) == lead


# ---- end-to-end trên cache fixture ----
def clip_features(rng, label, lead=2):
    """AV-HuBERT giả: hình = tiếng (dịch 0) với real; landmark môi + đường bao có điểm bắt đầu nói."""
    audio = rng.normal(size=(FRAMES, 8)).astype("float32")
    visual = (audio + 0.1 * rng.normal(size=(FRAMES, 8)) + label).astype("float32")
    envelope = np.full(FRAMES, 1e-3, np.float32)
    for start in range(20, FRAMES - 20, 25):
        envelope[start : start + 12] = 0.2 + 0.05 * rng.random(12)
    aperture = np.roll(np.sqrt(envelope), -lead)
    points = np.zeros((FRAMES, 68, 2), np.float32)
    points[:, 48] = (10, 50)
    points[:, 54] = (30, 50)
    for top, bottom in ((61, 67), (62, 66), (63, 65)):
        points[:, top] = np.c_[np.full(FRAMES, 20), 50 - 10 * aperture]
        points[:, bottom] = np.c_[np.full(FRAMES, 20), 50 + 10 * aperture]
    return audio, visual, points, envelope


@pytest.fixture
def project(tmp_path):
    rows = []
    cache = tmp_path / "cache"
    dataset = tmp_path / "dataset"
    rng = np.random.default_rng(3)
    variants = {
        "train": ["real", "sham", "w2l_donor", "mech_full", "mech_art"],
        "validation": ["real", "sham", "w2l_donor", "w2l_source", "mech_full", "mech_art"],
        "test": ["real", "sham", "w2l_donor", "w2l_partial", "diag"],
    }
    for split, names in variants.items():
        for parent in range(3):
            real_sid = f"{split}_{parent}_real"
            for variant in names:
                sid = f"{split}_{parent}_{variant}"
                digest = fingerprint(sid)
                label = int(variant.startswith(("w2l", "mech")))
                row = {
                    "sample_id": sid,
                    "video": sid + ".mp4",
                    "sha256": digest,
                    "duration_s": FRAMES / 25,
                    "label": label,
                    "fake_intervals": [[0.0, FRAMES / 25]] if label else [],
                    "av_mismatch_intervals": [] if not label else None,
                    "control_type": None,
                    "audio_mode": None,
                    "variant": "real",
                    "split": split,
                    "source_id": f"{split}_{parent}",
                    "source_clip_id": f"{split}_{parent}",
                    "speaker_id": split,
                    "review_status": "keep",
                    "generator": "none",
                    "generator_version": "fixture",
                }
                if variant == "sham":
                    row.update(
                        control_type="conventional_audio_splice",
                        av_mismatch_intervals=[[2.0, 4.0]],
                        variant="sham",
                    )
                elif variant.startswith("w2l"):
                    row.update(generator="wav2lip_gan", variant="full", audio_mode="donor")
                    if variant == "w2l_source":
                        row["audio_mode"] = "source"
                    if variant == "w2l_partial":
                        row.update(variant="partial", fake_intervals=[[2.0, 4.0]])
                elif variant.startswith("mech"):
                    row.update(
                        generator="mech_blend" if variant == "mech_full" else "mech_lowres",
                        variant="full",
                        audio_mode="source",
                        av_mismatch_intervals=[],
                        box_sample_id=real_sid,
                        feature_scope="full" if variant == "mech_full" else "artifact",
                    )
                elif variant == "diag":
                    row.update(generator="diag_paste", variant="diagnostic", diagnostic="paste")
                rows.append(row)
                source = fingerprint({"assets": {"video": digest}, "variant": {"kind": "clean"}})
                audio, visual, points, envelope = clip_features(rng, label)
                if variant == "sham":
                    audio[50:100] = rng.normal(size=(50, 8))
                avh = cache / "avhubert" / (sid + ".npz")
                if row.get("feature_scope") != "artifact":
                    save_npz(
                        avh,
                        audio=audio,
                        visual=visual,
                        times_s=np.arange(FRAMES) * 0.04,
                        audio_valid=np.ones(FRAMES, bool),
                        visual_valid=np.ones(FRAMES, bool),
                        mouth_boxes=np.tile(np.array([[32, 40, 20]], np.float32), (FRAMES, 1)),
                        landmarks=points.astype(np.float16),
                        envelope=envelope,
                    )
                    write_json(
                        avh.with_suffix(".json"),
                        dict(
                            feature_sha256=sha(avh),
                            feature_signature="avh-test",
                            source_fingerprint=source,
                            step_s=0.04,
                            output_stride=5,
                            duration_s=FRAMES / 25,
                        ),
                    )
                dino = cache / "dinov2" / (sid + ".npz")
                save_npz(
                    dino,
                    artifact=(rng.normal(size=(FRAMES, 6)) + 2 * label).astype("float16"),
                    times_s=np.arange(FRAMES) * 0.04,
                    artifact_valid=np.ones(FRAMES, bool),
                )
                box = sid if row.get("feature_scope") != "artifact" else real_sid
                write_json(
                    dino.with_suffix(".json"),
                    dict(
                        feature_sha256=sha(dino),
                        feature_signature="dino-test",
                        source_fingerprint=source,
                        box_sha256=sha(cache / "avhubert" / (box + ".npz")),
                        box_sample_id=None if box == sid else box,
                        step_s=0.04,
                        output_stride=5,
                        duration_s=FRAMES / 25,
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
    both = {"encoder": "avhubert", "artifact_encoder": "dinov2"}
    cfg = {
        "dataset": str(dataset),
        "cache": str(cache),
        "runs": str(tmp_path / "runs"),
        "device": "cpu",
        "encoder": {},
        "encoders": {"avhubert": {"kind": "avhubert"}, "dinov2": {"kind": "dinov2"}},
        "methods": {
            "tri": {**both, "architecture": "tricue", "train_generators": ["mech_*"]},
            "art": {**both, "architecture": "artifact", "train_generators": ["mech_*"]},
            "tri_no_lead": {
                **both,
                "architecture": "tricue",
                "train_generators": ["wav2lip_gan"],
                "branches": ["sync", "artifact"],
            },
        },
        "architectures": ["tri", "art", "tri_no_lead"],
        "seeds": [42],
        "held_out_generators": [],
        "training": {
            "epochs": 2,
            "lr": 0.003,
            "weight_decay": 0.0001,
            "hidden": 8,
            "dropout": 0.0,
            "patience": 0,
            "top_fraction": 0.2,
            "fusion_epochs": 5,
            "fusion_lr": 0.05,
        },
        "sync": {"epochs": 2, "lr": 0.003, "patience": 0, "shift_frames": [2, 3]},
        "tricue": OPTIONS,
        "lead": {"shrinkage": 0.2},
    }
    return cfg, rows


def test_tricue_stages_protocol_two_outputs_and_diagnostics(project):
    cfg, rows = project
    train_selected(cfg)
    runs = Path(cfg["runs"])
    stages = read_json(runs / "tri_seed42/stages.json")
    assert set(stages) == {"sync", "lead", "artifact", "fusion"}
    labels = {r["sample_id"]: r["label"] for r in rows}
    # Nhánh lệch tiếng chỉ thấy real/sham; MechFake chỉ-artifact không vào stage cần AV-HuBERT.
    assert {labels[s] for s in stages["sync"]["train_ids"] + stages["sync"]["validation_ids"]} == {
        0
    }
    assert stages["artifact"]["train_samples"] == 12  # real, sham, 2 MechFake mỗi parent × 3.
    assert stages["fusion"]["train_samples"] == 9  # Bỏ MechFake không có cache AV-HuBERT.
    assert set(read_json(runs / "art_seed42/stages.json")) == {"artifact"}
    assert set(read_json(runs / "tri_no_lead_seed42/stages.json")) == {"sync", "artifact", "fusion"}
    model, state = load_model(runs / "tri_seed42/best.pt")
    assert state["train_generators"] == ["mech_*"] and model.config["branches"] == [
        "sync",
        "lead",
        "artifact",
    ]
    results = {r["architecture"]: r for r in evaluate(cfg)}
    tri = results["tri"]
    # Đầu ra phụ "lệch tiếng" và điểm từng nhánh có AUC theo ô; chẩn đoán báo riêng.
    assert {"sync", "lead", "artifact", "mismatch"} <= set(tri["branches"]["scores"])
    assert "diag/paste" not in tri["by_condition"]
    assert set(tri["diagnostics"]) == {"paste"} and tri["diagnostics"]["paste"]["samples"] == 3
    assert set(tri["localization_check"]) == {"wav2lip_gan"}
    assert tri["localization_check"]["wav2lip_gan"]["clips"] == 3
    assert set(results["art"]["branches"]["scores"]) == {"artifact"}
    diagnostics = read_json(runs / "tri_seed42/predictions-diagnostics.json")
    assert {p["diagnostic"] for p in diagnostics} == {"paste"}
    with pytest.raises(FileExistsError):
        from vn_av_df.experiment import train_one

        train_one(cfg, "tri", 42)
    assert train_one(cfg, "tri", 42, resume=True) == runs / "tri_seed42/best.pt"
    cfg["tricue"] = {**OPTIONS, "lead_lags": 3}
    with pytest.raises(ValueError, match="Resume"):
        train_one(cfg, "tri", 42, resume=True)
