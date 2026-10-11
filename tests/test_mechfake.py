"""MechFake: chỉ vùng miệng bị thay, ngoài vùng giữ nguyên; chẩn đoán toàn khung giữ hình học."""

import random

import numpy as np
import pytest

from vn_av_df import mechfake

FRAMES, HEIGHT, WIDTH = 24, 120, 160
# Mép miệng phải/trái và mũi (x, y): hộp nửa dưới mặt ≈ x 48–112, y 55–119.
POINTS = np.tile(np.array([[70, 80, 90, 80, 80, 60]], np.float32), (FRAMES, 1))


@pytest.fixture
def frames():
    rng = np.random.default_rng(5)
    return rng.integers(0, 255, (FRAMES, HEIGHT, WIDTH, 3), dtype=np.uint8)


@pytest.mark.parametrize(
    "name", ["mech_blend", "mech_lowres", "mech_vae", "mech_warp", "mech_temporal", "diag_paste"]
)
def test_mechanism_changes_only_the_mouth_region(name, frames, monkeypatch):
    monkeypatch.setattr(mechfake, "face_points", lambda frames, model: POINTS)
    # SD-VAE thật không có ở môi trường test: thay bằng phép biến đổi cùng hình dạng.
    monkeypatch.setattr(
        mechfake, "vae_roundtrip", lambda patches, side, folder, device: patches[:, ::-1].copy()
    )
    cfg = {"mechfake": {"face_model": "unused", "vae": "unused"}}
    out, detail = mechfake.render(name, frames, None, random.Random(1), cfg)
    assert out.shape == frames.shape and out.dtype == np.uint8
    # Góc khung ở xa miệng: không đổi một điểm ảnh nào.
    assert np.array_equal(out[:, :30, :30], frames[:, :30, :30])
    assert np.array_equal(out[:, :, -30:], frames[:, :, -30:])
    # Tâm vùng miệng: thay đổi rõ.
    assert np.abs(out[:, 75:95, 70:90].astype(int) - frames[:, 75:95, 70:90]).mean() > 5
    assert detail["region"] in ("lips", "lower")


def test_temporal_index_is_never_identity():
    for seed in range(30):
        index, mode = mechfake.temporal_index(100, random.Random(seed))
        assert mode in ("offset", "hold", "speed") and (index != np.arange(100)).any()
        assert index.min() >= 0 and index.max() <= 99


def test_global_diagnostic_keeps_geometry(frames):
    pcm = np.zeros(FRAMES * 1920, np.float32)
    out, detail = mechfake.render("diag_global", frames, pcm, random.Random(2), {})
    assert out.shape == frames.shape and detail["op"] in ("scale", "sharpen", "gamma")
    assert not np.array_equal(out, frames)


def test_adapter_rejects_unknown_mechanism_and_keeps_timeline(tmp_path, frames, monkeypatch):
    from vn_av_df.data.media import decode
    from vn_av_df.data.render import encode

    with pytest.raises(ValueError, match="Unknown"):
        mechfake.Adapter("mech_unknown")
    monkeypatch.setattr(mechfake, "face_points", lambda frames, model: POINTS)
    video = tmp_path / "clip.mp4"
    pcm = (0.2 * np.sin(np.arange(FRAMES * 1920) / 30)).astype(np.float32)
    encode(frames, pcm, video, 18)
    cfg = {"generation": {"max_side": 640}, "mechfake": {"face_model": "unused", "device": "cpu"}}
    output = tmp_path / "out.mkv"
    mechfake.Adapter("mech_lowres").synthesize(cfg, video, video, output, FRAMES, WIDTH, HEIGHT)
    original, result = decode(video, sample_rate=48000), decode(output, sample_rate=48000)
    # Cùng số khung/kích thước và độ dài tiếng; nội dung tiếng do pipeline sinh gán lại từ gốc.
    assert result["frames"].shape == original["frames"].shape
    assert result["pcm"].shape == original["pcm"].shape
    assert np.array_equal(result["frames"][:, :30, :30], original["frames"][:, :30, :30])
