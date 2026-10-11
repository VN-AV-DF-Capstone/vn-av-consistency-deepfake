"""Chọn encoder/cache theo phương pháp; ghép feature của một mẫu thành input model."""

import copy
from pathlib import Path

import numpy as np

# Model nhận feature native 25 Hz (pool về ô 0,2 s bên trong); TCN nhận feature đã pool sẵn.
NATIVE_MODELS = {"syncartifact", "tricue", "artifact"}
# Model cần đặc trưng môi (độ mở môi + đường bao tiếng) lưu trong cache AV-HuBERT.
LIP_MODELS = {"tricue"}


def method_config(cfg, name):
    """Gắn encoder/cache/head và danh sách generator fake được dùng khi train của method."""
    result = copy.deepcopy(cfg)
    spec = cfg.get("methods", {}).get(name)
    if spec:
        result["encoder"] = copy.deepcopy(cfg["encoders"][spec["encoder"]])
        result["cache"] = str(Path(cfg["cache"]) / spec["encoder"])
        result["model_architecture"] = spec["architecture"]
        result["train_generators"] = list(spec.get("train_generators", ["wav2lip_gan"]))
        if spec.get("branches"):
            result["branches"] = list(spec["branches"])
        if spec.get("artifact_encoder"):
            # Cache artifact riêng; hộp miệng đọc từ cache encoder chính (của mẫu hoặc clip gốc).
            result["artifact_encoder"] = {
                **copy.deepcopy(cfg["encoders"][spec["artifact_encoder"]]),
                "box_cache": result["cache"],
            }
            result["artifact_cache"] = str(Path(cfg["cache"]) / spec["artifact_encoder"])
    else:
        result["model_architecture"] = name
    result["method_name"] = name
    return result


def combined_signature(primary, artifact=None):
    """Checkpoint dùng hai encoder phụ thuộc cả hai; model một encoder giữ signature riêng."""
    if artifact is None:
        return primary
    from vn_av_df.common.runtime import fingerprint

    return fingerprint({"primary": primary, "artifact": artifact})


def artifact_arrays(path, meta, frames=None):
    """Đặc trưng DINOv2 native 25 Hz; nếu có frames thì phải khớp timeline AV-HuBERT."""
    with np.load(path, allow_pickle=False) as z:
        features = z["artifact"].astype(np.float32)
        valid = z["artifact_valid"].copy()
    if features.ndim != 2 or valid.shape != (len(features),):
        raise ValueError("Invalid artifact cache")
    if frames is not None and len(features) != frames:
        raise ValueError("Artifact cache differs from the AV-HuBERT timeline")
    if meta["step_s"] != 0.04:
        raise ValueError("Artifact cache must be native 25 Hz")
    return features, valid


def make_encoder(cfg):
    """Khởi tạo đúng backbone, tải weight chỉ ở bước setup tường minh."""
    if cfg.get("kind") == "avhubert":
        from vn_av_df.features.avhubert import AVHubertVideoEncoder

        return AVHubertVideoEncoder(cfg)
    if cfg.get("kind") == "dinov2":
        from vn_av_df.features.dinov2 import DinoMouthEncoder

        return DinoMouthEncoder(cfg)
    raise ValueError(f"Unknown encoder kind: {cfg.get('kind')}")


def feature_arrays(path, meta, architecture):
    """Native AV-HuBERT: model native nhận 25 Hz; TCN nhận mean 5 frame cùng ô 0,2 s."""
    with np.load(path, allow_pickle=False) as z:
        a, v = z["audio"].copy(), z["visual"].copy()
        mask = (z["audio_valid"] & z["visual_valid"]).copy()
        times = z["times_s"].copy()
    if (
        a.ndim != 2
        or v.ndim != 2
        or len(a) != len(v)
        or len(a) != len(times)
        or mask.shape != times.shape
    ):
        raise ValueError("Cache dimensions differ")
    if not len(times) or times[0] != 0 or not np.allclose(np.diff(times), meta["step_s"]):
        raise ValueError("Cache timeline differs")
    stride = int(meta.get("output_stride", 1))
    if stride < 1:
        raise ValueError("Invalid output stride")
    output_meta = {**meta, "step_s": meta["step_s"] * stride, "native_stride": stride}
    if stride > 1 and architecture not in NATIVE_MODELS:
        a = np.stack([a[i : i + stride].mean(0) for i in range(0, len(a), stride)])
        v = np.stack([v[i : i + stride].mean(0) for i in range(0, len(v), stride)])
        mask = np.array([mask[i : i + stride].all() for i in range(0, len(mask), stride)])
        output_meta["native_stride"] = 1
    return (a, v, mask), times[::stride], output_meta


def lip_arrays(path):
    """[T,3]: độ mở môi trong / bề rộng miệng, log đường bao tiếng, cờ hợp lệ.

    Landmark dlib 68 điểm (0-index): môi trong 60–67, mép miệng 48 và 54.
    Trả None nếu cache không có landmark (cache cũ).
    """
    with np.load(path, allow_pickle=False) as z:
        if "landmarks" not in z or "envelope" not in z:
            return None
        points = z["landmarks"].astype(np.float32)
        envelope = z["envelope"].astype(np.float32)
        valid = (z["audio_valid"] & z["visual_valid"]).copy()
    if points.shape != (len(valid), 68, 2) or envelope.shape != valid.shape:
        raise ValueError("Lip cache differs from the AV-HuBERT timeline")
    width = np.linalg.norm(points[:, 54] - points[:, 48], axis=1)
    gaps = [
        np.linalg.norm(points[:, a] - points[:, b], axis=1)
        for a, b in ((61, 67), (62, 66), (63, 65))
    ]
    aperture = np.mean(gaps, axis=0) / np.maximum(width, 1e-3)
    valid &= (width > 1) & np.isfinite(aperture) & np.isfinite(envelope)
    out = np.zeros((len(valid), 3), np.float32)
    out[:, 0] = np.where(valid, aperture, 0)
    out[:, 1] = np.where(valid, np.log(np.maximum(envelope, 1e-5)), 0)
    out[:, 2] = valid
    return out


def assemble(architecture, primary=None, artifact=None):
    """Input model một mẫu: [audio, visual, valid, artifact, lip], timeline ô output và meta.

    primary/artifact là (path, meta) hoặc None. Mẫu chỉ có cache artifact (MechFake không trích
    AV-HuBERT) cho audio/visual/lip = None; chỉ nhánh dấu vết vẽ dùng được.
    """
    if primary is None and artifact is None:
        raise ValueError("Need at least one feature cache")
    if primary is not None:
        (audio, visual, valid), times, meta = feature_arrays(*primary, architecture)
        lip = lip_arrays(primary[0]) if architecture in LIP_MODELS else None
        features = None
        if artifact is not None:
            features, artifact_valid = artifact_arrays(*artifact, len(audio))
            valid = valid & artifact_valid
        return [audio, visual, valid, features, lip], times, meta
    path, meta = artifact
    features, valid = artifact_arrays(path, meta)
    with np.load(path, allow_pickle=False) as z:
        times = z["times_s"].copy()
    stride = int(meta.get("output_stride", 5))
    output_meta = {
        **meta,
        "step_s": meta["step_s"] * stride,
        "native_stride": stride,
        "duration_s": meta.get("duration_s", len(features) * meta["step_s"]),
    }
    return [None, None, valid, features, None], times[::stride], output_meta
