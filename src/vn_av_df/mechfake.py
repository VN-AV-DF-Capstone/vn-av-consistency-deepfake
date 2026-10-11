"""MechFake (v002): giả vùng miệng theo cơ chế sinh, chỉ từ video thật, không cần generator.

- mech_blend (M1): dán/ghép, trộn hai biến thể của cùng vùng miệng (ý SBI / SAVe).
- mech_lowres (M2): vẽ ở độ phân giải thấp rồi phóng (họ GAN như Wav2Lip 96 px).
- mech_vae (M3): giải mã từ latent SD-VAE ft-mse (họ latent như MuseTalk, LatentSync; ý Rajan 2025).
- mech_warp (M4): uốn kết cấu miệng của khung khác bằng optical flow (họ biến dạng như DINet).
- mech_temporal (M5): sai thời điểm: miệng của khung khác / lặp khung / đổi tốc độ (gần AVPF).

Chẩn đoán E6, nhãn 0 (không AI): diag_paste (miệng thật thu 96 px rồi dán lại, chỉ có nghĩa với
model train bằng Wav2Lip), diag_global (xử lý toàn khung lành tính + nén lại).

Tham số lấy ngẫu nhiên trong khoảng chung, không chép thông số của generator test. Vùng miệng từ
YuNet (mép miệng, mũi) nên không cần dlib. Tiếng giữ nguyên tiếng gốc.
"""

import random
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from vn_av_df.common.runtime import fingerprint, read_json, sha
from vn_av_df.generators import DIAGNOSTICS, MECHANISMS

WORK = 256  # Cạnh patch xử lý (pixel) trước khi dán lại về kích thước thật.
VAE_REPO = "stabilityai/sd-vae-ft-mse"
_VAE = {}
# Clip gốc đã giải mã + điểm mặt, dùng chung cho 5 cơ chế của cùng parent (các luồng GPU cùng
# đọc; mọi bước chỉ tạo mảng mới, không sửa frame). Giữ vài clip gần nhất cho vừa RAM.
_SOURCES = OrderedDict()
_SOURCES_LOCK = threading.Lock()
_SOURCES_KEEP = 4


# ---- vùng miệng ----
def face_points(frames, model):
    """[T, 6]: mép miệng phải/trái và mũi (x, y) của mặt điểm cao nhất mỗi frame, đã nội suy và
    làm mượt. Thiếu mặt quá 20% frame → lỗi "Face not detected" (cặp bị bỏ, như generator thật).
    """
    import cv2

    h, w = frames[0].shape[:2]
    detector = cv2.FaceDetectorYN.create(str(model), "", (w, h), 0.6, 0.3, 50)
    points = np.full((len(frames), 6), np.nan, np.float32)
    for i, frame in enumerate(frames):
        _, faces = detector.detect(frame)
        if faces is not None and len(faces):
            best = faces[np.argmax(faces[:, 14])]
            points[i] = best[[10, 11, 12, 13, 8, 9]]
    found = ~np.isnan(points[:, 0])
    if found.mean() < 0.8:
        raise ValueError("Face not detected in enough frames for MechFake")
    index = np.arange(len(frames))
    for j in range(points.shape[1]):
        points[:, j] = np.interp(index, index[found], points[found, j])
    kernel = np.ones(5) / 5
    padded = np.pad(points, ((2, 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, j], kernel, "valid") for j in range(6)], 1)


def geometry(points, region):
    """[T, 3] = (cx, cy, cạnh) hộp vuông quanh môi hoặc nửa dưới mặt."""
    right, left = points[:, 0:2], points[:, 2:4]
    mouth = (right + left) / 2
    width = np.maximum(np.linalg.norm(left - right, axis=1), 4)
    if region == "lips":
        return np.c_[mouth, 2.2 * width]
    # Nửa dưới mặt: dời tâm xuống phía cằm, hộp đủ rộng gồm môi, hàm và viền dán.
    return np.c_[mouth + np.c_[np.zeros(len(mouth)), 0.35 * width], 3.2 * width]


def affine(cx, cy, side, size):
    scale = size / side
    return np.array([[scale, 0, size / 2 - scale * cx], [0, scale, size / 2 - scale * cy]])


def crop(frame, box, size=WORK):
    import cv2

    return cv2.warpAffine(
        frame, affine(*box, size), (size, size), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT
    )


def paste(frame, patch, alpha, box):
    """Dán patch (và mặt nạ alpha) về đúng vị trí; ngoài mặt nạ giữ nguyên frame gốc."""
    import cv2

    h, w = frame.shape[:2]
    inverse = cv2.invertAffineTransform(affine(*box, patch.shape[0]))
    image = cv2.warpAffine(patch, inverse, (w, h), flags=cv2.INTER_LINEAR)
    weight = cv2.warpAffine(alpha, inverse, (w, h), flags=cv2.INTER_LINEAR)[..., None]
    return (frame * (1 - weight) + image * weight).round().clip(0, 255).astype(np.uint8)


def soft_mask(rng, region, size=WORK):
    """Elip quanh môi / nửa dưới mặt, biến dạng nhẹ và làm mềm viền (cố định trong một clip)."""
    import cv2

    mask = np.zeros((size, size), np.float32)
    if region == "lips":
        center, axes = (size / 2, size / 2), (0.38 * size, 0.24 * size)
    else:
        center, axes = (size / 2, 0.45 * size), (0.42 * size, 0.36 * size)
    cv2.ellipse(
        mask,
        (round(center[0]), round(center[1])),
        (round(axes[0]), round(axes[1])),
        0,
        0,
        360,
        1.0,
        -1,
    )
    matrix = cv2.getRotationMatrix2D(
        (size / 2, size / 2), rng.uniform(-8, 8), rng.uniform(0.9, 1.1)
    )
    matrix[:, 2] += np.array([rng.uniform(-0.04, 0.04), rng.uniform(-0.04, 0.04)]) * size
    mask = cv2.warpAffine(mask, matrix, (size, size))
    sigma = rng.uniform(0.02, 0.06) * size
    return cv2.GaussianBlur(mask, (0, 0), sigma).clip(0, 1)


# ---- từng cơ chế trên patch [T, S, S, 3] ----
def blend_source(patches, rng):
    """M1: biến thể màu/độ nét/dịch nhỏ của chính vùng đó (một bộ tham số mỗi clip)."""
    import cv2

    hue, saturation, value = rng.uniform(-5, 5), rng.uniform(0.9, 1.1), rng.uniform(0.9, 1.1)
    shift = np.array([rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02)]) * WORK
    sigma = rng.choice([0.0, rng.uniform(0.5, 1.5)])
    sharpen = rng.random() < 0.3
    out = []
    for patch in patches:
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + hue) % 180
        hsv[..., 1] *= saturation
        hsv[..., 2] *= value
        image = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        image = cv2.warpAffine(
            image,
            np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]),
            (WORK, WORK),
            borderMode=cv2.BORDER_REFLECT,
        )
        if sigma:
            image = cv2.GaussianBlur(image, (0, 0), sigma)
        if sharpen:
            image = cv2.addWeighted(image, 1.5, cv2.GaussianBlur(image, (0, 0), 1.0), -0.5, 0)
        out.append(image)
    return np.stack(out)


def lowres(patches, side, interpolation):
    """M2 / diag_paste: thu nhỏ còn side px rồi phóng lại."""
    import cv2

    small = [cv2.resize(p, (side, side), interpolation=cv2.INTER_AREA) for p in patches]
    return np.stack([cv2.resize(s, (WORK, WORK), interpolation=interpolation) for s in small])


def vae_roundtrip(patches, side, folder, device):
    """M3: encode → decode qua SD-VAE ft-mse (không khử nhiễu, không UNet)."""
    import cv2
    import torch

    key = (str(folder), str(device))
    if key not in _VAE:
        from diffusers import AutoencoderKL

        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        _VAE[key] = AutoencoderKL.from_pretrained(str(folder), torch_dtype=dtype).to(device).eval()
    vae = _VAE[key]
    out = []
    with torch.no_grad():
        for start in range(0, len(patches), 16):
            batch = np.stack(
                [
                    cv2.resize(p, (side, side), interpolation=cv2.INTER_AREA)
                    for p in patches[start : start + 16]
                ]
            )
            x = torch.from_numpy(batch[..., ::-1].copy()).permute(0, 3, 1, 2).float() / 127.5 - 1
            x = x.to(device=device, dtype=vae.dtype)
            decoded = vae.decode(vae.encode(x).latent_dist.mode()).sample
            images = ((decoded.float().clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()
            for image in images.transpose(0, 2, 3, 1)[..., ::-1]:
                out.append(cv2.resize(image, (WORK, WORK), interpolation=cv2.INTER_LINEAR))
    return np.stack(out)


def warp_from(patches, offset):
    """M4: kết cấu của khung t+offset uốn theo bố cục khung t (DIS optical flow)."""
    import cv2

    flow_model = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    grid = np.stack(np.meshgrid(np.arange(WORK), np.arange(WORK)), -1).astype(np.float32)
    out = []
    for t, patch in enumerate(patches):
        other = patches[int(np.clip(t + offset, 0, len(patches) - 1))]
        flow = flow_model.calc(
            cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), cv2.cvtColor(other, cv2.COLOR_BGR2GRAY), None
        )
        coords = grid + flow
        out.append(
            cv2.remap(
                other,
                coords[..., 0],
                coords[..., 1],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
        )
    return np.stack(out)


def temporal_index(n, rng):
    """M5: chỉ số khung lấy miệng cho mỗi frame và kiểu thao tác."""
    mode = rng.choice(["offset", "hold", "speed"])
    index = np.arange(n)
    if mode == "offset":
        index = index + rng.choice((-1, 1)) * rng.randint(5, 25)
    elif mode == "hold":
        step = rng.randint(2, 4)
        index = index - index % step
    else:
        index = np.round(index * rng.uniform(0.8, 1.2)).astype(int)
    return np.clip(index, 0, n - 1), mode


def global_processing(frames, pcm, rng):
    """diag_global: một xử lý toàn khung lành tính rồi nén lại H.264 CRF 28–35."""
    import cv2

    from vn_av_df.data.media import decode
    from vn_av_df.data.render import encode

    h, w = frames.shape[1:3]
    op = rng.choice(["scale", "sharpen", "gamma"])
    if op == "scale":
        factor = rng.uniform(0.5, 0.75)
        small = (max(2, round(w * factor)), max(2, round(h * factor)))
        frames = np.stack(
            [cv2.resize(cv2.resize(f, small, interpolation=cv2.INTER_AREA), (w, h)) for f in frames]
        )
    elif op == "sharpen":
        amount = rng.uniform(0.5, 1.5)
        frames = np.stack(
            [
                cv2.addWeighted(f, 1 + amount, cv2.GaussianBlur(f, (0, 0), 1.2), -amount, 0)
                for f in frames
            ]
        )
    else:
        gamma = rng.uniform(0.8, 1.25)
        table = (np.linspace(0, 1, 256) ** gamma * 255).round().astype(np.uint8)
        frames = np.stack([cv2.LUT(f, table) for f in frames])
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "global.mp4"
        encode(frames, pcm, path, rng.randint(28, 35))
        again = decode(path, max_side=max(h, w), sample_rate=48000)["frames"][: len(frames)]
    return again[:, :h, :w], op


def source(video, max_side, model, digest):
    """(clip đã giải mã, điểm mặt hoặc lỗi dò mặt) của parent, lấy từ bộ nhớ nếu vừa dùng."""
    from vn_av_df.data.media import decode

    key = (digest, max_side, str(model))
    with _SOURCES_LOCK:
        if key in _SOURCES:
            _SOURCES.move_to_end(key)
            return _SOURCES[key]
    decoded = decode(video, max_side=max_side, sample_rate=48000)
    try:
        points = face_points(decoded["frames"], model)
    except ValueError as exc:
        points = exc
    with _SOURCES_LOCK:
        _SOURCES[key] = decoded, points
        while len(_SOURCES) > _SOURCES_KEEP:
            _SOURCES.popitem(last=False)
    return decoded, points


def render(name, frames, pcm, rng, cfg, device="cpu", points=None):
    """Frame mới của cơ chế `name`; trả (frames, mô tả tham số). `points` = face_points đã tính."""
    if name == "diag_global":
        out, op = global_processing(frames, pcm, rng)
        return out, {"op": op}
    import cv2

    if points is None:
        points = face_points(frames, cfg["mechfake"]["face_model"])
    if isinstance(points, Exception):
        raise ValueError(str(points))
    # M5 chỉ dạy sai thời điểm: lấy riêng môi, vì nửa dưới mặt của khung khác kéo theo cổ/áo lệch
    # tư thế thành vết dán lớn (lẫn sang M1).
    if name == "diag_paste":
        region = "lower"
    elif name == "mech_temporal":
        region = "lips"
    else:
        region = rng.choice(["lips", "lower"])
    boxes = geometry(points, region)
    patches = np.stack([crop(f, b) for f, b in zip(frames, boxes)])
    # Cạnh hộp (pixel của frame): tham số M2/M3 tính theo nó, không theo WORK. Frame 640 px chỉ có
    # hộp miệng ~50–120 px; cạnh cố định lớn hơn hộp thì thu–phóng không mất gì (mẫu fake rỗng).
    box_px = float(np.median(boxes[:, 2]))
    detail = {"region": region, "box_px": round(box_px, 1)}
    if name == "mech_blend":
        new = blend_source(patches, rng)
    elif name in ("mech_lowres", "diag_paste"):
        if name == "diag_paste":
            # Như Wav2Lip (mặt về 96 px): hộp nửa dưới mặt lớn hơn 96 px mới bị thu.
            side = round(min(96.0, box_px))
        else:
            side = max(12, round(box_px * rng.uniform(0.3, 0.7)))
        interpolation = rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC])
        new = lowres(patches, side, interpolation)
        detail["side"] = side
    elif name == "mech_vae":
        # Mỗi ô latent (8 px của ảnh vào VAE) phủ 2–5 px frame (MuseTalk trên mặt 1080p rồi thu
        # về 640 px: ~3 px).
        cell = rng.uniform(2.0, 5.0)
        side = max(64, 8 * round(box_px / cell))
        new = vae_roundtrip(patches, side, cfg["mechfake"]["vae"], device)
        detail.update(side=side, latent_cell_px=round(cell, 2))
    elif name == "mech_warp":
        offset = rng.choice((-1, 1)) * rng.randint(3, 25)
        new = warp_from(patches, offset)
        detail["offset"] = int(offset)
    elif name == "mech_temporal":
        index, mode = temporal_index(len(frames), rng)
        new = np.stack([crop(frames[i], boxes[i]) for i in index])
        detail["mode"] = mode
    else:
        raise ValueError(f"Unknown MechFake mechanism: {name}")
    alpha = soft_mask(rng, region)
    out = np.stack([paste(f, p, alpha, b) for f, p, b in zip(frames, new, boxes)])
    return out, detail


class Adapter:
    """Cùng giao diện generator thật (setup / provenance / synthesize) để chạy trong generation."""

    def __init__(self, name):
        if name not in MECHANISMS and name not in DIAGNOSTICS:
            raise ValueError(f"Unknown MechFake mechanism: {name}")
        self.name = name

    def setup(self, cfg):
        from vn_av_df.assets import MEDIA, fetch_file, pinned_snapshot

        model = Path(cfg["mechfake"]["face_model"])
        record = model.with_suffix(model.suffix + ".download.json")
        if not (model.exists() and record.exists()):
            fetch_file(MEDIA["face"], model)
        if self.name == "mech_vae":
            pinned_snapshot(
                VAE_REPO,
                cfg["mechfake"]["vae"],
                ["config.json", "diffusion_pytorch_model.safetensors"],
            )
        return self.provenance(cfg)

    def provenance(self, cfg):
        model = Path(cfg["mechfake"]["face_model"])
        if not model.is_file():
            raise FileNotFoundError("YuNet missing: run generator setup for MechFake")
        result = {
            "generator": self.name,
            "version": "mechfake-v1",
            "implementation_sha256": sha(__file__),
            "face_model_sha256": sha(model),
        }
        if self.name == "mech_vae":
            folder = Path(cfg["mechfake"]["vae"])
            result["vae"] = read_json(folder / "vn_av_revision.json")
            result["vae_sha256"] = sha(folder / "diffusion_pytorch_model.safetensors")
        return result

    def synthesize(self, cfg, video, audio_video, output, frames, width, height, gpu=None):
        """Đọc clip gốc, thay vùng miệng theo cơ chế, ghi FFV1 + PCM gốc (chưa nén mất mát)."""
        from vn_av_df.data.render import write_lossless

        digest = sha(video)
        model = cfg["mechfake"]["face_model"]
        decoded, points = source(video, cfg["generation"]["max_side"], model, digest)
        if len(decoded["frames"]) != int(frames):
            raise ValueError("Decoded frame count differs from plan")
        device = cfg["mechfake"].get("device", "cpu")
        if str(device).startswith("cuda") and gpu is not None:
            device = f"cuda:{gpu}"
        rng = random.Random(fingerprint([digest, self.name]))
        out, _ = render(self.name, decoded["frames"], decoded["pcm"], rng, cfg, device, points)
        write_lossless(out, decoded["pcm"], output)
