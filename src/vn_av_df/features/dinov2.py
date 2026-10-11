"""Nhánh artifact của P2: crop miệng RGB → DINOv2 ViT-S/14 đóng băng, native 25 Hz.

Hộp miệng lấy từ landmark dlib trong cache AV-HuBERT cùng sample, không chạy landmark
lần hai. Frame thiếu mặt/hộp giữ zero và mask=False, không nội suy.
"""

from pathlib import Path

import numpy as np
import torch

from vn_av_df.common.runtime import fingerprint, read_json, require_file, save_npz, sha, write_json

MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # Chuẩn hóa ImageNet của DINOv2.
STD = np.array([0.229, 0.224, 0.225], np.float32)


def mouth_crop(frame, box, size):
    """Cắt hộp vuông [cx, cy, side] rồi resize; phần ngoài frame là zero như nhau cho mọi nhãn."""
    import cv2

    cx, cy, side = (float(x) for x in box)
    scale = size / side
    matrix = np.array([[scale, 0, size / 2 - scale * cx], [0, scale, size / 2 - scale * cy]])
    crop = cv2.warpAffine(
        frame,
        matrix,
        (size, size),
        flags=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


class DinoMouthEncoder:
    """CLS + mean patch token mỗi frame; cache ràng buộc với hash cache hộp miệng AV-HuBERT."""

    def __init__(self, cfg, backbone=None):
        self.cfg = dict(cfg)
        self.crop = int(cfg.get("crop_size", 224))
        if self.crop < 14 or self.crop % 14:
            raise ValueError("crop_size must be a positive multiple of the ViT-S/14 patch")
        self.device = cfg.get("device", "cpu")
        if backbone is None:
            from transformers import Dinov2Model

            folder = Path(cfg["model"])
            assets = {
                name: sha(require_file(folder / name, "DINOv2 " + name))
                for name in ("config.json", "model.safetensors")
            }
            backbone = Dinov2Model.from_pretrained(str(folder), local_files_only=True)
        else:
            assets = {"backbone": "test-fixture"}
        self.backbone = backbone.eval().requires_grad_(False).to(self.device)
        self.signature = fingerprint(
            dict(
                assets=assets,
                implementation=sha(__file__),
                media=sha(Path(__file__).parents[1] / "data/media.py"),
                options={
                    k: cfg.get(k, default)
                    for k, default in (("crop_size", 224), ("max_duration", 60), ("max_side", 640))
                },
                format="dinov2-mouth-v1",
            )
        )

    @torch.no_grad()
    def embed(self, crops):
        pixels = torch.from_numpy((crops.astype(np.float32) / 255 - MEAN) / STD)
        pixels = pixels.permute(0, 3, 1, 2).to(self.device)
        precision = self.cfg.get("precision", "float32")
        use_half = str(self.device).startswith("cuda") and precision == "float16"
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_half):
            tokens = self.backbone(pixel_values=pixels).last_hidden_state.float()
        return torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], -1).cpu().numpy()

    def extract(self, row, output):
        """Hộp miệng: row['boxes'], box_cache/<tên output> (cache AV-HuBERT của cùng video),
        hoặc box_cache/<box_sample_id> (clip real gốc) cho MechFake không trích AV-HuBERT —
        mặt giữ nguyên vị trí khi chỉ vùng miệng bị thay.
        """
        from vn_av_df.data.media import decode

        output = Path(output)
        video = require_file(row["video"], "video")
        source = fingerprint({"assets": {"video": sha(video)}, "variant": {"kind": "clean"}})
        boxes_path = Path(row.get("boxes") or Path(self.cfg["box_cache"]) / output.name)
        borrowed = None
        if not row.get("boxes") and not boxes_path.is_file() and row.get("box_sample_id"):
            borrowed = row["box_sample_id"]
            boxes_path = Path(self.cfg["box_cache"]) / (borrowed + ".npz")
        box_meta = read_json(boxes_path.with_suffix(".json"))
        if box_meta["feature_sha256"] != sha(boxes_path) or (
            borrowed is None and box_meta["source_fingerprint"] != source
        ):
            raise ValueError("Mouth boxes are stale or belong to another video")
        if output.is_file() and output.with_suffix(".json").is_file():
            meta = read_json(output.with_suffix(".json"))
            if (
                meta.get("feature_signature") == self.signature
                and meta.get("source_fingerprint") == source
                and meta.get("box_sha256") == box_meta["feature_sha256"]
                and meta.get("feature_sha256") == sha(output)
            ):
                return meta
        with np.load(boxes_path, allow_pickle=False) as z:
            boxes = z["mouth_boxes"].copy()
            valid = (z["audio_valid"] & z["visual_valid"]).copy()
        decoded = decode(
            video, self.cfg.get("max_duration", 60), self.cfg.get("max_side", 640), 16000
        )
        frames = decoded["frames"]
        if len(frames) != len(boxes):
            raise ValueError("Artifact decode differs from AV-HuBERT timeline")
        valid &= boxes[:, 2] >= 8
        hidden = self.backbone.config.hidden_size
        features = np.zeros((len(frames), 2 * hidden), np.float16)
        index = np.flatnonzero(valid)
        batch = int(self.cfg.get("batch_size", 64))
        for start in range(0, len(index), batch):
            chosen = index[start : start + batch]
            crops = np.stack([mouth_crop(frames[i], boxes[i], self.crop) for i in chosen])
            features[chosen] = self.embed(crops)
        if not np.isfinite(features.astype(np.float32)).all():
            raise ValueError("Nonfinite DINOv2 output")
        save_npz(
            output,
            artifact=features,
            times_s=np.arange(len(frames)) / 25,
            artifact_valid=valid,
        )
        meta = dict(
            format="dinov2-mouth-v1",
            step_s=0.04,
            output_stride=box_meta.get("output_stride", 5),
            duration_s=len(frames) / 25,
            feature_signature=self.signature,
            feature_sha256=sha(output),
            source_fingerprint=source,
            box_sha256=box_meta["feature_sha256"],
            box_sample_id=borrowed,
            crop_size=self.crop,
        )
        write_json(output.with_suffix(".json"), meta)
        return meta
