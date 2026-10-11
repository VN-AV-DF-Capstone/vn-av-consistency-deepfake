"""Head thời gian TCN của baseline B-AVH trên feature đóng băng, và hàm dựng mọi model.

Không phải bản tái hiện AuViRe; B-AVH gần ý AV-Lip-Sync+ (AV-HuBERT + TCN).
"""

import math

import torch
from torch import nn


class TemporalDetector(nn.Module):
    """Head TCN; input là một clip [T,D], mask [T]."""

    def __init__(self, audio_dim, visual_dim, architecture="tcn", hidden=128, dropout=0.1):
        super().__init__()
        if architecture != "tcn" or hidden < 4 or hidden % 4:
            raise ValueError("Unknown architecture or hidden size not divisible by four")
        self.config = dict(
            audio_dim=audio_dim,
            visual_dim=visual_dim,
            architecture=architecture,
            hidden=hidden,
            dropout=dropout,
        )
        self.fusion = nn.Sequential(
            nn.Linear(audio_dim + visual_dim, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.temporal = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(hidden, hidden, 3, padding=d, dilation=d),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for d in (1, 2, 4)
        )
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, audio, visual, valid, artifact=None, lip=None):
        """Không cho temporal context đi xuyên qua đoạn thiếu quan sát."""
        if audio.ndim != 2 or visual.shape[0] != len(audio) or valid.shape != (len(audio),):
            raise ValueError("Expected one clip: audio[T,D], visual[T,D], valid[T]")
        if not torch.isfinite(audio[valid]).all() or not torch.isfinite(visual[valid]).all():
            raise ValueError("Nonfinite features")
        x = self.fusion(
            torch.cat(
                [audio.masked_fill(~valid[:, None], 0), visual.masked_fill(~valid[:, None], 0)], -1
            )
        )
        # Mỗi đoạn quan sát liên tục xử lý riêng: thiếu bằng chứng thì không truyền qua thời gian.
        edges = torch.diff(
            torch.cat([valid.new_tensor([False]), valid, valid.new_tensor([False])]).int()
        )
        result = x.new_zeros(len(x))
        for a, b in zip(torch.where(edges == 1)[0].tolist(), torch.where(edges == -1)[0].tolist()):
            z = x[a:b][None].transpose(1, 2)
            for block in self.temporal:
                z = z + block(z)
            result[a:b] = self.classifier(z.transpose(1, 2))[0, :, 0]
        return result


def pool_score(scores, valid, top_fraction=0.1):
    """Lấy trung bình top-k ô hợp lệ; không có quan sát thì trả None."""
    values = scores[valid]
    if not len(values):
        return None
    return values.topk(max(1, math.ceil(len(values) * top_fraction))).values.mean()


def build_model(
    audio_dim,
    visual_dim,
    architecture="tcn",
    hidden=128,
    dropout=0.1,
    native_stride=1,
    artifact_dim=None,
    **options,
):
    """Khôi phục đúng kiến trúc từ checkpoint, không đổi model âm thầm."""
    if architecture == "syncartifact":
        from vn_av_df.syncartifact import SyncArtifactDetector

        if artifact_dim is None:
            raise ValueError("P2 needs artifact features")
        return SyncArtifactDetector(
            audio_dim, visual_dim, architecture, hidden, dropout, native_stride, artifact_dim
        )
    if architecture in ("tricue", "artifact"):
        from vn_av_df.tricue import TriCue

        return TriCue(
            audio_dim,
            visual_dim,
            architecture,
            hidden,
            dropout,
            native_stride,
            artifact_dim,
            **options,
        )
    if options:
        raise ValueError(f"Unexpected options for {architecture}: {sorted(options)}")
    return TemporalDetector(audio_dim, visual_dim, architecture, hidden, dropout)


def output_valid(model, valid):
    """Mask đầu ra có thể thưa hơn input native 25 Hz."""
    return model.output_valid(valid) if hasattr(model, "output_valid") else valid
