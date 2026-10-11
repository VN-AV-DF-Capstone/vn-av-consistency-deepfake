"""P2 SyncArtifact (đề xuất v001, giữ làm mốc v002): nhánh sync không thấy fake + DINOv2 + gate.

Stage A (chung P1): R học A→V chỉ từ real. Stage S: head sync học real, sham và real
bị dịch lệch tiếng trên cache, rồi đóng băng. Stage C: nhánh artifact + fusion học nhãn
AI, sham = 0. Thiết kế của project, không phải bản tái lập một paper cụ thể.
"""

import copy
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

from vn_av_df import tracking
from vn_av_df.reconstruction import (
    Reconstructor,
    TemporalBlock,
    balanced_order,
    reconstruction_error,
    valid_runs,
)

ARCHITECTURES = ("syncartifact",)


def pool_cells(x, stride):
    """Trung bình mỗi stride frame native thành một ô output; ô cuối có thể ngắn."""
    return torch.stack([x[i : i + stride].mean(0) for i in range(0, len(x), stride)])


def run_temporal(blocks, x, mask):
    """TCN riêng từng đoạn quan sát liên tục; không truyền ngữ cảnh qua gap."""
    out = x.new_zeros(x.shape)
    for left, right in valid_runs(mask):
        out[left:right] = blocks(x[left:right])
    return out


class SyncArtifactDetector(nn.Module):
    """Đầu ra: logit AI theo ô; self.outputs giữ logit sync/artifact làm bằng chứng riêng."""

    def __init__(
        self,
        audio_dim,
        visual_dim,
        architecture="syncartifact",
        hidden=128,
        dropout=0.1,
        native_stride=5,
        artifact_dim=768,
    ):
        super().__init__()
        if architecture not in ARCHITECTURES or native_stride < 1 or hidden < 4:
            raise ValueError("Unknown SyncArtifact variant or invalid stride/hidden")
        self.config = dict(
            audio_dim=audio_dim,
            visual_dim=visual_dim,
            architecture=architecture,
            hidden=hidden,
            dropout=dropout,
            native_stride=native_stride,
            artifact_dim=artifact_dim,
        )
        for name, dim in (("audio", audio_dim), ("visual", visual_dim)):
            self.register_buffer(name + "_mean", torch.zeros(dim))
            self.register_buffer(name + "_std", torch.ones(dim))
        self.register_buffer("reconstruction_ready", torch.tensor(False))
        self.register_buffer("sync_ready", torch.tensor(False))
        self.reconstructor = Reconstructor(audio_dim, visual_dim, hidden)
        self.sync_projection = nn.Sequential(
            nn.Linear(visual_dim + 2, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.sync_temporal = nn.Sequential(*(TemporalBlock(hidden, d, dropout) for d in (1, 2, 4)))
        self.sync_head = nn.Linear(hidden, 1)
        # Frame hiện tại + hiệu frame liền trước: bắt nhấp nháy/không ổn định vùng miệng.
        self.artifact_projection = nn.Sequential(
            nn.Linear(2 * artifact_dim, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.artifact_temporal = nn.Sequential(
            *(TemporalBlock(hidden, d, dropout) for d in (1, 2, 4))
        )
        self.artifact_head = nn.Linear(hidden, 1)
        self.gate = nn.Sequential(nn.Linear(hidden * 2, 64), nn.GELU(), nn.Linear(64, 2))
        self.fusion_input = nn.Sequential(
            nn.Linear(hidden + 2, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.fusion_temporal = nn.Sequential(*(TemporalBlock(hidden, d, dropout) for d in (1, 2)))
        self.classifier = nn.Linear(hidden, 1)
        self.outputs = {}
        self.freeze_reconstruction()

    def sync_modules(self):
        return self.sync_projection, self.sync_temporal, self.sync_head

    def freeze_reconstruction(self):
        """R và nhánh sync luôn đóng băng ngoài stage A/S."""
        self.reconstructor.eval().requires_grad_(False)
        for module in self.sync_modules():
            module.eval().requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.freeze_reconstruction()
        return self

    def normalize(self, audio, visual):
        return (
            (audio - self.audio_mean) / self.audio_std,
            (visual - self.visual_mean) / self.visual_std,
        )

    def output_valid(self, valid):
        """Một ô chỉ hợp lệ nếu mọi frame native của ô được quan sát."""
        stride = self.config["native_stride"]
        return torch.stack([valid[i : i + stride].all() for i in range(0, len(valid), stride)])

    def branches(self, audio, visual, valid, artifact=None, need_artifact=True):
        """Hidden/logit của từng nhánh trên lưới ô; logit ô thiếu quan sát bằng 0."""
        if audio.ndim != 2 or visual.shape[0] != len(audio) or valid.shape != (len(audio),):
            raise ValueError("Expected audio[T,D], visual[T,D], valid[T]")
        stride = self.config["native_stride"]
        mask = self.output_valid(valid)
        if not bool(self.reconstruction_ready):
            raise ValueError("Train real-only reconstruction before detector training")
        if not torch.isfinite(audio[valid]).all() or not torch.isfinite(visual[valid]).all():
            raise ValueError("Nonfinite features")
        a, v = self.normalize(
            audio.masked_fill(~valid[:, None], 0), visual.masked_fill(~valid[:, None], 0)
        )
        error = pool_cells(reconstruction_error(self.reconstructor, a, v, valid), stride)
        sync_hidden = run_temporal(self.sync_temporal, self.sync_projection(error), mask)
        sync_logit = self.sync_head(sync_hidden)[:, 0].masked_fill(~mask, 0)
        artifact_hidden = artifact_logit = None
        if need_artifact:
            if artifact is None or artifact.shape != (len(audio), self.config["artifact_dim"]):
                raise ValueError("P2 needs native artifact features [T, artifact_dim]")
            if not torch.isfinite(artifact[valid]).all():
                raise ValueError("Nonfinite artifact features")
            x = artifact.masked_fill(~valid[:, None], 0)
            delta = torch.zeros_like(x)
            for left, right in valid_runs(valid):
                delta[left + 1 : right] = x[left + 1 : right] - x[left : right - 1]
            native = run_temporal(
                self.artifact_temporal, self.artifact_projection(torch.cat([x, delta], -1)), valid
            )
            artifact_hidden = pool_cells(native, stride)
            artifact_logit = self.artifact_head(artifact_hidden)[:, 0].masked_fill(~mask, 0)
        return sync_hidden, sync_logit, artifact_hidden, artifact_logit, mask

    def forward(self, audio, visual, valid, artifact=None, lip=None):
        if not bool(self.sync_ready):
            raise ValueError("Train the sync stage before detector training")
        sync_hidden, sync_logit, artifact_hidden, artifact_logit, mask = self.branches(
            audio, visual, valid, artifact
        )
        weights = self.gate(torch.cat([sync_hidden, artifact_hidden], -1)).softmax(-1)
        fused = weights[:, :1] * sync_hidden + weights[:, 1:] * artifact_hidden
        evidence = torch.stack([sync_logit, artifact_logit], -1)
        hidden = run_temporal(
            self.fusion_temporal, self.fusion_input(torch.cat([fused, evidence], -1)), mask
        )
        self.outputs = {"sync": sync_logit, "artifact": artifact_logit}
        return self.classifier(hidden)[:, 0].masked_fill(~mask, 0)


def sync_rows(rows):
    """Real (không lệch) và sham có đoạn lệch đã biết; fake không bao giờ vào stage S."""
    result = []
    for row in rows:
        if row["label"] != 0:
            continue
        spans = row.get("av_mismatch_intervals", None if row.get("control_type") else [])
        if spans is not None:
            result.append({**row, "av_mismatch_intervals": spans})
    return result


def shifted(tensors, rng, shift_range, stride):
    """Dịch audio so với hình trên cache (toàn clip hoặc một đoạn); nhãn lệch theo frame."""
    audio, visual, valid = tensors[:3]
    n = len(audio)
    k = rng.choice((-1, 1)) * rng.randint(*shift_range)
    if rng.random() < 0.5 or n < 4 * stride:
        start, end = 0, n
    else:
        length = rng.randint(2 * stride, n // 2)
        start = rng.randrange(0, n - length + 1)
        end = start + length
    source = torch.arange(n, device=audio.device)
    source[start:end] += k
    inside = (source >= 0) & (source < n)
    source = source.clamp(0, n - 1)
    target = torch.zeros(n, device=audio.device)
    target[start:end] = 1
    return (audio[source], visual, valid & valid[source] & inside, *tensors[3:]), target


def fit_sync(model, train, validation, loader, options, seed, top_fraction=0.1):
    """Stage S: chọn head sync theo real/sham validation; không đọc fake hoặc test."""
    from sklearn.metrics import roc_auc_score

    from vn_av_df.dataset import temporal_targets
    from vn_av_df.models import pool_score

    began = time.perf_counter()
    train, validation = sync_rows(train), sync_rows(validation)
    for name, rows in (("train", train), ("validation", validation)):
        if not any(not r.get("control_type") for r in rows):
            raise ValueError(f"Sync stage needs synchronized real clips in {name}")
    stride = model.config["native_stride"]
    shift_range = tuple(options.get("shift_frames", (3, 15)))
    probability = options.get("shift_probability", 0.5)
    if len(shift_range) != 2 or not 1 <= shift_range[0] <= shift_range[1]:
        raise ValueError("shift_frames must be [min, max] with 1 <= min <= max")

    def sample(row, rng, augment):
        tensors, _, times, meta = loader(row)
        tensors = tensors[:3]
        if augment and not row.get("control_type"):
            tensors, frames = shifted(tensors, rng, shift_range, stride)
            return tensors, pool_cells(frames[:, None], stride)[:, 0], 1
        target = temporal_targets(row, times, meta["step_s"], "av_mismatch_intervals")
        return tensors, torch.as_tensor(target, device=tensors[0].device), int(target.max() > 0)

    def loss_for(tensors, target):
        _, logits, _, _, mask = model.branches(*tensors, need_artifact=False)
        observed = mask & (target >= 0)
        if not observed.any():
            return None, None
        loss = F.binary_cross_entropy_with_logits(logits[observed], target[observed])
        return loss, pool_score(logits.sigmoid(), mask, top_fraction)

    def measure():
        # Cố định: mỗi real một bản gốc + một bản dịch lệch theo seed; sham giữ nguyên.
        losses, labels, scores = [], [], []
        for row in validation:
            for augment in (False, True) if not row.get("control_type") else (False,):
                rng = random.Random(f"{seed}-{row['sample_id']}")
                loss, score = loss_for(*sample(row, rng, augment)[:2])
                if loss is not None:
                    label = int(augment) or int(bool(row["av_mismatch_intervals"]))
                    losses.append(float(loss))
                    labels.append(label)
                    scores.append(float(score))
        auc = float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None
        return (sum(losses) / len(losses) if losses else None), auc

    modules = model.sync_modules()
    for module in modules:
        module.requires_grad_(True)
    parameters = [p for module in modules for p in module.parameters()]
    optimizer = torch.optim.AdamW(
        parameters, lr=options.get("lr", 3e-4), weight_decay=options.get("weight_decay", 1e-4)
    )
    best, best_auc, saved, history, stale = float("inf"), None, None, [], 0
    shams = [sum(bool(r.get("control_type")) for r in rows) for rows in (train, validation)]
    print(
        f"Stage S (head sync, không thấy fake): {len(train)} train ({shams[0]} sham), "
        f"{len(validation)} validation ({shams[1]} sham); dịch lệch {list(shift_range)} frame, "
        f"xác suất {probability}",
        flush=True,
    )
    for epoch in range(options.get("epochs", 20)):
        tick = time.perf_counter()
        for module in modules:
            module.train()
        rng = random.Random(seed + epoch)
        losses = []
        for row in balanced_order(train, seed + epoch):
            tensors, target, _ = sample(row, rng, rng.random() < probability)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = loss_for(tensors, target)
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite sync loss")
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 5)
            optimizer.step()
            losses.append(float(loss.detach()))
        for module in modules:
            module.eval()
        with torch.no_grad():
            value, auc = measure()
        if value is None or not losses:
            raise ValueError("No usable sync samples")
        history.append(
            dict(
                epoch=epoch + 1,
                train_loss=sum(losses) / len(losses),
                validation_loss=value,
                validation_auc=auc,
                epoch_seconds=time.perf_counter() - tick,
            )
        )
        tracking.log("stageS", history[-1])
        if value < best:
            best, best_auc, stale = value, auc, 0
            saved = [copy.deepcopy(module.state_dict()) for module in modules]
        else:
            stale += 1
        if options.get("patience", 5) and stale >= options.get("patience", 5):
            break
    if saved is None:
        raise ValueError("Sync epochs must be positive")
    for module, state in zip(modules, saved):
        module.load_state_dict(state)
    model.sync_ready.fill_(True)
    model.freeze_reconstruction()
    return dict(
        history=history,
        best_validation_loss=best,
        best_validation_auc=best_auc,
        train_ids=[r["sample_id"] for r in train],
        validation_ids=[r["sample_id"] for r in validation],
        sham_train=sum(bool(r.get("control_type")) for r in train),
        sham_validation=sum(bool(r.get("control_type")) for r in validation),
        shift_frames=list(shift_range),
        shift_probability=probability,
        note="Validation AUC: real (0) vs sham/shifted real (1); fake never used in stage S",
        elapsed_s=time.perf_counter() - began,
    )
