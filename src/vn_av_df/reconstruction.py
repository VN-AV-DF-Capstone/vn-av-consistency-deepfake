"""Stage A của P2: tái dựng A→V chỉ từ real ở 25 Hz, rồi đóng băng.

Lấy cảm hứng residual tái dựng của AuViRe, không phải implementation chính thức.
Ở v001 R không vượt dự đoán hằng số; v002 giữ để train lại P2 làm mốc so sánh.
"""

import copy
import time
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from vn_av_df import tracking


def valid_runs(valid):
    """Trả các khoảng [start,end) liên tục để tránh truyền qua missing data."""
    edge = torch.diff(
        torch.cat([valid.new_tensor([False]), valid, valid.new_tensor([False])]).int()
    )
    return zip(torch.where(edge == 1)[0].tolist(), torch.where(edge == -1)[0].tolist())


def reconstruction_error(reconstructor, a, v, valid):
    """Residual |V−R(A)| + MSE + cosine distance theo frame; R không nhận gradient."""
    error = v.new_zeros((len(v), v.shape[1] + 2))
    with torch.no_grad():
        for left, right in valid_runs(valid):
            predicted = reconstructor(a[left:right])
            delta = v[left:right] - predicted
            error[left:right] = torch.cat(
                [
                    delta.abs(),
                    delta.square().mean(-1, keepdim=True),
                    (1 - F.cosine_similarity(v[left:right], predicted))[:, None],
                ],
                -1,
            )
    return error


class TemporalBlock(nn.Module):
    """Conv giãn theo thời gian; LayerNorm chỉ chuẩn hóa chiều feature."""

    def __init__(self, width, dilation, dropout=0):
        super().__init__()
        self.conv = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(F.gelu(self.norm(self.conv(x.T[None])[0].T)))


class Reconstructor(nn.Module):
    """A→V; không nhận V target, tránh nghiệm sao chép target."""

    def __init__(self, audio_dim, visual_dim, width):
        super().__init__()
        self.input = nn.Linear(audio_dim, width)
        self.blocks = nn.Sequential(*(TemporalBlock(width, d) for d in (1, 2, 4)))
        self.output = nn.Linear(width, visual_dim)

    def forward(self, audio):
        return self.output(self.blocks(F.gelu(self.input(audio))))


def balanced_order(rows, seed):
    """Mỗi parent lấy một variant mỗi epoch, chọn ngẫu nhiên có seed."""
    import random

    groups = defaultdict(list)
    for row in rows:
        groups[row.get("source_clip_id", row["sample_id"])].append(row)
    rng = random.Random(seed)
    result = [rng.choice(group) for _, group in sorted(groups.items())]
    rng.shuffle(result)
    return result


def fit_reconstruction(model, train, validation, loader, options, seed):
    """Fit stats bằng real train; chọn R bằng real validation; không đọc test."""
    began = time.perf_counter()

    def eligible(row):
        return (
            row["label"] == 0
            and not row.get("control_type")
            and row.get("sync_status", "reviewed_match") == "reviewed_match"
        )

    train, validation = [r for r in train if eligible(r)], [r for r in validation if eligible(r)]
    if not train or not validation:
        raise ValueError("Reconstruction needs synchronized real train and validation")
    # Mỗi parent đóng góp cùng trọng số vào moments, không giữ toàn bộ dataset trên GPU.
    moments = [[], []]
    for row in balanced_order(train, seed):
        tensors = loader(row)[:3]
        for index in (0, 1):
            values = tensors[index][tensors[2]]
            if len(values):
                moments[index].append((values.mean(0), values.square().mean(0)))
    for index, name in enumerate(("audio", "visual")):
        if not moments[index]:
            raise ValueError("No valid real reconstruction frames")
        mean = torch.stack([x[0] for x in moments[index]]).mean(0)
        variance = torch.stack([x[1] for x in moments[index]]).mean(0) - mean.square()
        getattr(model, name + "_mean").copy_(mean)
        getattr(model, name + "_std").copy_(variance.clamp_min(1e-4).sqrt())
    model.reconstructor.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.reconstructor.parameters(),
        lr=options.get("lr", 3e-4),
        weight_decay=options.get("weight_decay", 1e-4),
    )

    def loss_for(row, constant=False):
        audio, visual, valid = loader(row)[:3]
        a, v = model.normalize(audio, visual)
        terms = [
            F.smooth_l1_loss(
                torch.zeros_like(v[left:right]) if constant else model.reconstructor(a[left:right]),
                v[left:right],
            )
            * (right - left)
            for left, right in valid_runs(valid)
        ]
        return sum(terms) / valid.sum() if terms else None

    with torch.no_grad():
        constants = [loss_for(row, True) for row in validation]
        constants = [float(x) for x in constants if x is not None]
    if not constants:
        raise ValueError("No valid real validation frames")
    best, saved, history, stale = float("inf"), None, [], 0
    print(
        f"Stage A (tái dựng A→V, real-only): {len(train)} train, {len(validation)} validation; "
        f"loss hằng số trên validation {sum(constants) / len(constants):.4f}",
        flush=True,
    )
    for epoch in range(options.get("epochs", 20)):
        tick = time.perf_counter()
        model.reconstructor.train()
        losses = []
        for row in balanced_order(train, seed + epoch):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(row)
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite reconstruction loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.reconstructor.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.reconstructor.eval()
        with torch.no_grad():
            values = [loss_for(row) for row in validation]
            values = [float(x) for x in values if x is not None]
        if not values or not losses:
            raise ValueError("No usable reconstruction samples")
        score = sum(values) / len(values)
        history.append(
            dict(
                epoch=epoch + 1,
                train_loss=sum(losses) / len(losses),
                validation_loss=score,
                epoch_seconds=time.perf_counter() - tick,
            )
        )
        tracking.log("stageA", history[-1])
        if score < best:
            best, stale = score, 0
            saved = copy.deepcopy(model.reconstructor.state_dict())
        else:
            stale += 1
        if options.get("patience", 5) and stale >= options.get("patience", 5):
            break
    if saved is None:
        raise ValueError("Reconstruction epochs must be positive")
    model.reconstructor.load_state_dict(saved)
    model.reconstruction_ready.fill_(True)
    model.freeze_reconstruction()
    return dict(
        history=history,
        best_validation_loss=best,
        constant_validation_loss=sum(constants) / len(constants),
        train_ids=[r["sample_id"] for r in train],
        validation_ids=[r["sample_id"] for r in validation],
        elapsed_s=time.perf_counter() - began,
    )
