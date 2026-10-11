"""TriCue (v002): ba bằng chứng → hai đầu ra "lệch tiếng?" và "AI vẽ?".

- m_sync: hồ sơ cos(aₜ, vₜ₊ₖ) AV-HuBERT, k = −K…K → TCN phân loại độ lệch (+ lớp "không khớp").
  Học từ real tự dịch lệch + sham, không thấy fake, rồi đóng băng.
- m_lead: tương quan chéo độ mở môi ↔ đường bao tiếng quanh điểm bắt đầu nói; điểm = khoảng cách
  Mahalanobis so với real train. Không train, chỉ thống kê real.
- r_art: DINOv2 crop miệng → TCN (như nhánh artifact của P2); học real vs fake của method
  (MechFake, Wav2Lip hoặc cả hai).
- Hợp điểm: logistic trọng số không âm trên điểm từng nhánh; nhánh học riêng nên không nhánh nào
  lấn át qua gate. Kiến trúc "artifact" chỉ có r_art (phân tích cơ chế MechFake).

Thiết kế của project, không phải bản tái lập paper.
"""

import copy
import math
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from vn_av_df import tracking
from vn_av_df.reconstruction import TemporalBlock, balanced_order, valid_runs

BRANCHES = ("sync", "lead", "artifact")
IGNORE = -100


def pool_cells(x, stride):
    return torch.stack([x[i : i + stride].mean(0) for i in range(0, len(x), stride)])


def run_temporal(blocks, x, mask):
    """TCN riêng từng đoạn quan sát liên tục; không truyền ngữ cảnh qua gap."""
    out = x.new_zeros(x.shape)
    for left, right in valid_runs(mask):
        out[left:right] = blocks(x[left:right])
    return out


def run_ids(valid):
    """Số thứ tự đoạn quan sát liên tục của mỗi frame (-1 = không quan sát)."""
    ids = torch.full(valid.shape, -1, dtype=torch.long, device=valid.device)
    for number, (left, right) in enumerate(valid_runs(valid)):
        ids[left:right] = number
    return ids


def lead_profiles(lip, valid, window, lags, quiet=5):
    """Hồ sơ tương quan chéo quanh mỗi điểm bắt đầu nói: [(frame, profile[2·lags+1])].

    Điểm bắt đầu nói: đường bao log vượt trung vị đoạn sau ít nhất `quiet` frame dưới trung vị.
    corr(độ mở môi[t], đường bao[t+l]) cao ở l > 0 nghĩa là môi đi trước tiếng.
    """
    lip = lip.detach().cpu().numpy() if torch.is_tensor(lip) else np.asarray(lip)
    valid = valid.detach().cpu().numpy() if torch.is_tensor(valid) else np.asarray(valid)
    observed = valid & (lip[:, 2] > 0)
    edges = np.diff(np.r_[False, observed, False].astype(int))
    result = []
    for left, right in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        aperture, envelope = lip[left:right, 0], lip[left:right, 1]
        if len(envelope) < 2 * (window + lags) + quiet + 1:
            continue
        active = envelope > np.median(envelope)
        for t in range(quiet + window + lags, len(envelope) - window - lags):
            if not active[t] or active[t - quiet : t].any():
                continue
            x = aperture[t - window : t + window + 1]
            profile = np.zeros(2 * lags + 1, np.float32)
            for i, lag in enumerate(range(-lags, lags + 1)):
                y = envelope[t - window + lag : t + window + 1 + lag]
                if x.std() > 1e-6 and y.std() > 1e-6:
                    profile[i] = np.corrcoef(x, y)[0, 1]
            result.append((left + t, profile))
    return result


class TriCue(nn.Module):
    """Đầu ra: logit "AI vẽ?" theo ô; self.outputs giữ điểm từng nhánh và logit "lệch tiếng?"."""

    def __init__(
        self,
        audio_dim,
        visual_dim,
        architecture="tricue",
        hidden=128,
        dropout=0.1,
        native_stride=5,
        artifact_dim=768,
        branches=BRANCHES,
        max_offset=15,
        lead_window=12,
        lead_lags=8,
    ):
        super().__init__()
        branches = ("artifact",) if architecture == "artifact" else tuple(branches)
        if (
            architecture not in ("tricue", "artifact")
            or not branches
            or set(branches) - set(BRANCHES)
            or native_stride < 1
            or hidden < 4
        ):
            raise ValueError("Invalid TriCue configuration")
        if "artifact" in branches and not artifact_dim:
            raise ValueError("Artifact branch needs artifact features")
        self.config = dict(
            audio_dim=audio_dim,
            visual_dim=visual_dim,
            architecture=architecture,
            hidden=hidden,
            dropout=dropout,
            native_stride=native_stride,
            artifact_dim=artifact_dim,
            branches=list(branches),
            max_offset=max_offset,
            lead_window=lead_window,
            lead_lags=lead_lags,
        )
        self.branch_names = [b for b in BRANCHES if b in branches]
        for name in ("sync", "lead", "artifact", "fusion"):
            self.register_buffer(f"{name}_ready", torch.tensor(False))
        if "sync" in branches:
            for name, dim in (("audio", audio_dim), ("visual", visual_dim)):
                self.register_buffer(name + "_mean", torch.zeros(dim))
                self.register_buffer(name + "_std", torch.ones(dim))
            classes = 2 * max_offset + 1
            self.sync_projection = nn.Sequential(
                nn.Linear(classes, hidden), nn.LayerNorm(hidden), nn.GELU()
            )
            self.sync_temporal = nn.Sequential(
                *(TemporalBlock(hidden, d, dropout) for d in (1, 2, 4))
            )
            self.sync_head = nn.Linear(hidden, classes + 1)  # Lớp cuối = "không khớp".
        if "lead" in branches:
            size = 2 * lead_lags + 1
            self.register_buffer("lead_mean", torch.zeros(size))
            self.register_buffer("lead_precision", torch.eye(size))
            self.register_buffer("lead_scale", torch.tensor([0.0, 1.0]))
        if "artifact" in branches:
            self.artifact_projection = nn.Sequential(
                nn.Linear(2 * artifact_dim, hidden), nn.LayerNorm(hidden), nn.GELU()
            )
            self.artifact_temporal = nn.Sequential(
                *(TemporalBlock(hidden, d, dropout) for d in (1, 2, 4))
            )
            self.artifact_head = nn.Linear(hidden, 1)
        # Hợp điểm: trọng số qua softplus nên không âm (bằng chứng nghi hơn → điểm không giảm).
        self.ai_weight = nn.Parameter(torch.zeros(len(self.branch_names)))
        self.ai_bias = nn.Parameter(torch.zeros(()))
        mismatch = [b for b in self.branch_names if b in ("sync", "lead")]
        self.mismatch_names = mismatch
        self.mismatch_weight = nn.Parameter(torch.zeros(len(mismatch)))
        self.mismatch_bias = nn.Parameter(torch.zeros(()))
        if architecture == "artifact":
            self.fusion_ready.fill_(True)
        self.outputs = {}
        self.freeze(*self.branch_names)

    # ---- trạng thái học ----
    def branch_modules(self, name):
        if name == "sync":
            return [self.sync_projection, self.sync_temporal, self.sync_head]
        if name == "artifact":
            return [self.artifact_projection, self.artifact_temporal, self.artifact_head]
        return []

    def freeze(self, *names):
        for name in names:
            for module in self.branch_modules(name):
                module.eval().requires_grad_(False)

    def unfreeze(self, name):
        for module in self.branch_modules(name):
            module.train().requires_grad_(True)

    def fusion_parameters(self):
        return [self.ai_weight, self.ai_bias, self.mismatch_weight, self.mismatch_bias]

    def train(self, mode=True):
        super().train(mode)
        self.freeze(*self.branch_names)  # Nhánh chỉ học trong stage riêng của nó.
        return self

    def output_valid(self, valid):
        stride = self.config["native_stride"]
        return torch.stack([valid[i : i + stride].all() for i in range(0, len(valid), stride)])

    # ---- từng nhánh ----
    def sync_profile(self, audio, visual, valid):
        """[T, 2K+1]: cos(aₜ, vₜ₊ₖ) chỉ trong cùng đoạn quan sát liên tục, ngoài ra 0."""
        a = (audio - self.audio_mean) / self.audio_std
        v = (visual - self.visual_mean) / self.visual_std
        a = F.normalize(a.masked_fill(~valid[:, None], 0), dim=-1)
        v = F.normalize(v.masked_fill(~valid[:, None], 0), dim=-1)
        similarity = a @ v.T
        n, k = len(a), self.config["max_offset"]
        rows = torch.arange(n, device=a.device)[:, None]
        cols = rows + torch.arange(-k, k + 1, device=a.device)[None]
        ids = run_ids(valid)
        inside = (cols >= 0) & (cols < n)
        clamped = cols.clamp(0, n - 1)
        same = inside & (ids[:, None] >= 0) & (ids[clamped] == ids[:, None])
        return similarity.gather(1, clamped).masked_fill(~same, 0)

    def sync_logits(self, audio, visual, valid):
        """Logit lớp độ lệch theo ô [C, 2K+2]."""
        if audio is None or visual is None:
            raise ValueError("Sync branch needs AV-HuBERT features")
        if not torch.isfinite(audio[valid]).all() or not torch.isfinite(visual[valid]).all():
            raise ValueError("Nonfinite features")
        mask = self.output_valid(valid)
        profile = pool_cells(self.sync_profile(audio, visual, valid), self.config["native_stride"])
        hidden = run_temporal(self.sync_temporal, self.sync_projection(profile), mask)
        return self.sync_head(hidden), mask

    def sync_evidence(self, logits):
        """Logit "lệch": 1 − P(|độ lệch| ≤ 1 frame)."""
        k = self.config["max_offset"]
        probability = logits.softmax(-1)
        synced = probability[:, k - 1 : k + 2].sum(-1).clamp(1e-6, 1 - 1e-6)
        return torch.log1p(-synced) - torch.log(synced)

    def lead_evidence(self, lip, valid, cells):
        """Điểm z bất thường sớm–muộn theo ô và mask ô có điểm bắt đầu nói."""
        stride, window = self.config["native_stride"], self.config["lead_window"]
        score = torch.zeros(cells, device=valid.device)
        mask = torch.zeros(cells, dtype=torch.bool, device=valid.device)
        if lip is None:
            raise ValueError("Lead branch needs lip features (AV-HuBERT cache v2)")
        for frame, profile in lead_profiles(lip, valid, window, self.config["lead_lags"]):
            delta = torch.as_tensor(profile, device=valid.device) - self.lead_mean
            distance = (delta @ self.lead_precision @ delta).clamp_min(0).sqrt()
            z = (distance - self.lead_scale[0]) / self.lead_scale[1]
            first = max(0, (frame - window) // stride)
            last = min(cells - 1, (frame + window) // stride)
            region = slice(first, last + 1)
            score[region] = torch.where(mask[region], torch.maximum(score[region], z), z)
            mask[region] = True
        return score, mask

    def artifact_logits(self, artifact, valid):
        if artifact is None or artifact.shape[-1] != self.config["artifact_dim"]:
            raise ValueError("Artifact branch needs native artifact features [T, artifact_dim]")
        if not torch.isfinite(artifact[valid]).all():
            raise ValueError("Nonfinite artifact features")
        mask = self.output_valid(valid)
        x = artifact.masked_fill(~valid[:, None], 0)
        delta = torch.zeros_like(x)
        for left, right in valid_runs(valid):
            delta[left + 1 : right] = x[left + 1 : right] - x[left : right - 1]
        native = run_temporal(
            self.artifact_temporal, self.artifact_projection(torch.cat([x, delta], -1)), valid
        )
        hidden = pool_cells(native, self.config["native_stride"])
        return self.artifact_head(hidden)[:, 0].masked_fill(~mask, 0), mask

    def evidence(self, audio, visual, valid, artifact=None, lip=None):
        """Điểm từng nhánh theo ô (0 ở ô không có bằng chứng) và mask ô hợp lệ."""
        mask = self.output_valid(valid)
        result = {}
        for name in self.branch_names:
            if not bool(getattr(self, f"{name}_ready")):
                raise ValueError(f"Train the {name} stage before using TriCue")
        if "sync" in self.branch_names:
            logits, _ = self.sync_logits(audio, visual, valid)
            result["sync"] = self.sync_evidence(logits).masked_fill(~mask, 0)
        if "lead" in self.branch_names:
            score, has = self.lead_evidence(lip, valid, len(mask))
            result["lead"] = score.masked_fill(~(has & mask), 0)
        if "artifact" in self.branch_names:
            result["artifact"], _ = self.artifact_logits(artifact, valid)
        return result, mask

    def combine(self, evidence, mask):
        """(logit AI, logit lệch tiếng hoặc None) theo ô từ điểm từng nhánh."""
        stacked = torch.stack([evidence[name] for name in self.branch_names], -1)
        ai = stacked @ F.softplus(self.ai_weight) + self.ai_bias
        mismatch = None
        if self.mismatch_names:
            parts = torch.stack([evidence[name] for name in self.mismatch_names], -1)
            mismatch = parts @ F.softplus(self.mismatch_weight) + self.mismatch_bias
            mismatch = mismatch.masked_fill(~mask, 0)
        return ai.masked_fill(~mask, 0), mismatch

    def forward(self, audio, visual, valid, artifact=None, lip=None):
        evidence, mask = self.evidence(audio, visual, valid, artifact, lip)
        if self.config["architecture"] == "artifact":
            self.outputs = {"artifact": evidence["artifact"]}
            return evidence["artifact"]
        if not bool(self.fusion_ready):
            raise ValueError("Train the fusion stage before using TriCue")
        ai, mismatch = self.combine(evidence, mask)
        self.outputs = {**evidence, "mismatch": mismatch}
        return ai


# ---------------------------------------------------------------------------------------------
# Các stage học. loader(row) → (tensors[audio, visual, valid, artifact, lip], target, times, meta)
# ---------------------------------------------------------------------------------------------


def synchronized_real(row):
    return (
        row["label"] == 0
        and not row.get("control_type")
        and not row.get("diagnostic")
        and row.get("sync_status", "reviewed_match") == "reviewed_match"
    )


def has_primary(row):
    """Mẫu MechFake ngoài tập con hợp điểm chỉ có cache DINOv2 (feature_scope = artifact)."""
    return row.get("feature_scope", "full") != "artifact"


def fit_statistics(model, rows, loader, seed):
    """μ/σ AV-HuBERT từ real train; mỗi parent cùng trọng số."""
    moments = [[], []]
    for row in balanced_order([r for r in rows if synchronized_real(r)], seed):
        tensors = loader(row)[0]
        for index in (0, 1):
            values = tensors[index][tensors[2]]
            if len(values):
                moments[index].append((values.mean(0), values.square().mean(0)))
    for index, name in enumerate(("audio", "visual")):
        if not moments[index]:
            raise ValueError("No synchronized real frames for normalization")
        mean = torch.stack([x[0] for x in moments[index]]).mean(0)
        variance = torch.stack([x[1] for x in moments[index]]).mean(0) - mean.square()
        getattr(model, name + "_mean").copy_(mean)
        getattr(model, name + "_std").copy_(variance.clamp_min(1e-4).sqrt())


def offset_sample(model, row, tensors, rng, shift_range, probability, augment=True):
    """Input (có thể đã dịch tiếng) và nhãn lớp độ lệch theo ô; ô biên đoạn bị bỏ."""
    audio, visual, valid = tensors[0], tensors[1], tensors[2]
    k_max, stride = model.config["max_offset"], model.config["native_stride"]
    n = len(audio)
    cells = math.ceil(n / stride)
    target = torch.full((cells,), k_max, dtype=torch.long, device=audio.device)
    spans = row.get("av_mismatch_intervals") or []
    if row.get("control_type"):
        # Sham: đoạn tiếng ghép từ clip khác → lớp "không khớp"; ô dính biên bỏ qua.
        for a, b in spans:
            first, last = a * 25 / stride, b * 25 / stride
            for cell in range(int(first), min(cells, math.ceil(last))):
                inside = min(cell + 1, last) - max(cell, first)
                target[cell] = 2 * k_max + 1 if inside >= 1 - 1e-6 else IGNORE
        return (audio, visual, valid), target, int(bool(spans))
    if not augment or rng.random() >= probability:
        return (audio, visual, valid), target, 0
    shift = rng.choice((-1, 1)) * rng.randint(*shift_range)
    if rng.random() < 0.5 or n < 4 * stride:
        start, end = 0, n
    else:
        length = rng.randint(2 * stride, n // 2)
        start = rng.randrange(0, n - length + 1)
        end = start + length
    source = torch.arange(n, device=audio.device)
    source[start:end] += shift
    inside = (source >= 0) & (source < n)
    source = source.clamp(0, n - 1)
    moved = valid & valid[source] & inside
    for cell in range(cells):
        a, b = cell * stride, min(n, (cell + 1) * stride)
        if start <= a and b <= end:
            target[cell] = shift + k_max
        elif not (b <= start or end <= a):
            target[cell] = IGNORE
    return (audio[source], visual, moved), target, 1


def fit_sync(model, train, validation, loader, options, seed, top_fraction=0.1):
    """Stage S: phân loại độ lệch trên real tự dịch + sham; không đọc fake hoặc test."""
    from sklearn.metrics import roc_auc_score

    from vn_av_df.models import pool_score

    began = time.perf_counter()
    pick = [r for r in train if has_primary(r) and r["label"] == 0 and not r.get("diagnostic")]
    held = [r for r in validation if has_primary(r) and r["label"] == 0]
    if not any(synchronized_real(r) for r in pick) or not any(synchronized_real(r) for r in held):
        raise ValueError("Sync stage needs synchronized real clips in train and validation")
    k_max = model.config["max_offset"]
    low, high = options.get("shift_frames", (2, k_max))
    if not 1 <= low <= high <= k_max:
        raise ValueError("shift_frames must satisfy 1 <= min <= max <= max_offset")
    probability = options.get("shift_probability", 0.5)

    def loss_for(row, rng, augment):
        tensors = loader(row)[0]
        inputs, target, label = offset_sample(
            model, row, tensors, rng, (low, high), probability, augment
        )
        logits, mask = model.sync_logits(*inputs)
        keep = mask & (target != IGNORE)
        if not keep.any():
            return None, None, label
        loss = F.cross_entropy(logits[keep], target[keep])
        score = pool_score(model.sync_evidence(logits).sigmoid(), mask, top_fraction)
        return loss, score, label

    def measure():
        losses, labels, scores = [], [], []
        for row in held:
            for augment in (False, True) if synchronized_real(row) else (False,):
                rng = random.Random(f"{seed}-{row['sample_id']}")
                loss, score, label = loss_for(row, rng, augment)
                if loss is not None:
                    losses.append(float(loss))
                    labels.append(label)
                    scores.append(float(score))
        auc = float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None
        return (sum(losses) / len(losses) if losses else None), auc

    model.unfreeze("sync")
    parameters = [p for m in model.branch_modules("sync") for p in m.parameters()]
    optimizer = torch.optim.AdamW(
        parameters, lr=options.get("lr", 3e-4), weight_decay=options.get("weight_decay", 1e-4)
    )
    best, best_auc, saved, history, stale = float("inf"), None, None, [], 0
    print(
        f"Stage S (độ lệch, không thấy fake): {len(pick)} train, {len(held)} validation; "
        f"dịch lệch {low}-{high} frame, xác suất {probability}",
        flush=True,
    )
    for epoch in range(options.get("epochs", 20)):
        tick = time.perf_counter()
        model.unfreeze("sync")
        rng = random.Random(seed + epoch)
        losses = []
        for row in balanced_order(pick, seed + epoch):
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = loss_for(row, rng, True)
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite sync loss")
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 5)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.freeze("sync")
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
            saved = [copy.deepcopy(m.state_dict()) for m in model.branch_modules("sync")]
        else:
            stale += 1
        if options.get("patience", 5) and stale >= options.get("patience", 5):
            break
    if saved is None:
        raise ValueError("Sync epochs must be positive")
    for module, state in zip(model.branch_modules("sync"), saved):
        module.load_state_dict(state)
    model.freeze("sync")
    model.sync_ready.fill_(True)
    return dict(
        history=history,
        best_validation_loss=best,
        best_validation_auc=best_auc,
        train_ids=[r["sample_id"] for r in pick],
        validation_ids=[r["sample_id"] for r in held],
        shift_frames=[low, high],
        shift_probability=probability,
        note="AUC: real (0) vs real dịch lệch / sham (1); fake không bao giờ vào stage S",
        elapsed_s=time.perf_counter() - began,
    )


def fit_lead(model, train, loader, options):
    """Stage L: phân phối hồ sơ sớm–muộn của real train (không train tham số)."""
    began = time.perf_counter()
    window, lags = model.config["lead_window"], model.config["lead_lags"]
    profiles, counted = [], 0
    for row in train:
        if not (synchronized_real(row) and has_primary(row)):
            continue
        tensors = loader(row)[0]
        if tensors[4] is None:
            raise ValueError("Lead stage needs lip features (re-extract AV-HuBERT cache v2)")
        found = lead_profiles(tensors[4], tensors[2], window, lags)
        profiles.extend(p for _, p in found)
        counted += 1
    size = 2 * lags + 1
    if len(profiles) < size + 5:
        raise ValueError(f"Too few speech onsets in real train ({len(profiles)})")
    data = np.stack(profiles).astype(np.float64)
    mean = data.mean(0)
    covariance = np.cov(data, rowvar=False)
    shrink = float(options.get("shrinkage", 0.1))
    covariance = (1 - shrink) * covariance + shrink * np.trace(covariance) / size * np.eye(size)
    precision = np.linalg.inv(covariance)
    distances = np.sqrt(
        np.maximum(np.einsum("ni,ij,nj->n", data - mean, precision, data - mean), 0)
    )
    model.lead_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
    model.lead_precision.copy_(torch.as_tensor(precision, dtype=torch.float32))
    model.lead_scale.copy_(
        torch.tensor([distances.mean(), max(distances.std(), 1e-6)], dtype=torch.float32)
    )
    model.lead_ready.fill_(True)
    print(f"Stage L (sớm–muộn, real-only): {len(profiles)} điểm bắt đầu nói / {counted} clip")
    return dict(
        onsets=len(profiles),
        clips=counted,
        mean_profile=mean.tolist(),
        shrinkage=shrink,
        elapsed_s=time.perf_counter() - began,
    )


def fit_artifact(model, train, validation, loader, options, seed):
    """Stage R: nhánh dấu vết vẽ, BCE theo ô + điểm video, chọn theo loss validation."""
    from vn_av_df.experiment import objective

    began = time.perf_counter()

    def loss_for(row):
        tensors, target, _, _ = loader(row, need_primary=False)
        logits, mask = model.artifact_logits(tensors[3], tensors[2])
        return objective(logits, mask, target, row, options["top_fraction"])

    model.unfreeze("artifact")
    parameters = [p for m in model.branch_modules("artifact") for p in m.parameters()]
    optimizer = torch.optim.AdamW(
        parameters, lr=options["lr"], weight_decay=options["weight_decay"]
    )
    best, saved, history, stale = float("inf"), None, [], 0
    print(f"Stage R (dấu vết vẽ): {len(train)} train, {len(validation)} validation", flush=True)
    for epoch in range(options["epochs"]):
        tick = time.perf_counter()
        model.unfreeze("artifact")
        losses = []
        for row in balanced_order(train, seed + epoch):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(row)
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite artifact loss")
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 5)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.freeze("artifact")
        with torch.no_grad():
            values = [loss_for(row) for row in validation]
            values = [float(v) for v in values if v is not None]
        if not values or not losses:
            raise ValueError("No usable artifact samples")
        value = sum(values) / len(values)
        history.append(
            dict(
                epoch=epoch + 1,
                train_loss=sum(losses) / len(losses),
                validation_loss=value,
                epoch_seconds=time.perf_counter() - tick,
            )
        )
        tracking.log("stageR", history[-1])
        if value < best:
            best, stale = value, 0
            saved = [copy.deepcopy(m.state_dict()) for m in model.branch_modules("artifact")]
        else:
            stale += 1
        if options["patience"] and stale >= options["patience"]:
            break
    if saved is None:
        raise ValueError("Artifact epochs must be positive")
    for module, state in zip(model.branch_modules("artifact"), saved):
        module.load_state_dict(state)
    model.freeze("artifact")
    model.artifact_ready.fill_(True)
    return dict(
        history=history,
        best_validation_loss=best,
        train_samples=len(train),
        validation_samples=len(validation),
        elapsed_s=time.perf_counter() - began,
    )


def fit_fusion(model, train, validation, loader, options, seed):
    """Stage F: trọng số hợp điểm trên bằng chứng đã đóng băng (tính một lần mỗi mẫu)."""
    from vn_av_df.dataset import temporal_targets
    from vn_av_df.models import pool_score

    began = time.perf_counter()
    top = options["top_fraction"]

    def collect(rows):
        items = []
        with torch.no_grad():
            for row in rows:
                if not has_primary(row):
                    continue
                tensors, target, times, meta = loader(row)
                evidence, mask = model.evidence(*tensors)
                spans = temporal_targets(row, times, meta["step_s"], "av_mismatch_intervals")
                items.append(
                    (row, evidence, mask, target, torch.as_tensor(spans, device=mask.device))
                )
        return items

    def loss_for(item):
        row, evidence, mask, target, mismatch_target = item
        ai, mismatch = model.combine(evidence, mask)
        observed = mask & (target >= 0)
        terms = []
        if observed.any():
            terms.append(F.binary_cross_entropy_with_logits(ai[observed], target[observed]))
        score = pool_score(ai.sigmoid(), mask, top)
        if score is not None:
            terms.append(
                F.binary_cross_entropy(
                    score.clamp(1e-6, 1 - 1e-6), score.new_tensor(row["label"] * 1.0)
                )
            )
        known = mask & (mismatch_target >= 0)
        if mismatch is not None and known.any():
            terms.append(
                F.binary_cross_entropy_with_logits(mismatch[known], mismatch_target[known])
            )
        return sum(terms) if terms else None

    train_items, validation_items = collect(train), collect(validation)
    if not train_items or not validation_items:
        raise ValueError("Fusion needs samples with full features in train and validation")
    parameters = model.fusion_parameters()
    optimizer = torch.optim.Adam(parameters, lr=options.get("fusion_lr", 0.05))
    best, saved, history, stale = float("inf"), None, [], 0
    epochs, patience = options.get("fusion_epochs", 60), options.get("fusion_patience", 10)
    rng = random.Random(seed)
    for epoch in range(epochs):
        rng.shuffle(train_items)
        losses = []
        for item in train_items:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(item)
            if loss is None:
                continue
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            values = [loss_for(item) for item in validation_items]
            value = float(np.mean([float(v) for v in values if v is not None]))
        history.append(
            dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), validation_loss=value)
        )
        if value < best:
            best, stale = value, 0
            saved = [p.detach().clone() for p in parameters]
        else:
            stale += 1
        if patience and stale >= patience:
            break
    with torch.no_grad():
        for parameter, value in zip(parameters, saved):
            parameter.copy_(value)
    model.fusion_ready.fill_(True)
    weights = dict(zip(model.branch_names, F.softplus(model.ai_weight).tolist()))
    print(f"Stage F (hợp điểm): {len(train_items)} train, trọng số AI {weights}", flush=True)
    return dict(
        history=history,
        best_validation_loss=best,
        ai_weights=weights,
        mismatch_weights=dict(
            zip(model.mismatch_names, F.softplus(model.mismatch_weight).tolist())
        ),
        train_samples=len(train_items),
        validation_samples=len(validation_items),
        elapsed_s=time.perf_counter() - began,
    )


def train_stages(model, train, validation, loader, cfg, seed):
    """Học lần lượt từng nhánh rồi hợp điểm; trả báo cáo từng stage."""
    options = cfg["training"]
    reports = {}
    names = model.branch_names
    if "sync" in names:
        fit_statistics(model, [r for r in train if has_primary(r)], loader, seed)
        reports["sync"] = fit_sync(
            model, train, validation, loader, cfg.get("sync", {}), seed, options["top_fraction"]
        )
    if "lead" in names:
        reports["lead"] = fit_lead(model, train, loader, cfg.get("lead", {}))
    if "artifact" in names:
        reports["artifact"] = fit_artifact(model, train, validation, loader, options, seed)
    if model.config["architecture"] == "tricue":
        reports["fusion"] = fit_fusion(model, train, validation, loader, options, seed)
    return reports
