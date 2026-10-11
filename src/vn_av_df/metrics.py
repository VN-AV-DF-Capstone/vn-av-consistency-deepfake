"""Validation operating points and ranked, one-to-one temporal AP."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def binary_metrics(labels, scores, threshold):
    """Metric video tại ngưỡng validation; PR-AUC dùng định nghĩa average precision."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    if not len(labels):
        return {
            "samples": 0,
            "roc_auc": None,
            "pr_auc": None,
            "f1": None,
            "precision": None,
            "recall": None,
            "false_alarm_rate": None,
            "confusion_matrix": [[0, 0], [0, 0]],
        }
    predicted = scores >= threshold
    p, r, f, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", zero_division=0
    )
    neg = labels == 0
    return {
        "samples": len(labels),
        "roc_auc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
        "pr_auc": float(average_precision_score(labels, scores)) if (labels == 1).any() else None,
        "f1": float(f),
        "precision": float(p),
        "recall": float(r),
        "false_alarm_rate": float(predicted[neg].mean()) if neg.any() else None,
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
    }


def choose_threshold(labels, scores):
    """Max F1, hòa ưu tiên FPR thấp rồi ngưỡng cao; đếm TP/FP mọi ngưỡng bằng một lần sắp xếp."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Validation needs both real and fake labels")
    candidates = np.unique(np.r_[0.5, scores, np.nextafter(scores, np.inf)])
    if len(candidates) > 1001:
        candidates = np.unique(np.r_[0.5, np.quantile(candidates, np.linspace(0, 1, 1001))])
    candidates = candidates[(candidates >= 0) & (candidates <= 1)]
    positives, negatives = np.sort(scores[labels == 1]), np.sort(scores[labels == 0])
    # Số điểm >= t: cùng quy ước predicted = scores >= threshold của binary_metrics.
    tp = len(positives) - np.searchsorted(positives, candidates, side="left")
    fp = len(negatives) - np.searchsorted(negatives, candidates, side="left")
    f1 = np.where(tp > 0, 2 * tp / np.maximum(1, tp + len(positives) + fp), 0.0)
    best = np.lexsort((candidates, -fp / len(negatives), f1))[-1]
    return float(candidates[best])


def intervals(scores, valid, times, step, duration, threshold, min_duration=0):
    active = np.asarray(valid) & (np.asarray(scores) >= threshold)
    edges = np.diff(np.r_[False, active, False].astype(int))
    out = []
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        start, end = float(times[a]), min(float(times[b - 1] + step), duration)
        if end - start + 1e-6 >= min_duration:
            out.append({"start_sec": start, "end_sec": end, "score": float(np.mean(scores[a:b]))})
    return out


def iou(a, b):
    overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return overlap / max(1e-9, a[1] - a[0] + b[1] - b[0] - overlap)


def temporal_ap(predictions, truth, threshold):
    """All proposals ranked globally, one match per GT, interpolated precision envelope."""
    total = sum(len(spans) for spans in truth.values())
    if total == 0:
        return {
            "ap": None,
            "matched": 0,
            "mean_tiou": None,
            "start_mae_s": None,
            "end_mae_s": None,
            "short_recall": None,
        }
    used, tp, fp, errors = set(), [], [], []
    short_total = sum(b - a <= 1 for spans in truth.values() for a, b in spans)
    short_hit = 0
    ranked = sorted(
        ((p["score"], sid, p) for sid, spans in predictions.items() for p in spans if sid in truth),
        key=lambda x: (-x[0], x[1]),
    )
    for _, sid, p in ranked:
        span = [p["start_sec"], p["end_sec"]]
        choices = [(iou(span, gt), i) for i, gt in enumerate(truth[sid]) if (sid, i) not in used]
        overlap, index = max(choices, default=(0, -1))
        match = index >= 0 and overlap >= threshold
        tp.append(int(match))
        fp.append(int(not match))
        if match:
            used.add((sid, index))
            gt = truth[sid][index]
            errors.append((overlap, abs(span[0] - gt[0]), abs(span[1] - gt[1])))
            short_hit += gt[1] - gt[0] <= 1
    recalls = np.r_[0, np.cumsum(tp) / total, 1]
    precisions = np.r_[0, np.cumsum(tp) / np.maximum(1, np.cumsum(tp) + np.cumsum(fp)), 0]
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]
    ap = float(np.sum(np.diff(recalls) * precisions[1:]))
    means = np.mean(errors, axis=0).tolist() if errors else [None] * 3
    return {
        "ap": ap,
        "matched": len(errors),
        "mean_tiou": means[0],
        "start_mae_s": means[1],
        "end_mae_s": means[2],
        "short_recall": short_hit / short_total if short_total else None,
    }


def localization_check(predictions):
    """E7: với fake partial, điểm theo ô có tăng đúng trong đoạn bị vẽ không (theo generator).

    Chỉ so ô trong và ngoài đoạn của cùng clip, nên đặc tính chung cả clip không giúp được.
    peak_inside_rate: tỷ lệ clip có ô điểm cao nhất nằm trong đoạn bị vẽ.
    """
    result = {}
    groups = {}
    for p in predictions:
        row = p["row"]
        if row["label"] != 1 or row.get("variant") != "partial":
            continue
        groups.setdefault(row.get("generator", "unknown"), []).append(p)
    for generator, items in sorted(groups.items()):
        labels, scores, gaps, peaks = [], [], [], []
        for p in items:
            observed = p["valid"] & (p["target"] >= 0)
            inside, outside = observed & (p["target"] >= 0.5), observed & (p["target"] == 0)
            if not inside.any() or not outside.any():
                continue
            values = np.asarray(p["scores"])
            labels += [1] * int(inside.sum()) + [0] * int(outside.sum())
            scores += values[inside].tolist() + values[outside].tolist()
            gaps.append(float(values[inside].mean() - values[outside].mean()))
            peaks.append(bool(inside[np.flatnonzero(observed)[np.argmax(values[observed])]]))
        result[generator] = {
            "clips": len(gaps),
            "cell_auc_inside_vs_outside": float(roc_auc_score(labels, scores))
            if len(set(labels)) == 2
            else None,
            "mean_inside_minus_outside": float(np.mean(gaps)) if gaps else None,
            "peak_inside_rate": float(np.mean(peaks)) if peaks else None,
        }
    return result


def diagnostic_summary(predictions, threshold, real_false_alarm=None):
    """E6: mẫu chẩn đoán nhãn 0 (xử lý không AI) có bị báo AI không, theo từng loại."""
    groups = {}
    for p in predictions:
        if p["video_score"] is not None:
            groups.setdefault(p["row"]["diagnostic"], []).append(p["video_score"])
    return {
        name: {
            "samples": len(values),
            "false_alarm_rate": float(np.mean(np.asarray(values) >= threshold)),
            "mean_video_score": float(np.mean(values)),
            "real_false_alarm_rate": real_false_alarm,
        }
        for name, values in sorted(groups.items())
    }


def grouped_auc_interval(predictions, replicates=1000, seed=42):
    """Bootstrap component nguồn/người, không bootstrap từng variant hoặc frame."""
    from vn_av_df.data.groups import connected_groups

    measured = [p for p in predictions if p["video_score"] is not None]
    groups = connected_groups([p["row"] for p in measured]) if measured else []
    result = {
        "groups": len(groups),
        "replicates": replicates,
        "seed": seed,
        "roc_auc_95ci": None,
        "note": "Insufficient independent groups",
    }
    if len(groups) < 5:
        return result
    lookup = {p["row"]["sample_id"]: p["video_score"] for p in measured}
    rng, scores = np.random.default_rng(seed), []
    for _ in range(replicates):
        rows = [r for i in rng.integers(0, len(groups), len(groups)) for r in groups[i]]
        labels = [r["label"] for r in rows]
        if len(set(labels)) == 2:
            scores.append(roc_auc_score(labels, [lookup[r["sample_id"]] for r in rows]))
    result.update(
        roc_auc_95ci=np.quantile(scores, [0.025, 0.975]).tolist() if scores else None,
        valid_replicates=len(scores),
        note="Cluster bootstrap; interpret with number of groups",
    )
    return result


def paired_auc_interval(first, second, replicates=1000, seed=42):
    """Delta AUC first−second trên cùng clip quan sát được, resample cùng component."""
    from vn_av_df.data.groups import connected_groups

    right = {p["row"]["sample_id"]: p for p in second if p["video_score"] is not None}
    common = [p for p in first if p["video_score"] is not None and p["row"]["sample_id"] in right]
    groups = connected_groups([p["row"] for p in common]) if common else []
    result = dict(
        samples=len(common),
        groups=len(groups),
        seed=seed,
        replicates=replicates,
        delta_auc=None,
        delta_auc_95ci=None,
    )
    left = {p["row"]["sample_id"]: p for p in common}

    def delta(rows):
        labels = [r["label"] for r in rows]
        if len(set(labels)) != 2:
            return None
        return float(
            roc_auc_score(labels, [left[r["sample_id"]]["video_score"] for r in rows])
            - roc_auc_score(labels, [right[r["sample_id"]]["video_score"] for r in rows])
        )

    result["delta_auc"] = delta([p["row"] for p in common])
    if len(groups) >= 5:
        rng = np.random.default_rng(seed)
        values = [
            delta([r for i in rng.integers(len(groups), size=len(groups)) for r in groups[i]])
            for _ in range(replicates)
        ]
        values = [v for v in values if v is not None]
        result["delta_auc_95ci"] = np.quantile(values, [0.025, 0.975]).tolist() if values else None
        result["valid_replicates"] = len(values)
    return result
