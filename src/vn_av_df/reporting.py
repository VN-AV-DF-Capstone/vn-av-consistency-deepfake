"""Báo cáo từng detector từ JSON sau train; không cần GPU, checkpoint hoặc video gốc."""

from pathlib import Path

import numpy as np

from vn_av_df.common.runtime import read_json, write_json


def detector_report(folder, split="validation"):
    """Vẽ dữ liệu thực đã lưu; validation là best.pt, loss curve là toàn bộ epoch.

    split='test' chỉ đọc artifact test đã tồn tại, không tự gọi evaluate hoặc chọn ngưỡng.
    Mọi ảnh thuộc một detector/seed; không xếp hạng hoặc vẽ so sánh giữa các detector.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    if split not in {"validation", "test"}:
        raise ValueError("Report split must be validation or test")
    folder = Path(folder)
    config = read_json(folder / "configuration.json")
    history = read_json(folder / "history.json")
    best = read_json(folder / "best.json")
    metrics = read_json(folder / f"{split}.json")
    predictions = read_json(folder / f"predictions-{split}.json")
    resources = read_json(folder / "resources.json")
    # TriCue / nhánh dấu vết vẽ: từng stage học riêng (stages.json); history chỉ một dòng đánh giá.
    stages = read_json(folder / "stages.json") if (folder / "stages.json").exists() else None
    curves = {k: v["history"] for k, v in (stages or {}).items() if v.get("history")}
    out = folder / "report"
    out.mkdir(parents=True, exist_ok=True)
    measured = [p for p in predictions if p["video_score"] is not None]
    labels = np.array([p["label"] for p in measured])
    scores = np.array([p["video_score"] for p in measured])
    chosen = "staged" if stages else f"best epoch {best['epoch']}"
    title = f"{config['architecture']} | seed {config['seed']} | {split} | {chosen}"
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle(title)
    epochs = [h["epoch"] for h in history]
    ax = axes[0, 0]
    if stages:
        for i, (name, rows) in enumerate(curves.items()):
            stage_epochs = [r["epoch"] for r in rows]
            for key, style in (("train_loss", "--"), ("validation_loss", "-")):
                values = [r.get(key) if r.get(key) is not None else np.nan for r in rows]
                ax.plot(
                    stage_epochs,
                    values,
                    color=f"C{i}",
                    linestyle=style,
                    marker="o",
                    markersize=3,
                    label=name if style == "-" else None,
                )
        ax.set(title="Loss per stage (dashed = train)", xlabel="Stage epoch", ylabel="Loss")
    else:
        ax.plot(epochs, [h["train_loss"] for h in history], marker="o", markersize=3, label="Train")
        ax.plot(
            epochs,
            [h["validation_loss"] for h in history],
            marker="o",
            markersize=3,
            label="Validation",
        )
        ax.axvline(best["epoch"], color="gray", linestyle="--", label="Best checkpoint")
        ax.set(title="Loss per epoch", xlabel="Epoch", ylabel="Loss")
    ax.legend()
    ax = axes[0, 1]
    for key, name in (("validation_roc_auc", "ROC-AUC"), ("validation_f1", "F1")):
        values = [h.get(key) if h.get(key) is not None else np.nan for h in history]
        ax.plot(epochs, values, marker="o", markersize=3, label=name)
    ax.set(title="Validation during training", xlabel="Epoch", ylim=(0, 1.05))
    ax.legend()
    ax = axes[0, 2]
    if set(labels.tolist()) == {0, 1}:
        fpr, tpr, _ = roc_curve(labels, scores)
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], "--", color="gray")
    else:
        ax.text(0.5, 0.5, "ROC unavailable: need both labels", ha="center", transform=ax.transAxes)
    ax.set(title=f"{split}: ROC", xlabel="False positive rate", ylabel="True positive rate")
    ax = axes[1, 0]
    if set(labels.tolist()) == {0, 1}:
        precision, recall, _ = precision_recall_curve(labels, scores)
        ax.plot(recall, precision)
    else:
        ax.text(0.5, 0.5, "PR unavailable: need both labels", ha="center", transform=ax.transAxes)
    ax.set(title=f"{split}: precision-recall", xlabel="Recall", ylabel="Precision", ylim=(0, 1.05))
    ax = axes[1, 1]
    matrix = np.asarray(metrics["video"]["confusion_matrix"])
    ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color)
    ax.set(
        title="Confusion matrix (validation threshold)",
        xlabel="Predicted",
        ylabel="Actual",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
    )
    ax = axes[1, 2]
    for label, name in ((0, "Real"), (1, "Fake")):
        ax.hist(scores[labels == label], bins=np.linspace(0, 1, 21), alpha=0.55, label=name)
    ax.axvline(
        best["thresholds"]["video"], color="gray", linestyle="--", label="Validation threshold"
    )
    ax.set(title=f"{split}: video scores", xlabel="Suspicion score", ylabel="Clips")
    ax.legend()
    diagnostics = out / f"{split}-diagnostics.png"
    fig.savefig(diagnostics, dpi=140)
    plt.close(fig)

    # Một real và một fake đầu tiên theo ID: không chọn ví dụ chỉ vì dự đoán đẹp.
    examples = []
    for label in (0, 1):
        candidates = sorted(
            (p for p in predictions if p["label"] == label), key=lambda p: p["sample_id"]
        )
        if candidates:
            examples.append(candidates[0])
    images = [str(diagnostics)]
    reconstruction = None
    if (folder / "reconstruction.json").exists():
        stage = read_json(folder / "reconstruction.json")
        reconstruction = {
            k: stage[k] for k in ("best_validation_loss", "constant_validation_loss", "elapsed_s")
        }
        stage_epochs = [r["epoch"] for r in stage["history"]]
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        ax.plot(
            stage_epochs,
            [r["train_loss"] for r in stage["history"]],
            marker="o",
            markersize=3,
            label="Real train",
        )
        ax.plot(
            stage_epochs,
            [r["validation_loss"] for r in stage["history"]],
            marker="o",
            markersize=3,
            label="Real validation",
        )
        ax.axhline(
            stage["constant_validation_loss"],
            color="gray",
            linestyle="--",
            label="Constant predictor (validation)",
        )
        ax.set(
            title="Stage A: real-only A to V reconstruction",
            xlabel="Epoch",
            ylabel="Smooth L1 loss",
        )
        ax.legend()
        stage_image = out / "reconstruction.png"
        fig.savefig(stage_image, dpi=140)
        plt.close(fig)
        images.append(str(stage_image))
    sync = None
    if (folder / "sync.json").exists():
        stage = read_json(folder / "sync.json")
        sync = {
            k: stage[k]
            for k in ("best_validation_loss", "best_validation_auc", "sham_train", "elapsed_s")
        }
        stage_epochs = [r["epoch"] for r in stage["history"]]
        fig, (loss_ax, auc_ax) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for key, name in (("train_loss", "Train"), ("validation_loss", "Validation")):
            loss_ax.plot(stage_epochs, [r[key] for r in stage["history"]], marker="o", label=name)
        loss_ax.set(title="Stage S: sync head (never sees fake)", xlabel="Epoch", ylabel="BCE")
        loss_ax.legend()
        auc_ax.plot(
            stage_epochs,
            [
                r["validation_auc"] if r["validation_auc"] is not None else np.nan
                for r in stage["history"]
            ],
            marker="o",
        )
        auc_ax.set(
            title="Validation AUC: real vs sham/shifted real",
            xlabel="Epoch",
            ylim=(0, 1.05),
        )
        stage_image = out / "sync.png"
        fig.savefig(stage_image, dpi=140)
        plt.close(fig)
        images.append(str(stage_image))
    elif stages and "sync" in stages:
        # TriCue: stage S nằm trong stages.json (đường loss đã vẽ ở ô đầu).
        sync = {
            k: stages["sync"].get(k)
            for k in ("best_validation_loss", "best_validation_auc", "sham_train", "elapsed_s")
        }
    if examples:
        fig, axes = plt.subplots(
            len(examples),
            1,
            figsize=(12, 3 * len(examples)),
            squeeze=False,
            constrained_layout=True,
        )
        for ax, sample in zip(axes[:, 0], examples):
            times = np.array(sample["times_s"])
            values = np.array(sample["scores"])
            values[~np.asarray(sample["valid"], dtype=bool)] = np.nan
            if len(times):
                # Ô score kéo dài tới ô kế tiếp/cuối clip, không nội suy qua gap.
                ax.step(
                    np.r_[times, sample["duration_s"]],
                    np.r_[values, values[-1]],
                    where="post",
                    label="Score (gaps = unobserved)",
                )
            for i, (start, end) in enumerate(sample["fake_intervals"] or []):
                ax.axvspan(
                    start,
                    end,
                    color="red",
                    alpha=0.18,
                    label="Ground truth fake" if i == 0 else None,
                )
            threshold = best["thresholds"]["temporal"]
            if threshold is not None:
                ax.axhline(threshold, linestyle="--", color="gray", label="Validation threshold")
            annotation = " | interval labels unknown" if sample["fake_intervals"] is None else ""
            ax.set(
                title=f"{split}: {sample['sample_id']} | label={sample['label']}{annotation}",
                xlabel="Time (s)",
                ylabel="Suspicion score",
                ylim=(0, 1.05),
                xlim=(0, sample["duration_s"]),
            )
            ax.legend(loc="upper right")
        timeline = out / f"{split}-timelines.png"
        fig.savefig(timeline, dpi=140)
        plt.close(fig)
        images.append(str(timeline))
    summary = {
        "architecture": config["architecture"],
        "seed": config["seed"],
        "split": split,
        "best_epoch": best["epoch"],
        "epochs_completed": len(history),
        "stage_epochs": {k: len(v) for k, v in curves.items()} if stages else None,
        "video_metrics": metrics["video"],
        "video_coverage": metrics["video_coverage"],
        "window_coverage": metrics["window_coverage"],
        "thresholds": best["thresholds"],
        "resources": resources,
        "reconstruction": reconstruction,
        "sync": sync,
        # Ô 2×2: fake gộp với real gốc; real/sham riêng để đọc false alarm.
        "by_condition": {
            cell: value["video"] for cell, value in (metrics.get("by_condition") or {}).items()
        },
        "branches": metrics.get("branches"),
        "note": "Validation is used for checkpoint/threshold selection; it is not held-out test performance.",
    }
    write_json(out / f"{split}-summary.json", summary)
    return {"summary": summary, "images": images}


def training_reports(folder, split="validation"):
    """Đọc danh mục model đã train xong; vẫn báo cáo được phần hoàn tất của phiên bị ngắt."""
    folder = Path(folder)
    index = read_json(folder / "training.json")
    return [
        detector_report(folder / f"{r['architecture']}_seed{r['seed']}", split)
        for r in index["runs"]
    ]
