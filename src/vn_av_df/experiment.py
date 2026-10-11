"""Shared protocol: cache once, train architectures, calibrate on validation, test once."""

import copy
import io
import math
import random
import time
from collections import Counter, defaultdict
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from vn_av_df import tracking
from vn_av_df.common.runtime import atomic_bytes, fingerprint, read_json, sha, write_json
from vn_av_df.dataset import (
    condition,
    media_path,
    protocol_rows,
    temporal_targets,
    training_dataset,
)
from vn_av_df.features.registry import (
    assemble,
    combined_signature,
    make_encoder,
    method_config,
)
from vn_av_df.metrics import (
    binary_metrics,
    choose_threshold,
    diagnostic_summary,
    intervals,
    localization_check,
    temporal_ap,
)
from vn_av_df.models import build_model, output_valid, pool_score

FORMAT = "vn-av-df-temporal-v2"
STAGED = ("tricue", "artifact")  # Học từng nhánh rồi hợp điểm (tricue.train_stages).


def prediction_records(predictions):
    """Lưu đủ nhãn/timeline để vẽ báo cáo hoặc so sánh offline, không cần video."""
    return [
        dict(
            sample_id=p["row"]["sample_id"],
            label=p["row"]["label"],
            split=p["row"]["split"],
            condition=condition(p["row"]),
            dataset_part=p["row"].get("dataset_part"),
            source_id=p["row"]["source_id"],
            # Clip gốc/speaker cho bootstrap theo nhóm ở compare.ipynb.
            source_clip_id=p["row"].get("source_clip_id"),
            speaker_id=p["row"].get("speaker_id"),
            generator=p["row"].get("generator"),
            variant=p["row"].get("variant"),
            diagnostic=p["row"].get("diagnostic"),
            duration_s=p["row"]["duration_s"],
            fake_intervals=p["row"]["fake_intervals"],
            av_mismatch_intervals=p["row"].get("av_mismatch_intervals"),
            video_score=p["video_score"],
            times_s=p["times"].tolist(),
            scores=p["scores"].tolist(),
            valid=p["valid"].tolist(),
            branch_scores={name: value.tolist() for name, value in p["branches"].items()},
        )
        for p in predictions
    ]


def save_checkpoint(path, state):
    stream = io.BytesIO()
    torch.save(state, stream)
    atomic_bytes(path, stream.getvalue())


ENCODER_CACHES = ("avhubert", "dinov2")  # Thứ tự trích: DINOv2 cần hộp miệng AV-HuBERT.


def attach_caches(sources, target, writable=()):
    """Gom cache encoder từ nhiều thư mục (Kaggle input chỉ đọc) vào `target/<encoder>`.

    Encoder trong `writable` được chép để trích tiếp; encoder khác chỉ tạo symlink (chỉ đọc).
    Trả về {encoder: thư mục nguồn}.
    """
    import shutil

    found = {}
    for source in map(Path, sources):
        for key in ENCODER_CACHES:
            folder = source / key
            if folder.is_dir():
                if key in found:
                    raise ValueError(f"Cache {key} có ở cả {found[key]} và {folder}")
                found[key] = folder
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for key, folder in found.items():
        destination = target / key
        if destination.is_symlink() or destination.exists():
            continue  # Chạy lại cell: giữ bản đã gom/đang trích.
        if key in writable:
            shutil.copytree(folder, destination)
        else:
            destination.symlink_to(folder.resolve(), target_is_directory=True)
    return found


def needs_primary(row):
    """MechFake ngoài tập con hợp điểm chỉ trích DINOv2 (hộp miệng lấy từ clip gốc)."""
    return row.get("feature_scope", "full") != "artifact"


def prepare(cfg):
    """Trích mỗi backbone một lần; test chỉ được extract, không fit thống kê.

    cfg["prepare_encoders"] (vd. ["dinov2"]) giới hạn encoder được trích; None = mọi encoder
    mà các kiến trúc đã chọn cần.
    """
    if cfg.get("methods"):
        reports, seen = {}, set()
        only = cfg.get("prepare_encoders")
        for name in cfg["architectures"]:
            spec = cfg["methods"][name]
            scoped = method_config(cfg, name)
            artifact = {
                "encoder": scoped.pop("artifact_encoder", None),
                "cache": scoped.pop("artifact_cache", None),
            }
            scoped.pop("methods", None)
            # Artifact đọc hộp miệng từ cache encoder chính nên luôn chạy sau nó.
            stages = [(spec["encoder"], scoped)]
            if artifact["encoder"]:
                stages.append((spec["artifact_encoder"], {**scoped, **artifact}))
            for key, stage in stages:
                if key not in seen and (only is None or key in only):
                    reports[key] = prepare(stage)
                    seen.add(key)
        return reports
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor

    rows, selection = training_dataset(cfg, verify_media=True)
    if cfg["encoder"].get("kind") == "avhubert":
        rows = [r for r in rows if needs_primary(r)]
    # Mỗi device (vd. cuda:0, cuda:1) workers_per_device encoder, mẫu chia động giữa các encoder.
    # Device không nằm trong feature signature nên cache giống nhau dù trích ở GPU nào.
    # workers_per_device > 1 khi phần chậm chạy CPU một luồng (dlib của AV-HuBERT).
    copies = int(cfg["encoder"].get("workers_per_device", 1))
    devices = [d for d in (cfg.get("prepare_devices") or [None]) for _ in range(copies)]
    encoders = queue.Queue()
    for device in devices:
        encoders.put(
            make_encoder(cfg["encoder"] if device is None else {**cfg["encoder"], "device": device})
        )
    name = Path(cfg["cache"]).name
    print(
        f"Prepare {name}: {len(rows)} mẫu, cache {cfg['cache']}, device {devices}",
        flush=True,
    )
    start = time.perf_counter()
    guard, progress = threading.Lock(), {"done": 0}

    def extract(group):
        # Cả nhóm cùng parent qua một encoder: real/sham/partial có frame giống nhau, encoder
        # dùng lại được kết quả theo frame (landmark AV-HuBERT).
        encoder = encoders.get()
        try:
            for row in group:
                tick = time.perf_counter()
                request = {"video": str(media_path(cfg["dataset"], row))}
                if row.get("box_sample_id"):
                    request["box_sample_id"] = row["box_sample_id"]
                encoder.extract(request, Path(cfg["cache"]) / (row["sample_id"] + ".npz"))
                with guard:
                    progress["done"] += 1
                    done, spent = progress["done"], time.perf_counter() - start
                # Mẫu đã có cache mất ~0 s; thời gian còn lại ước theo tốc độ trung bình đến giờ.
                print(
                    f"Features {name} {done}/{len(rows)} {row['sample_id']}: "
                    f"{time.perf_counter() - tick:.1f}s | đã chạy {spent / 60:.1f} phút, "
                    f"còn ~{spent / done * (len(rows) - done) / 60:.1f} phút",
                    flush=True,
                )
        finally:
            encoders.put(encoder)

    groups = defaultdict(list)
    for row in rows:
        groups[row.get("source_clip_id") or row["sample_id"]].append(row)
    try:
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = [pool.submit(extract, group) for group in groups.values()]
            try:
                for future in futures:
                    future.result()
            except BaseException:
                # Một mẫu lỗi: không nhận nhóm mới, chờ nhóm đang chạy xong rồi báo lỗi.
                for future in futures:
                    future.cancel()
                raise
    finally:
        signature = encoders.queue[0].signature
        while not encoders.empty():  # Đóng worker riêng (AV-HuBERT) nếu encoder có.
            getattr(encoders.get(), "close", lambda: None)()
    elapsed = time.perf_counter() - start
    write_json(
        Path(cfg["cache"]) / "extraction.json",
        {
            "samples": len(rows),
            "dataset_selection": selection,
            "elapsed_s": elapsed,
            "device": cfg["encoder"].get("device"),
            "devices": devices,
            "feature_signature": signature,
            "note": "Includes loading, decoding and reusable-cache checks",
        },
    )
    return {"samples": len(rows), "elapsed_s": elapsed}


def checked_meta(path, expected):
    """Meta cache của đúng media; lỗi rõ nếu thiếu hoặc lệch."""
    if not path.with_suffix(".json").is_file():
        raise FileNotFoundError(
            f"Thiếu feature cache {path.with_suffix('.json')}: bước prepare chưa chạy xong "
            "(lỗi/bị dừng giữa chừng hoặc khác dữ liệu/encoder). Chạy lại prepare đến hết rồi train."
        )
    meta = read_json(path.with_suffix(".json"))
    if meta["feature_sha256"] != sha(path) or meta["source_fingerprint"] != expected:
        raise ValueError("Stale/corrupt feature cache or media provenance")
    return meta


def cached(row, cfg, device, need_primary=True):
    """Kiểm hash rồi ghép feature thành input model và lưới ô output.

    need_primary=False: cho phép mẫu chỉ có cache artifact (MechFake ngoài tập con hợp điểm).
    """
    expected = fingerprint({"assets": {"video": row["sha256"]}, "variant": {"kind": "clean"}})
    path = Path(cfg["cache"]) / (row["sample_id"] + ".npz")
    primary = None
    if needs_primary(row) or need_primary:
        primary = (path, checked_meta(path, expected))
    artifact = None
    if cfg.get("artifact_cache"):
        extra = Path(cfg["artifact_cache"]) / (row["sample_id"] + ".npz")
        extra_meta = checked_meta(extra, expected)
        # Hộp miệng phải đúng cache AV-HuBERT đã dùng lúc trích (của mẫu hoặc clip gốc).
        box_sid = extra_meta.get("box_sample_id") or row["sample_id"]
        box_meta = read_json(Path(cfg["cache"]) / (box_sid + ".json"))
        if extra_meta["box_sha256"] != box_meta["feature_sha256"]:
            raise ValueError("Artifact cache was cut with other mouth boxes")
        artifact = (extra, extra_meta)
    elif primary is None:
        raise ValueError("Model without artifact encoder needs the primary cache")
    tensors, times, meta = assemble(cfg.get("model_architecture", "tcn"), primary, artifact)
    if (
        not len(times)
        or not np.allclose(np.diff(times), meta["step_s"])
        or times[0] != 0
        or abs(meta["duration_s"] - row["duration_s"]) > 0.1
    ):
        raise ValueError("Cache timeline differs from labeled video")
    primary_signature = primary[1]["feature_signature"] if primary else None
    artifact_signature = artifact[1]["feature_signature"] if artifact else None
    meta = {
        **meta,
        "feature_sha256": primary[1]["feature_sha256"] if primary else None,
        "artifact_sha256": artifact[1]["feature_sha256"] if artifact else None,
        "primary_signature": primary_signature,
        "artifact_signature": artifact_signature,
        "feature_signature": combined_signature(primary_signature, artifact_signature)
        if primary
        else None,
    }
    tensors = [None if x is None else torch.as_tensor(x, device=device) for x in tensors]
    target = torch.as_tensor(temporal_targets(row, times, meta["step_s"]), device=device)
    return tensors, target, times, meta


def objective(logits, valid, target, row, top_fraction):
    observed = valid & (target >= 0)
    loss = (
        F.binary_cross_entropy_with_logits(logits[observed], target[observed])
        if observed.any()
        else None
    )
    # Strong labels: no clip-positive loss if its manipulated region is entirely unobservable.
    video_observed = bool(valid.any()) and (
        row["label"] == 0 or row["fake_intervals"] is None or bool((valid & (target > 0)).any())
    )
    if video_observed:
        score = pool_score(logits.sigmoid(), valid, top_fraction)
        video_loss = F.binary_cross_entropy(
            score.clamp(1e-6, 1 - 1e-6), score.new_tensor(float(row["label"]))
        )
        loss = video_loss if loss is None else loss + video_loss
    return loss


def training_loss(model, logits, valid, target, row, options):
    """Loss AI; P2 cộng head phụ artifact theo ô để timeline artifact có nghĩa riêng."""
    loss = objective(logits, valid, target, row, options["top_fraction"])
    artifact = getattr(model, "outputs", {}).get("artifact")
    weight = options.get("artifact_aux_weight", 0.5)
    observed = valid & (target >= 0)
    if artifact is not None and weight and observed.any():
        extra = weight * F.binary_cross_entropy_with_logits(artifact[observed], target[observed])
        loss = extra if loss is None else loss + extra
    return loss


def branch_reports(groups, top_fraction):
    """AUC không ngưỡng của từng nhánh/đầu ra phụ: real gốc (0) so với từng ô 2×2 (1)."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    names = sorted({k for items in groups.values() for p in items for k in p["branches"]})
    if not names:
        return None

    def pooled(p, name):
        values = p["branches"][name][p["valid"]]
        if not len(values):
            return None
        return float(np.sort(values)[-max(1, math.ceil(len(values) * top_fraction)) :].mean())

    result = {}
    for name in names:
        result[name] = {}
        for cell, items in sorted(groups.items()):
            if cell == "real":
                continue
            pairs = [(0, pooled(p, name)) for p in groups.get("real", [])]
            pairs += [(1, pooled(p, name)) for p in items]
            kept = [(y, s) for y, s in pairs if s is not None]
            labels, scores = [y for y, _ in kept], [s for _, s in kept]
            both = len(set(labels)) == 2
            result[name][cell] = {
                "samples": len(labels),
                "roc_auc": float(roc_auc_score(labels, scores)) if both else None,
                "average_precision": float(average_precision_score(labels, scores))
                if both
                else None,
            }
    return {
        "top_fraction": top_fraction,
        "scores": result,
        "note": "Positive = condition vs original real; sham measures A/V-mismatch sensitivity",
    }


def breakdown(predictions, thresholds, top_fraction):
    """Theo ô 2×2 (real gốc tách khỏi sham) và theo nhánh bằng chứng nếu model có."""
    groups = defaultdict(list)
    for p in predictions:
        groups[condition(p["row"])].append(p)
    reals = groups.get("real", [])
    by_condition = {
        cell: summarize(items if cell in ("real", "sham") else reals + items, thresholds)
        for cell, items in sorted(groups.items())
    }
    return by_condition, branch_reports(groups, top_fraction)


@torch.no_grad()
def predict_rows(model, rows, cfg):
    model.eval()
    results = []
    losses = []
    device = cfg["device"]
    need = cfg.get("model_architecture") != "artifact"
    for row in rows:
        tensors, target, times, meta = cached(row, cfg, device, need)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        logits = model(*tensors)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        valid = output_valid(model, tensors[2])
        loss = training_loss(model, logits, valid, target, row, cfg["training"])
        if loss is not None:
            losses.append(float(loss))
        score = pool_score(logits.sigmoid(), valid, cfg["training"]["top_fraction"])
        results.append(
            {
                "row": row,
                "scores": logits.sigmoid().cpu().numpy(),
                "valid": valid.cpu().numpy(),
                "target": target.cpu().numpy(),
                "times": times,
                "step": meta["step_s"],
                "video_score": float(score) if score is not None else None,
                "inference_s": elapsed,
                "branches": {
                    name: value.sigmoid().cpu().numpy()
                    for name, value in getattr(model, "outputs", {}).items()
                    if value is not None
                },
            }
        )
    return results, float(np.mean(losses)) if losses else None


def calibrate(results):
    measured = [r for r in results if r["video_score"] is not None]
    video_threshold = choose_threshold(
        [r["row"]["label"] for r in measured], [r["video_score"] for r in measured]
    )
    labels = []
    scores = []
    for r in results:
        # A small edited fraction is positive for localization; no rounding short events away.
        mask = r["valid"] & (r["target"] >= 0)
        labels.extend((r["target"][mask] > 0).astype(int).tolist())
        scores.extend(r["scores"][mask].tolist())
    temporal_threshold = choose_threshold(labels, scores) if set(labels) == {0, 1} else None
    return {"video": video_threshold, "temporal": temporal_threshold}


def summarize(results, thresholds):
    measured = [r for r in results if r["video_score"] is not None]
    video = binary_metrics(
        [r["row"]["label"] for r in measured],
        [r["video_score"] for r in measured],
        thresholds["video"],
    )
    prediction = {}
    truth = {}
    real_alerts = []
    real_events, real_observed_seconds, real_input_seconds = 0, 0.0, 0.0
    known_cells = 0
    valid_cells = 0
    for r in results:
        row = r["row"]
        sid = row["sample_id"]
        spans = (
            intervals(
                r["scores"],
                r["valid"],
                r["times"],
                r["step"],
                row["duration_s"],
                thresholds["temporal"],
            )
            if thresholds["temporal"] is not None
            else []
        )
        prediction[sid] = spans
        if row["fake_intervals"] is not None:
            truth[sid] = row["fake_intervals"]
        if row["label"] == 0:
            real_alerts.append(bool(spans))
            real_events += len(spans)
            real_input_seconds += row["duration_s"]
            widths = np.minimum(r["times"] + r["step"], row["duration_s"]) - r["times"]
            real_observed_seconds += float(widths[r["valid"]].sum())
        known_cells += len(r["valid"])
        valid_cells += int(r["valid"].sum())
    temporal = (
        {str(t): temporal_ap(prediction, truth, t) for t in (0.3, 0.5, 0.75, 0.95)}
        if thresholds["temporal"] is not None
        else None
    )
    return {
        "video": video,
        "temporal": temporal,
        "temporal_labeled_videos": len(truth),
        "video_coverage": len(measured) / max(1, len(results)),
        "window_coverage": valid_cells / max(1, known_cells),
        "real_video_interval_false_alarm_rate": float(np.mean(real_alerts))
        if real_alerts and temporal is not None
        else None,
        "head_inference_seconds": sum(r["inference_s"] for r in results),
        "false_intervals_per_real_hour": real_events * 3600 / real_observed_seconds
        if real_observed_seconds and temporal is not None
        else None,
        "real_observed_seconds": real_observed_seconds,
        "real_input_seconds": real_input_seconds,
        "latency_note": "Temporal detector only; feature extraction cost recorded separately",
    }


def trainable(row, allowed):
    """Train/val: real/sham luôn giữ; fake chỉ của generator method cho phép; chẩn đoán thì không."""
    if row.get("diagnostic"):
        return False
    if row["label"] == 0 or allowed is None:
        return True
    return any(fnmatch(row["generator"], pattern) for pattern in allowed)


def model_options(cfg, head):
    """Tham số dựng model ngoài kích thước feature (ghi vào checkpoint)."""
    if head == "tricue":
        from vn_av_df.tricue import BRANCHES

        return {"branches": list(cfg.get("branches") or BRANCHES), **cfg.get("tricue", {})}
    return {}


def validation_report(model, rows, cfg):
    """Dự đoán validation, khoá ngưỡng, báo cáo theo ô/nhánh."""
    prediction, loss = predict_rows(model, rows, cfg)
    thresholds = calibrate(prediction)
    report = summarize(prediction, thresholds)
    report["by_condition"], report["branches"] = breakdown(
        prediction, thresholds, cfg["training"]["top_fraction"]
    )
    cells = {}
    for cell, metrics in report["by_condition"].items():
        key = "auc" if cell not in ("real", "sham") else "false_alarm"
        cells[f"validation_{key}/{cell}"] = metrics["video"][
            "roc_auc" if key == "auc" else "false_alarm_rate"
        ]
    for branch, values in ((report["branches"] or {}).get("scores") or {}).items():
        for cell, metrics in values.items():
            cells[f"branch_auc/{branch}/{cell}"] = metrics["roc_auc"]
    return prediction, loss, thresholds, report, cells


def train_one(cfg, architecture, seed, resume=False):
    """Một method/seed: các stage học từ real (nếu có), head có giám sát, validation."""
    cfg = method_config(cfg, architecture)
    head = cfg["model_architecture"]
    staged = head in STAGED
    need = head != "artifact"
    torch.set_num_threads(cfg["training"].get("torch_threads", 2))
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if str(cfg["device"]).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    rows, selection = training_dataset(cfg)
    rows = protocol_rows(rows, cfg.get("held_out_generators", []))
    allowed = cfg.get("train_generators")
    train = [r for r in rows if r["split"] == "train" and trainable(r, allowed)]
    val = [r for r in rows if r["split"] == "validation" and trainable(r, allowed)]
    if not staged:
        train = [r for r in train if needs_primary(r)]
        val = [r for r in val if needs_primary(r)]
    for name, subset in (("train", train), ("validation", val)):
        if {r["label"] for r in subset} != {0, 1}:
            raise ValueError(f"Need real and fake in {name}")
    primary_signatures, artifact_signatures = set(), set()
    cache_hashes = []
    dims = {}
    native_stride = 1
    for r in train + val:
        tensors, _, _, meta = cached(r, cfg, "cpu", need and not staged)
        if meta["primary_signature"]:
            primary_signatures.add(meta["feature_signature"])
            dims.setdefault("audio", tensors[0].shape[-1])
            dims.setdefault("visual", tensors[1].shape[-1])
        if meta["artifact_signature"]:
            artifact_signatures.add(meta["artifact_signature"])
            dims.setdefault("artifact", tensors[3].shape[-1])
        cache_hashes.append([meta["feature_sha256"], meta["artifact_sha256"]])
        native_stride = meta.get("native_stride", 1)
    if len(primary_signatures) > 1 or len(artifact_signatures) > 1:
        raise ValueError("Mixed encoder versions in cache")
    signature = next(iter(primary_signatures), None) or next(iter(artifact_signatures))
    options = cfg["training"]
    if options["epochs"] < 1:
        raise ValueError("epochs must be positive")
    if not 0 < options["top_fraction"] <= 1:
        raise ValueError("top_fraction must be in (0,1]")
    model = build_model(
        dims.get("audio", 0),
        dims.get("visual", 0),
        head,
        options["hidden"],
        options["dropout"],
        native_stride=native_stride,
        **({"artifact_dim": dims["artifact"]} if "artifact" in dims else {}),
        **model_options(cfg, head),
    ).to(cfg["device"])
    out = Path(cfg["runs"]) / f"{architecture}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    implementation = (
        "experiment.py",
        "models.py",
        "reconstruction.py",
        "syncartifact.py",
        "tricue.py",
        "dataset.py",
        "metrics.py",
        "features/registry.py",
    )
    run_id = fingerprint(
        {
            "rows": train + val,
            "manifest_sha256": selection["manifest_sha256"],
            "cache": cache_hashes,
            "options": {k: v for k, v in options.items() if k != "epochs"},
            "model": model.config,
            "seed": seed,
            "train_generators": allowed,
            "reconstruction": cfg.get("reconstruction", {}),
            "sync": cfg.get("sync", {}),
            "lead": cfg.get("lead", {}),
            "implementation": {
                name: fingerprint((Path(__file__).parent / name).read_text(encoding="utf-8"))
                for name in implementation
            },
        }
    )
    # Số mẫu theo split/ô 2×2 (test chỉ đếm, không đọc) để biết detector train trên dữ liệu nào.
    data = dict(sorted(Counter(f"{r['split']}/{condition(r)}" for r in rows).items()))
    print(
        f"Train {out.name}: {len(train)} train, {len(val)} validation, fake train "
        f"{sorted({r['generator'] for r in train if r['label']})}",
        flush=True,
    )
    common = {
        "format": FORMAT,
        "model_config": model.config,
        "feature_signature": signature,
        "encoder_config": cfg["encoder"],
        "artifact_encoder_config": cfg.get("artifact_encoder"),
        "method_name": architecture,
        "train_generators": allowed,
        "run_id": run_id,
        "seed": seed,
        "held_out_generators": cfg.get("held_out_generators", []),
        "manifest_sha256": selection["manifest_sha256"],
        "dataset_selection": selection,
    }
    if staged:
        return train_staged(
            cfg, model, train, val, out, run_id, common, data, architecture, seed, resume
        )
    return train_supervised(
        cfg, model, train, val, out, run_id, common, data, architecture, seed, resume
    )


def train_staged(cfg, model, train, val, out, run_id, common, data, architecture, seed, resume):
    """TriCue / nhánh dấu vết vẽ: các stage học riêng, không resume giữa stage (train rẻ)."""
    from vn_av_df.tricue import has_primary, train_stages

    best_path = out / "best.pt"
    if best_path.exists():
        if not resume:
            raise FileExistsError("Run exists: set RESUME=True or choose RUN_NAME")
        state = torch.load(best_path, map_location="cpu", weights_only=True)
        if state["run_id"] != run_id:
            raise ValueError("Resume configuration/data/code changed")
        return best_path
    tracking.start(cfg, out, architecture, seed, data)
    began = time.perf_counter()

    def load(row, need_primary=True):
        return cached(row, cfg, cfg["device"], need_primary)

    write_json(
        out / "configuration.json",
        {"run_id": run_id, "architecture": architecture, "seed": seed, "settings": cfg},
    )
    stages = train_stages(model, train, val, load, cfg, seed)
    write_json(out / "stages.json", stages)
    # Đánh giá validation trên mẫu chạy được cả model (TriCue cần đủ cache AV-HuBERT).
    rows = val if model.config["architecture"] == "artifact" else [r for r in val if has_primary(r)]
    prediction, val_loss, thresholds, report, cells = validation_report(model, rows, cfg)
    history = [
        {
            "epoch": 1,
            "validation_loss": val_loss,
            "validation_roc_auc": report["video"]["roc_auc"],
            "validation_f1": report["video"]["f1"],
            "validation_pr_auc": report["video"]["pr_auc"],
            "validation_false_alarm_rate": report["video"]["false_alarm_rate"],
            "video_threshold": thresholds["video"],
            "epoch_seconds": time.perf_counter() - began,
            **cells,
        }
    ]
    tracking.log("detector", history[0])
    state = {
        **common,
        "state": model.state_dict(),
        "stage_reports": stages,
        "sync_report": stages.get("sync"),
        "thresholds": thresholds,
        "top_fraction": cfg["training"]["top_fraction"],
        "epoch": 0,
        "history": history,
        "validation_report": report,
    }
    save_checkpoint(out / "last.pt", state)
    save_checkpoint(best_path, state)
    write_json(out / "validation.json", report)
    write_json(out / "predictions-validation.json", prediction_records(prediction))
    write_json(
        out / "best.json",
        {"epoch": 1, "validation_loss": val_loss, "thresholds": thresholds, "selection": "stages"},
    )
    write_json(out / "history.json", history)
    finish_run(
        cfg, model, out, began, {"stage_seconds": {k: v["elapsed_s"] for k, v in stages.items()}}
    )
    return best_path


def finish_run(cfg, model, out, began, extra):
    resources = {
        "elapsed_this_session_s": time.perf_counter() - began,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_head_parameters": sum(p.numel() for p in model.parameters()),
        "device": cfg["device"],
        "peak_gpu_bytes": torch.cuda.max_memory_allocated()
        if str(cfg["device"]).startswith("cuda")
        else None,
        **extra,
    }
    write_json(out / "resources.json", resources)
    tracking.finish(
        {
            "best_epoch": read_json(out / "best.json")["epoch"],
            **{f"resources/{k}": v for k, v in resources.items()},
        }
    )


def train_supervised(cfg, model, train, val, out, run_id, common, data, architecture, seed, resume):
    """B-AVH (TCN) và P2: vòng epoch có giám sát, resume theo last.pt."""
    options = cfg["training"]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=options["lr"],
        weight_decay=options["weight_decay"],
    )
    start = 0
    best = float("inf")
    stale = 0
    history = []
    reconstruction_report = sync_report = None
    if (out / "last.pt").exists() and not resume:
        raise FileExistsError("Run exists: set RESUME=True or choose RUN_NAME")
    tracking.start(cfg, out, architecture, seed, data)  # train_selected đóng run kể cả khi lỗi.
    if (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
        if state["run_id"] != run_id:
            raise ValueError("Resume configuration/data/code changed")
        model.load_state_dict(state["state"])
        optimizer.load_state_dict(state["optimizer"])
        start = state["epoch"] + 1
        best = state["best"]
        stale = state["stale"]
        history = state["history"]
        reconstruction_report = state.get("reconstruction_report")
        sync_report = state.get("sync_report")
        # Run W&B mới của phiên resume: ghi lại lịch sử đã có trước khi train tiếp.
        for stage, rows in (
            ("stageA", (reconstruction_report or {}).get("history", [])),
            ("stageS", (sync_report or {}).get("history", [])),
            ("detector", history),
        ):
            for row in rows:
                tracking.log(stage, row, echo=False)
        torch.set_rng_state(state["rng"])
        if str(cfg["device"]).startswith("cuda") and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if options["patience"] and stale >= options["patience"]:
            start = options["epochs"]  # Đã early stop: resume không train thêm, giữ best.pt.
    elif hasattr(model, "reconstructor"):
        from vn_av_df.reconstruction import fit_reconstruction
        from vn_av_df.syncartifact import fit_sync

        def load(row):
            return cached(row, cfg, cfg["device"])

        reconstruction_report = fit_reconstruction(
            model, train, val, lambda row: load(row)[0], cfg.get("reconstruction", options), seed
        )
        write_json(out / "reconstruction.json", reconstruction_report)
        sync_report = fit_sync(
            model, train, val, load, cfg.get("sync", {}), seed, options["top_fraction"]
        )
        write_json(out / "sync.json", sync_report)
    began = time.perf_counter()
    write_json(
        out / "configuration.json",
        {"run_id": run_id, "architecture": architecture, "seed": seed, "settings": cfg},
    )
    for epoch in range(start, options["epochs"]):
        tick = time.perf_counter()
        model.train()
        if options.get("balance_parents", True):
            from vn_av_df.reconstruction import balanced_order

            order = balanced_order(train, seed + epoch)
        else:
            order = train.copy()
            random.Random(seed + epoch).shuffle(order)
        losses = []
        for row in order:
            tensors, target, _, _ = cached(row, cfg, cfg["device"])
            optimizer.zero_grad(set_to_none=True)
            logits = model(*tensors)
            loss = training_loss(
                model, logits, output_valid(model, tensors[2]), target, row, options
            )
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach()))
        if not losses:
            raise ValueError("No usable training windows")
        prediction, val_loss, thresholds, report, cells = validation_report(model, val, cfg)
        if val_loss is None:
            raise ValueError("No usable validation windows")
        improved = val_loss < best
        best, stale = (val_loss, 0) if improved else (best, stale + 1)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "validation_loss": val_loss,
                "validation_roc_auc": report["video"]["roc_auc"],
                "validation_f1": report["video"]["f1"],
                "validation_coverage": report["video_coverage"],
                "validation_pr_auc": report["video"]["pr_auc"],
                "validation_false_alarm_rate": report["video"]["false_alarm_rate"],
                "validation_temporal_ap_0.5": ((report["temporal"] or {}).get("0.5") or {}).get(
                    "ap"
                ),
                "video_threshold": thresholds["video"],
                "best_validation_loss": best,
                "epochs_without_improvement": stale,
                "epoch_seconds": time.perf_counter() - tick,
                **cells,
            }
        )
        tracking.log("detector", history[-1])
        state = {
            **common,
            "state": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "reconstruction_report": reconstruction_report,
            "sync_report": sync_report,
            "thresholds": thresholds,
            "top_fraction": options["top_fraction"],
            "epoch": epoch,
            "best": best,
            "stale": stale,
            "history": history,
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if str(cfg["device"]).startswith("cuda")
            else None,
            "validation_report": report,
        }
        save_checkpoint(out / "last.pt", state)
        if improved:
            save_checkpoint(out / "best.pt", state)
            write_json(out / "validation.json", report)
            write_json(out / "predictions-validation.json", prediction_records(prediction))
            write_json(
                out / "best.json",
                {
                    "epoch": epoch + 1,
                    "validation_loss": val_loss,
                    "thresholds": thresholds,
                    "selection": "minimum validation loss",
                },
            )
        write_json(out / "history.json", history)
        if improved:
            print(f"{out.name}: best mới ở epoch {epoch + 1}, lưu best.pt", flush=True)
        if options["patience"] and stale >= options["patience"]:
            print(f"{out.name}: early stop sau {stale} epoch không cải thiện", flush=True)
            break
    finish_run(
        cfg,
        model,
        out,
        began,
        {
            "reconstruction_seconds": reconstruction_report["elapsed_s"]
            if reconstruction_report
            else 0,
            "sync_seconds": sync_report["elapsed_s"] if sync_report else 0,
        },
    )
    return out / "best.pt"


def load_model(path, device="cpu"):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != FORMAT:
        raise ValueError("Need a v002 temporal checkpoint; older checkpoints are incompatible")
    model = build_model(**state["model_config"]).to(device)
    model.load_state_dict(state["state"])
    model.eval()
    return model, state


def resolve_checkpoint(cfg, recorded):
    path = Path(recorded)
    if path.is_file():
        return path
    # Artifact ZIP preserves <architecture_seed>/best.pt beneath the configured run.
    parts = str(recorded).replace("\\", "/").split("/")
    if len(parts) >= 2 and parts[-1] in {"best.pt", "last.pt"}:
        candidate = Path(cfg["runs"]) / parts[-2] / parts[-1]
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Checkpoint not found after artifact relocation: {recorded}")


def train_selected(cfg, resume=False):
    """Train đúng danh sách model/seed, lưu từng kết quả; không xếp hạng hoặc chạy test."""
    _, selection = training_dataset(cfg)
    if not cfg["architectures"] or not cfg["seeds"]:
        raise ValueError("Configure at least one architecture and seed")
    if len(set(cfg["architectures"])) != len(cfg["architectures"]) or len(set(cfg["seeds"])) != len(
        cfg["seeds"]
    ):
        raise ValueError("Architectures and seeds must not contain duplicates")
    if cfg.get("methods") and set(cfg["architectures"]) - set(cfg["methods"]):
        raise ValueError("Unknown architecture in selection")
    index_path = Path(cfg["runs"]) / "training.json"
    requested = {
        "architectures": cfg["architectures"],
        "seeds": cfg["seeds"],
        "dataset_selection": selection,
    }
    if index_path.exists():
        previous = read_json(index_path)
        if any(previous.get(k) != v for k, v in requested.items()):
            raise ValueError("Training selection changed; use a new RUN_NAME")
    # Ghi sau từng detector để phần hoàn thành vẫn báo cáo được nếu phiên bị ngắt.
    report = {**requested, "runs": [], "status": "training"}
    results = []
    for architecture in cfg["architectures"]:
        for seed in cfg["seeds"]:
            try:
                checkpoint = train_one(cfg, architecture, seed, resume)
            finally:
                tracking.finish()  # Không để run W&B mở khi một detector lỗi.
            _, state = load_model(checkpoint)
            if state["manifest_sha256"] != selection["manifest_sha256"]:
                raise ValueError("Dataset parts changed during training")
            results.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "validation": state["validation_report"],
                    "resources": read_json(checkpoint.parent / "resources.json"),
                }
            )
            report["runs"] = results
            write_json(index_path, report)
    report["status"] = "complete"
    write_json(index_path, report)
    write_json(Path(cfg["runs"]) / "dataset-selection.json", selection)
    return report


def evaluate(cfg):
    """Test với checkpoint/threshold đã khóa; chạy lại cùng protocol không đổi quyết định.

    Mẫu chẩn đoán (E6: xử lý không AI, nhãn 0) báo riêng, không vào chỉ số test chính.
    """
    folder = Path(cfg["runs"])
    index_path = folder / "training.json"
    comparison = read_json(index_path)
    if comparison.get("status") != "complete":
        raise ValueError("Finish the selected training runs before test evaluation")
    all_rows, selection = training_dataset(cfg)
    lock_path = Path(cfg["runs"]) / "test-lock.json"
    lock = {
        "manifest_sha256": selection["manifest_sha256"],
        "run_index_sha256": sha(index_path),
        "checkpoints": [sha(resolve_checkpoint(cfg, r["checkpoint"])) for r in comparison["runs"]],
    }
    if lock_path.exists() and read_json(lock_path) != lock:
        raise ValueError("Test protocol changed after first evaluation; use a new experiment")
    write_json(lock_path, lock)
    tests = [r for r in all_rows if r["split"] == "test"]
    rows = [r for r in tests if not r.get("diagnostic")]
    diagnostics = [r for r in tests if r.get("diagnostic")]
    reports = []
    for run in comparison["runs"]:
        scoped = method_config(cfg, run["architecture"])
        checkpoint = resolve_checkpoint(scoped, run["checkpoint"])
        model, state = load_model(checkpoint, cfg["device"])
        if state["manifest_sha256"] != selection["manifest_sha256"]:
            raise ValueError("Evaluation dataset differs from trained experiment")
        need = scoped["model_architecture"] != "artifact"
        for row in rows:
            meta = cached(row, scoped, "cpu", need)[3]
            if (
                meta["feature_signature"]
                and meta["feature_signature"] != state["feature_signature"]
            ):
                raise ValueError("Test encoder differs")
        eval_cfg = copy.deepcopy(scoped)
        eval_cfg["training"]["top_fraction"] = state["top_fraction"]
        predictions, _ = predict_rows(model, rows, eval_cfg)
        report = summarize(predictions, state["thresholds"])
        report["by_condition"], report["branches"] = breakdown(
            predictions, state["thresholds"], state["top_fraction"]
        )
        report["by_part"] = {
            part: summarize(
                [p for p in predictions if p["row"].get("dataset_part") == part],
                state["thresholds"],
            )
            for part in sorted({r["dataset_part"] for r in rows if "dataset_part" in r})
        }
        from vn_av_df.metrics import grouped_auc_interval

        report["uncertainty"] = grouped_auc_interval(
            predictions, cfg.get("bootstrap_replicates", 1000)
        )
        report["by_variant"] = {
            variant: summarize(
                [
                    r
                    for r in predictions
                    if r["row"]["label"] == 0 or r["row"].get("variant") == variant
                ],
                state["thresholds"],
            )
            for variant in sorted(
                {r["row"].get("variant", "unknown") for r in predictions if r["row"]["label"] == 1}
            )
        }
        write_json(
            checkpoint.parent / "predictions-test.json",
            prediction_records(predictions),
        )
        generators = sorted({r["generator"] for r in rows if r["label"] == 1})
        report["by_generator"] = {
            g: summarize(
                [r for r in predictions if r["row"]["label"] == 0 or r["row"]["generator"] == g],
                state["thresholds"],
            )
            for g in generators
        }
        # E7: điểm tăng đúng trong đoạn bị vẽ (fake partial) hay đều cả clip.
        report["localization_check"] = localization_check(predictions)
        if diagnostics:
            # E6: xử lý không AI (dán lại miệng thật / xử lý toàn khung) có bị báo AI không.
            extra, _ = predict_rows(model, diagnostics, eval_cfg)
            reals = [
                p["video_score"] >= state["thresholds"]["video"]
                for p in predictions
                if condition(p["row"]) == "real" and p["video_score"] is not None
            ]
            report["diagnostics"] = diagnostic_summary(
                extra, state["thresholds"]["video"], float(np.mean(reals)) if reals else None
            )
            write_json(
                checkpoint.parent / "predictions-diagnostics.json", prediction_records(extra)
            )
        report.update(
            checkpoint_sha256=sha(checkpoint),
            held_out_generators=state["held_out_generators"],
            thresholds=state["thresholds"],
        )
        write_json(checkpoint.parent / "test.json", report)
        video = report["video"]
        print(
            f"Test {checkpoint.parent.name}: {video['samples']} video, ROC-AUC "
            f"{video['roc_auc']}, F1 {video['f1']}; theo generator "
            f"{ {g: m['video']['roc_auc'] for g, m in report['by_generator'].items()} }",
            flush=True,
        )
        reports.append({"architecture": run["architecture"], "seed": run["seed"], **report})
    # Chỉ là danh mục báo cáo từng detector; so sánh nhiều kiến trúc ở notebooks/compare.ipynb.
    write_json(folder / "evaluation.json", {"dataset_selection": selection, "runs": reports})
    return reports
