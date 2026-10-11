"""Binary labels and temporal ground truth, independent of generator identity."""

import math
from pathlib import Path, PureWindowsPath

import numpy as np

from vn_av_df.common.runtime import fingerprint, read_json, sha
from vn_av_df.data.groups import connected_groups, read_manifest

SCHEMA = "vn-av-df-v1"
CONTROLS = {"global_lag", "local_lag", "motion_freeze", "sequence_swap", "content_splice"}


def validate_rows(rows, root=None, verify_media=False):
    """Kiểm binary labels, provenance, optional modality targets và leakage transitive."""
    if not rows:
        raise ValueError("Empty dataset")
    ids, hashes = set(), {}
    for row in rows:
        sid = row.get("sample_id", "")
        if (
            not sid
            or any(not (c.isascii() and (c.isalnum() or c in "_-")) for c in sid)
            or sid in ids
        ):
            raise ValueError("Invalid or duplicate sample_id")
        ids.add(sid)
        if row.get("split") not in {"train", "validation", "test"}:
            raise ValueError("Split must be assigned before generation")
        if row.get("label") not in (0, 1) or row.get("review_status") != "keep":
            raise ValueError("Require real=0/fake=1 and reviewed keep")
        duration = row.get("duration_s", 0)
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Invalid duration")
        if not row.get("source_id") or not row.get("generator") or not row.get("generator_version"):
            raise ValueError("Source and generator provenance required")
        if row["generator"] in CONTROLS:
            raise ValueError("Legacy inconsistency controls are not deepfake benchmark labels")
        if "fake_intervals" not in row:
            raise ValueError("fake_intervals must be explicit: intervals, [] for real, or null")
        spans = row["fake_intervals"]
        if "synthetic_audio" in row or "synthetic_visual" in row:
            values = [row.get("synthetic_audio"), row.get("synthetic_visual")]
            if any(x is not None and type(x) is not bool for x in values):
                raise ValueError("Modality synthetic labels must be bool or null")
            if all(x is not None for x in values) and int(any(values)) != row["label"]:
                raise ValueError("Binary AI label differs from modality labels")
            for modality in ("audio", "visual"):
                modal_spans = row.get(modality + "_fake_intervals")
                flag = row.get("synthetic_" + modality)
                if flag is False and modal_spans != []:
                    raise ValueError("Non-synthetic modality requires empty intervals")
                if flag is True and modal_spans == []:
                    raise ValueError("Synthetic modality needs intervals or unknown null")
                previous = 0
                for a, b in modal_spans or []:
                    if (
                        not all(math.isfinite(t) for t in (a, b))
                        or not 0 <= a < b <= duration + 1e-6
                        or a < previous
                    ):
                        raise ValueError("Invalid modality intervals")
                    previous = b
            modal_lists = [row.get(m + "_fake_intervals") for m in ("audio", "visual")]
            if spans is not None and all(x is not None for x in modal_lists):
                merged = []
                for a, b in sorted(modal_lists[0] + modal_lists[1]):
                    if merged and a <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], b)
                    else:
                        merged.append([a, b])
                if len(merged) != len(spans) or not np.allclose(merged, spans, atol=1e-6):
                    raise ValueError("Union AI intervals differs from modality intervals")
        if row["label"] == 0 and spans != []:
            raise ValueError("Real clips require empty fake_intervals")
        if row["label"] == 1 and spans == []:
            raise ValueError("Fake needs positive intervals or null for weak labels")
        check_intervals(spans, duration)
        if "av_mismatch_intervals" in row:
            mismatch = row["av_mismatch_intervals"]
            if row["label"] == 0 and not row.get("control_type") and mismatch != []:
                raise ValueError("Reviewed real clips have no A/V mismatch")
            check_intervals(mismatch, duration)
        digest = row.get("sha256", "")
        if len(digest) != 64:
            raise ValueError("Media SHA256 required")
        if digest in hashes and hashes[digest] != row["split"]:
            raise ValueError("Identical media crosses splits")
        hashes[digest] = row["split"]
        if verify_media:
            path = media_path(root, row)
            if sha(path) != digest:
                raise ValueError(f"Media hash mismatch: {sid}")
    for group in connected_groups(rows):
        if len({r["split"] for r in group}) != 1:
            raise ValueError("Source, donor or speaker leakage across splits")
    return rows


def check_intervals(spans, duration):
    """null = chưa biết; danh sách phải sắp xếp, không chồng và nằm trong media."""
    end = 0.0
    for a, b in spans or []:
        if (
            not all(math.isfinite(x) for x in (a, b))
            or not 0 <= a < b <= duration + 1e-6
            or a < end
        ):
            raise ValueError("Intervals must be sorted, disjoint and inside media")
        end = b


def condition(row):
    """Ô thiết kế 2×2: real, sham, <generator>/<audio_mode> cho fake; diag/<loại> cho E6."""
    if row.get("diagnostic"):
        return f"diag/{row['diagnostic']}"
    if row["label"] == 0:
        return "sham" if row.get("control_type") else "real"
    return f"{row['generator']}/{row.get('audio_mode') or 'unknown'}"


def media_path(root, row):
    root = Path(root).resolve()
    rel = Path(row["video"])
    path = (root / rel).resolve()
    if rel.is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing media or unsafe dataset path: {row['video']}")
    return path


def load_dataset(folder, manifest="manifest.jsonl", verify_media=False):
    """Đọc một bundle đã chốt; không suy luận nhãn từ tên part/thư mục."""
    root = Path(folder)
    info = read_json(root / "dataset_info.json")
    if info.get("schema_version") != SCHEMA:
        raise ValueError(
            "Expected binary vn-av-df-v1 dataset; old relation labels cannot be silently relabeled"
        )
    path = root / manifest
    receipt = info if manifest == "manifest.jsonl" else read_json(path.with_suffix(".info.json"))
    if sha(path) != receipt["manifest_sha256"]:
        raise ValueError("Manifest changed after finalization")
    rows = read_manifest(path)
    if len(rows) != receipt["samples"]:
        raise ValueError("Manifest count mismatch")
    return validate_rows(rows, root, verify_media)


def part_selection(paths):
    """Đường dẫn đầy đủ từng part (copy từ Kaggle) → (gốc chung, part tương đối với gốc).

    Tên part (thư mục cuối, vd. vn-av-df-data-part1) đi vào hash lựa chọn dữ liệu, nên cache và
    checkpoint vẫn khớp dù dataset được mount ở đâu.
    """
    import os

    from vn_av_data.common.runtime import find_part

    folders = [find_part(Path(p), "dataset_info.json").resolve() for p in paths]
    if not folders:
        raise ValueError("Need at least one dataset part")
    root = folders[0].parent if len(folders) == 1 else Path(os.path.commonpath(folders))
    return root, [f.relative_to(root).as_posix() for f in folders]


def selected_parts(cfg):
    """Tìm part local/Kaggle; 'all' chỉ gồm các part đang có trên đĩa."""
    root = Path(cfg["dataset"]).resolve()
    selection = cfg.get("dataset_parts")
    if selection is None:
        return [(None, root)]  # Bundle đơn vẫn được hỗ trợ.
    if selection == "all":
        selection = []
        for child in sorted(root.iterdir()):
            if child.is_dir() and child.name.startswith("vn-av-df-data-part"):
                # ZIP upload lên Kaggle có thể giữ thêm thư mục cùng tên ở bên trong.
                nested = child / child.name
                folder = (
                    nested
                    if not (child / "dataset_info.json").exists() and nested.is_dir()
                    else child
                )
                selection.append(folder.relative_to(root).as_posix())
    if not isinstance(selection, (list, tuple)) or not selection:
        raise ValueError("dataset_parts must be a nonempty list or 'all' with downloaded parts")
    result, seen = [], set()
    for value in selection:
        rel = Path(value)
        folder = (root / rel).resolve()
        if (
            rel.is_absolute()
            or PureWindowsPath(value).drive
            or ".." in rel.parts
            or not folder.is_relative_to(root)
        ):
            raise ValueError(f"Unsafe part path: {value}")
        if folder.name in seen:
            raise ValueError("Duplicate dataset part")
        seen.add(folder.name)
        if not (folder / "dataset_info.json").is_file():
            raise ValueError(f"Part missing or not finalized: {value}")
        result.append((folder.name, folder))
    return sorted(result, key=lambda item: item[0])


def training_dataset(cfg, verify_media=False):
    """Ghép manifest trong RAM; giữ video ở từng part, kiểm leakage trên toàn lựa chọn."""
    root = Path(cfg["dataset"]).resolve()
    rows, receipts, hashes = [], [], {}
    for name, folder in selected_parts(cfg):
        part_rows = load_dataset(folder, verify_media=verify_media)
        digest = sha(folder / "manifest.jsonl")
        receipts.append({"part": name, "manifest_sha256": digest, "samples": len(part_rows)})
        for row in part_rows:
            if name is not None:
                # Không namespace source/speaker theo part: cùng người vẫn phải cùng split.
                if row["sha256"] in hashes and hashes[row["sha256"]] != name:
                    raise ValueError("Duplicate media across dataset parts")
                hashes[row["sha256"]] = name
                rel = Path(row["video"])
                if rel.is_absolute() or PureWindowsPath(row["video"]).drive or ".." in rel.parts:
                    raise ValueError("Unsafe media path in part")
                row = {
                    **row,
                    "dataset_part": name,
                    "video": (folder.relative_to(root) / rel).as_posix(),
                }
            rows.append(row)
    validate_rows(rows)
    signature = (
        receipts[0]["manifest_sha256"] if receipts[0]["part"] is None else fingerprint(receipts)
    )
    return rows, {"manifest_sha256": signature, "parts": receipts, "samples": len(rows)}


def temporal_targets(row, times, step, key="fake_intervals"):
    """Fraction of each output cell manipulated; context window is not the label extent.

    key="av_mismatch_intervals" cho target lệch tiếng-miệng của nhánh sync P2.
    """
    if row.get(key) is None:
        return np.full(len(times), -1.0, np.float32)
    ends = np.minimum(np.asarray(times) + step, row["duration_s"])
    widths = ends - times
    out = np.zeros(len(times), np.float32)
    for a, b in row[key]:
        out += np.maximum(0, np.minimum(ends, b) - np.maximum(times, a)) / np.maximum(widths, 1e-6)
    return np.clip(out, 0, 1)


def protocol_rows(rows, held_out=()):
    """Test-only unseen generators; never tune thresholds on their validation samples."""
    excluded = set(held_out)
    if not excluded:
        return rows
    known = {r["generator"] for r in rows if r["label"] == 1}
    if not excluded <= known:
        raise ValueError("Unknown held-out generator")
    if not any(r["split"] == "test" and r["generator"] in excluded for r in rows):
        raise ValueError("Held-out generator needs independent test samples")
    return [r for r in rows if r["split"] == "test" or r["generator"] not in excluded]
