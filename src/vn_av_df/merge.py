"""Gộp Output các phiên sinh độc lập (đã tải về local) thành một part chờ review.

Mỗi phiên Kaggle tự lập plan (cùng seed, part sạch, cấu hình → trùng nhau) và chỉ sinh cặp của
generator phiên đó. Gộp = chép cặp đã sinh (`generation.merge_outputs`: kiểm cùng plan, cùng chữ
ký code, provenance, hash video) rồi ghi candidates/review như cuối `generate`. Không sửa
generation.py / render.py: hash hai file đó nằm trong chữ ký output của các phiên đã chạy.
"""

from collections import Counter
from pathlib import Path

from vn_av_df.common.runtime import fingerprint, read_json, write_json
from vn_av_df.data.groups import write_manifest
from vn_av_df.generation import csv_write, merge_outputs


def session_sources(root, plan, generated, folders):
    """[(plan.json, thư mục output)] trong từng Output đã tải.

    Mỗi folder là thư mục tải về của một phiên (chứa `vn-av-df-forensics/`) hoặc chính thư mục
    `vn-av-df-forensics` trong đó; plan/output nằm cùng đường dẫn tương đối như trong repo.
    """
    pairs = []
    for folder in map(Path, folders):
        nested = folder / "vn-av-df-forensics"
        base = nested if nested.is_dir() else folder
        pair = tuple(base / Path(p).relative_to(root) for p in (plan, generated))
        missing = [str(p) for p in pair if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Output thiếu {missing}; kiểm tra MERGE_INPUTS")
        pairs.append(pair)
    return pairs


def merge_sessions(cfg, sources):
    """Gộp khi mọi cặp của plan đã có record ở một trong các Output; thiếu thì báo, không chép."""
    if len(sources) < 2:
        raise ValueError("Cần ít nhất hai Output để gộp")
    plan = read_json(sources[0][0])
    present = {p.name for _, folder in sources for p in (Path(folder) / "records").glob("*.json")}
    left = Counter(
        job.get("generator", "wav2lip_gan")
        for job in plan["jobs"]
        if f"{fingerprint(job)[:20]}.json" not in present
    )
    if left:
        raise ValueError(
            f"Còn {sum(left.values())} cặp chưa sinh {dict(left)}: chạy tiếp phiên của generator "
            "đó (PREVIOUS_OUTPUT = Output của nó), tải lại rồi gộp"
        )
    merged = merge_outputs(cfg, sources)
    out = Path(cfg["generated_dataset"])
    rows, skipped = [], []
    for job in plan["jobs"]:  # Theo thứ tự plan, như generate.
        record = read_json(out / "records" / f"{fingerprint(job)[:20]}.json")
        if isinstance(record, dict):  # Cặp bị bỏ (không thấy mặt / lỗi riêng cặp).
            skipped.append(record)
        else:
            rows.extend(record)
    # MechFake lấy hộp miệng từ real của parent: cặp AI của parent bị bỏ thì bỏ luôn MechFake.
    produced = {r["sample_id"] for r in rows}
    orphans = {
        r["sample_id"]
        for r in rows
        if r.get("box_sample_id") and r["box_sample_id"] not in produced
    }
    rows = [r for r in rows if r["sample_id"] not in orphans]
    write_manifest(out / "candidates.jsonl", rows)
    write_json(out / "skipped_pairs.json", skipped)
    # Mặc định keep: review chỉ để đánh reject/uncertain mẫu lỗi.
    csv_write(out / "review.csv", [{"sample_id": r["sample_id"], "decision": "keep"} for r in rows])
    return {
        **merged,
        "status": "complete",
        "samples": len(rows),
        "pairs_skipped": len(skipped),
        "mechfake_orphans_dropped": len(orphans),
        "next": "Run review then finalize; pending media is not training data",
    }
