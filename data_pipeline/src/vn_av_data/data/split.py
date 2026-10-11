"""Chia train/validation/test cho clip sạch của một part, giữ nguyên nhóm người/nguồn.

Chạy ở bước 05 export: split là thuộc tính của clip sạch, mọi phiên generate (mỗi phiên một
generator) đọc chung một split. Nhóm liên thông nối các clip cùng nguồn, cùng speaker, cùng hash
hoặc cùng cụm repost; cả nhóm luôn vào cùng một split nên không rò rỉ người/nguồn qua split.
"""

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ("train", "validation", "test")
DEFAULT_SPLIT_RATIOS = {"train": 0.6, "validation": 0.15, "test": 0.25}
REGISTRY_SCHEMA = "vn-av-df-split-registry-v1"


def split_ratios(value=None):
    """Tỷ lệ theo số clip sạch; giữ nguyên nhóm nên số thực tế có thể lệch mục tiêu."""
    value = DEFAULT_SPLIT_RATIOS if value is None else value
    if (
        not isinstance(value, dict)
        or set(value) != set(SPLITS)
        or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v < 1
            for v in value.values()
        )
        or not math.isclose(sum(value.values()), 1.0, abs_tol=1e-9)
    ):
        raise ValueError("split_ratios must contain positive train/validation/test summing to 1")
    return {key: float(value[key]) for key in SPLITS}


def identity_list(value):
    """CSV uses semicolon-separated IDs; JSONL may use a list."""
    if value is None:
        return []
    values = value.split(";") if isinstance(value, str) else value
    if not isinstance(values, (list, tuple)):
        raise ValueError("Identity fields must be a list or semicolon-separated string")
    return sorted(
        {
            str(x).strip()
            for x in values
            if str(x).strip().lower() not in ("", "unknown", "nan", "none", "-1")
        }
    )


def identity_nodes(row):
    """Local IDs are dataset-scoped; hashes and global speaker IDs join datasets."""
    namespace = str(row.get("dataset", ""))
    nodes = [(namespace, "source", str(row["source_id"]))]
    # Hash trùng và cụm repost do review gán phải theo cùng split dù khác URL.
    if row.get("sha256"):
        nodes.append(("global", "media_sha256", row["sha256"]))
    if row.get("source_sha256"):
        nodes.append(("global", "source_sha256", row["source_sha256"]))
    if row.get("duplicate_group_id"):
        nodes.append(("global", "duplicate", str(row["duplicate_group_id"])))
    nodes.extend((namespace, "source", x) for x in identity_list(row.get("parent_ids")))
    speakers = identity_list(row.get("speaker_id")) + identity_list(row.get("speaker_ids"))
    nodes.extend((namespace, "speaker", x) for x in speakers)
    nodes.extend(("global", "speaker", x) for x in identity_list(row.get("global_speaker_ids")))
    if row.get("group_id"):
        nodes.append((namespace, "locked_group", str(row["group_id"])))
    return nodes


def connected_groups(rows):
    """Resolve transitive source, donor, speaker and repost connections."""
    parent = {}

    def find(node):
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while node != root:
            parent[node], node = root, parent[node]
        return root

    for row in rows:
        nodes = identity_nodes(row)
        for node in nodes[1:]:
            parent[find(node)] = find(nodes[0])
    members = defaultdict(list)
    for row in rows:
        members[find(identity_nodes(row)[0])].append(row)
    return [
        sorted(group, key=lambda r: r["sample_id"])
        for group in sorted(members.values(), key=lambda group: min(r["sample_id"] for r in group))
    ]


def load_history(paths, part):
    """Gộp split-lock của các part trước (tuỳ chọn); rỗng = chia part độc lập.

    Trả về (assignments, covered_parts). Khi bật, lịch sử phải phủ đủ mọi part trước `part`.
    """
    if not paths:
        return [], []
    prior, versions, covered = {}, [], set()
    for path in paths:
        lock = json.loads(Path(path).read_text(encoding="utf-8"))
        if lock.get("schema") != REGISTRY_SCHEMA:
            raise ValueError("Need a cumulative split-lock with complete source identities")
        versions.append(lock.get("data_part", 0))
        # Registry cũ luôn tích lũy; registry mới ghi rõ part thực sự có trong file.
        covered.update(lock.get("covered_parts", range(1, lock.get("data_part", 0) + 1)))
        for row in lock["assignments"]:
            key = row["clip_id"]
            row = {**row, "sample_id": row.get("sample_id", key)}
            if key in prior and prior[key] != row:
                raise ValueError("Conflicting split history for the same clip")
            prior[key] = row
    if not prior:
        raise ValueError("Empty split history")
    if part > 1 and max(versions) != part - 1:
        raise ValueError("Use the cumulative split-lock of the immediately preceding part")
    if part > 1 and covered != set(range(1, part)):
        raise ValueError("Split history must cover all preceding parts; include independent locks")
    return list(prior.values()), sorted(covered)


def split_clean(rows, seed=42, history=(), ratios=None):
    """Chia theo nhóm liên thông, tỷ lệ mặc định 60/15/25; chỉ giữ split part cũ khi có history."""
    ratios = split_ratios(ratios)
    current = [{**r, "sample_id": r["clip_id"]} for r in rows]
    ids = {r["clip_id"] for r in current}
    if len(ids) != len(current) or ids & {r["clip_id"] for r in history}:
        raise ValueError("Duplicate clean clip across parts; keep each clip in one part")
    old_hashes = {r["sha256"] for r in history if r.get("sha256")}
    if any(r.get("sha256") in old_hashes for r in current):
        raise ValueError("Duplicate clean media across parts")
    if any(r.get("split") not in SPLITS for r in history):
        raise ValueError("Invalid split history")
    groups = connected_groups([*history, *current])
    if len(groups) < 3 and not history:
        raise ValueError("Need three independent speaker/source groups to split a part")
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)
    total = len(rows) + len(history)
    target = {split: ratio * total for split, ratio in ratios.items()}
    counts = Counter()
    result = []
    fresh = []
    for group in groups:
        locked = {r["split"] for r in group if r["clip_id"] not in ids}
        if len(locked) > 1:
            raise ValueError("New part connects previously separated splits; review identities")
        if locked:
            split = next(iter(locked))
            counts[split] += len(group)
            result.extend({**r, "split": split} for r in group if r["clip_id"] in ids)
        else:
            fresh.append(group)
    for i, group in enumerate(fresh):
        empty = [s for s in target if not counts[s]]
        choices = empty if len(fresh) - i == len(empty) else list(target)
        split = min(
            choices,
            key=lambda s: sum(
                (counts[t] + (len(group) if t == s else 0) - target[t]) ** 2 for t in target
            ),
        )
        counts[split] += len(group)
        result.extend({**r, "split": split} for r in group)
    return result


def assign_splits(rows, part, seed=42, ratios=None, history_paths=()):
    """Gán split cho clip sạch của part; trả (rows có split, split-lock ghi kèm part)."""
    ratios = split_ratios(ratios)
    history, covered = load_history(history_paths, part)
    assigned = {r["clip_id"]: r["split"] for r in split_clean(rows, seed, history, ratios)}
    rows = [{**r, "split": assigned[r["clip_id"]]} for r in rows]
    everything = [*history, *({**r, "sample_id": r["clip_id"]} for r in rows)]
    groups = connected_groups(everything)
    registry = {
        "schema": REGISTRY_SCHEMA,
        "data_part": part,
        "split_ratios": ratios,
        "seed": seed,
        "covered_parts": sorted({*covered, part}),
        "assignments": everything,
        "groups": len(groups),
        # Số nhóm người/nguồn độc lập theo split: test cần đủ nhóm để kết luận về người mới.
        "groups_by_split": dict(Counter(group[0]["split"] for group in groups)),
        "counts": dict(Counter(r["split"] for r in [*history, *rows])),
        "note": "Source/identity grouped; channel-disjointness and near-duplicate review required separately",
    }
    return rows, registry
