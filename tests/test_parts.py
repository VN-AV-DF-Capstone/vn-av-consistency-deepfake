"""Part selection, portable media paths and split inheritance across collection batches."""

from collections import Counter

import pytest
from vn_av_data.data.split import load_history, split_clean

from vn_av_df.common.runtime import read_json, sha, write_json
from vn_av_df.data.groups import write_manifest
from vn_av_df.dataset import SCHEMA, media_path, training_dataset
from vn_av_df.generation import make_plan, validate_plan_settings


def bundle(root, part, split="train", speaker=None):
    """Small finalized bundle with real files, independent of encoder/generator weights."""
    folder = root / part
    video = folder / "clips" / f"{part}.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(part.encode())
    rows = [
        {
            "sample_id": part,
            "video": f"clips/{part}.mp4",
            "sha256": sha(video),
            "duration_s": 4,
            "label": 0,
            "fake_intervals": [],
            "split": split,
            "source_id": part,
            "speaker_id": speaker or part,
            "generator": "none",
            "generator_version": "fixture",
            "review_status": "keep",
        }
    ]
    seal(folder, rows)
    return folder, rows


def seal(folder, rows):
    """Re-finalize test-only fixtures so failures exercise semantics, not stale hashes."""
    write_manifest(folder / "manifest.jsonl", rows)
    write_json(
        folder / "dataset_info.json",
        {
            "schema_version": SCHEMA,
            "samples": len(rows),
            "manifest_sha256": sha(folder / "manifest.jsonl"),
        },
    )


def test_select_single_subset_all_and_kaggle_nested_parts(tmp_path):
    p1, _ = bundle(tmp_path, "vn-av-df-data-part1")
    nested = tmp_path / "vn-av-df-data-part2"
    p2, _ = bundle(nested, nested.name)
    cfg = {"dataset": str(tmp_path), "dataset_parts": [p1.name]}
    one, receipt = training_dataset(cfg, verify_media=True)
    assert len(one) == 1 and media_path(tmp_path, one[0]).read_bytes() == p1.name.encode()
    cfg["dataset_parts"] = "all"
    both, full = training_dataset(cfg, verify_media=True)
    assert len(both) == 2 and full["manifest_sha256"] != receipt["manifest_sha256"]
    assert [r["dataset_part"] for r in both] == [p1.name, p2.name]
    cfg["dataset_parts"] = [f"{p2.name}/{p2.name}", p1.name]
    assert training_dataset(cfg)[1] == full  # Order of selection does not change the experiment.
    assert len(list(tmp_path.rglob("*.mp4"))) == 2  # Loader never copies video.


def test_part_union_rejects_leakage_and_duplicate_samples(tmp_path):
    p1, first = bundle(tmp_path, "vn-av-df-data-part1", speaker="same_person")
    p2, second = bundle(tmp_path, "vn-av-df-data-part2", split="test", speaker="same_person")
    cfg = {"dataset": str(tmp_path), "dataset_parts": "all"}
    with pytest.raises(ValueError, match="leakage"):
        training_dataset(cfg)
    second[0].update(speaker_id="other", sample_id=first[0]["sample_id"])
    seal(p2, second)
    with pytest.raises(ValueError, match="duplicate sample_id"):
        training_dataset(cfg)
    second[0].update(sample_id=p2.name, sha256=first[0]["sha256"], split="train")
    seal(p2, second)
    with pytest.raises(ValueError, match="Duplicate media"):
        training_dataset(cfg)
    assert read_json(p1 / "dataset_info.json")["samples"] == 1


@pytest.mark.parametrize("selection", [[], ["missing"], ["../outside"], ["p", "p"]])
def test_invalid_or_missing_selection_never_silently_uses_other_parts(tmp_path, selection):
    bundle(tmp_path, "p")
    with pytest.raises(ValueError):
        training_dataset({"dataset": str(tmp_path), "dataset_parts": selection})


def clean(index, speaker=None):
    return {"clip_id": f"c{index}", "source_id": f"s{index}", "speaker_id": speaker or f"p{index}"}


def test_incremental_split_inherits_old_speaker_and_rejects_transitive_bridge():
    history = split_clean([clean(i) for i in range(6)])
    new = [clean(10 + i, row["speaker_id"]) for i, row in enumerate(history)]
    assigned = split_clean(new, history=history)
    old = {r["speaker_id"]: r["split"] for r in history}
    assert all(r["split"] == old[r["speaker_id"]] for r in assigned)
    assert split_clean([new[0]], history=history)[0]["split"] == old[new[0]["speaker_id"]]
    a = history[0]
    b = next(r for r in history if r["split"] != a["split"])
    bridge = {**clean(99, a["speaker_id"]), "parent_ids": [b["source_id"]]}
    with pytest.raises(ValueError, match="previously separated splits"):
        split_clean([bridge], history=history)
    with pytest.raises(ValueError, match="Duplicate clean clip"):
        split_clean([clean(0)], history=history)


def test_optional_registry_keeps_source_hash_identity_and_checks_preceding_part(tmp_path):
    assert load_history([], 2) == ([], [])
    history = split_clean([clean(i) for i in range(3)])
    history[0]["source_sha256"] = "raw-hash"
    path = tmp_path / "split-lock.json"
    write_json(path, {"schema": "vn-av-df-split-registry-v1", "assignments": history})
    assigned = split_clean(
        [{**clean(90), "source_sha256": "raw-hash"}], history=load_history([path], 1)[0]
    )
    assert assigned[0]["split"] == history[0]["split"]
    write_json(path, {"assignments": history})
    with pytest.raises(ValueError, match="complete source identities"):
        load_history([path], 1)
    write_json(
        path, {"schema": "vn-av-df-split-registry-v1", "data_part": 1, "assignments": history}
    )
    with pytest.raises(ValueError, match="immediately preceding"):
        load_history([path], 3)
    assert load_history([path], 2)[0] == history


def test_independent_split_uses_60_15_25_without_breaking_speaker_groups():
    rows = [clean(i, speaker=f"person{i // 2}") for i in range(200)]
    assigned = split_clean(rows)
    assert Counter(r["split"] for r in assigned) == {"train": 120, "validation": 30, "test": 50}
    speakers = {}
    for row in assigned:
        speakers.setdefault(row["speaker_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in speakers.values())
    assert split_clean(rows) == assigned
    custom = split_clean(rows, ratios={"train": 0.9, "validation": 0.05, "test": 0.05})
    assert Counter(r["split"] for r in custom) == {"train": 180, "validation": 10, "test": 10}


@pytest.mark.parametrize(
    "ratios",
    [
        {},
        {"train": 1},
        {"train": 0.8, "validation": 0.1, "test": 0.2},
        {"train": 1, "validation": 0, "test": 0},
        {"train": float("nan"), "validation": 0.1, "test": 0.1},
    ],
)
def test_invalid_split_ratios_fail_before_assignment(ratios):
    with pytest.raises(ValueError, match="split_ratios"):
        split_clean([clean(i) for i in range(100)], ratios=ratios)


@pytest.mark.parametrize("part", [1, 2, 6])
def test_plan_reads_export_split_and_freezes_settings(tmp_path, monkeypatch, part):
    from vn_av_df import generation

    ratios = {"train": 0.8, "validation": 0.1, "test": 0.1}
    rows = [{**clean(i), "duration_s": 6} for i in range(100)]
    assigned = {r["clip_id"]: r["split"] for r in split_clean(rows, ratios=ratios)}
    rows = [{**r, "split": assigned[r["clip_id"]]} for r in rows]
    monkeypatch.setattr(
        generation, "validate_bundle", lambda *a, **kw: (rows, {"manifest_sha256": "fixture"})
    )
    write_json(tmp_path / "dataset_info.json", {"split": {"covered_parts": [part]}})
    cfg = {
        "clean_dataset": str(tmp_path),
        "data_part": part,
        "seed": 42,
        "plan": str(tmp_path / "plan.json"),
        "generation": {"clips_per_split": 0},
    }
    # Split lấy nguyên từ part sạch (05_export), generate không chia lại.
    assert make_plan(cfg)["splits"] == {"train": 80, "validation": 10, "test": 10}
    plan = read_json(cfg["plan"])
    assert plan["split"] == {"covered_parts": [part]}
    assert {r["clip_id"]: r["split"] for r in plan["assignments"]} == assigned
    validate_plan_settings(cfg, plan)
    with pytest.raises(ValueError, match="frozen plan"):
        validate_plan_settings({**cfg, "seed": 7}, plan)
    unsplit = [{k: v for k, v in r.items() if k != "split"} for r in rows]
    monkeypatch.setattr(
        generation, "validate_bundle", lambda *a, **kw: (unsplit, {"manifest_sha256": "x"})
    )
    with pytest.raises(ValueError, match="no split"):
        make_plan({**cfg, "plan": str(tmp_path / "plan2.json")})


def test_reenable_history_requires_all_independent_parts(tmp_path):
    paths = []
    for part in (1, 2):
        path = tmp_path / f"part{part}.json"
        write_json(
            path,
            {
                "schema": "vn-av-df-split-registry-v1",
                "data_part": part,
                "covered_parts": [part],
                "assignments": split_clean([clean(part * 100 + i) for i in range(10)]),
            },
        )
        paths.append(str(path))
    with pytest.raises(ValueError, match="cover all preceding parts"):
        load_history(paths[1:], 3)
    history, covered = load_history(paths, 3)
    assert covered == [1, 2]
    assert len(history) == 20
    # Reusing a previous identity still inherits its split when the feature is enabled.
    new = split_clean([clean(999, history[0]["speaker_id"])], history=history)
    assert new[0]["split"] == history[0]["split"]


def test_export_writes_only_current_part(tmp_path, capsys):
    from vn_av_df.actions import execute

    p1, _ = bundle(tmp_path, "vn-av-df-data-part1")
    bundle(tmp_path, "vn-av-df-data-part2")
    execute("export", {"dataset": str(tmp_path), "generated_dataset": str(p1)})
    import zipfile

    with zipfile.ZipFile(p1.with_suffix(".zip")) as archive:
        assert all(p.startswith(p1.name + "/") for p in archive.namelist())
    assert not (tmp_path / "vn-av-df-data-part2.zip").exists()
