import csv
import json
import shutil

import pytest

from vn_av_data.common.runtime import sha, write_json
from vn_av_data.contract import validate_bundle
from vn_av_data.data.curation import write_csv
from vn_av_data.data.export import export_dataset


def _review_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture
def reviewed(tmp_path, monkeypatch):
    root = tmp_path / "reviewed"
    root.mkdir()
    rows = []
    for i in range(4):  # 4 nguồn độc lập: còn ≥3 nhóm để chia split khi loại một clip.
        path = root / f"clip{i}.mp4"
        path.write_bytes(f"fixture-video-{i}".encode())
        rows.append(
            {
                "clip_id": f"clip{i}",
                "file_path": path.name,
                "decision": "keep",
                "sync_status": "reviewed_match",
                "source_id": f"source{i}",
                "source_start_s": 10,
                "source_end_s": 14,
                "sha256": sha(path),
            }
        )
    rows.append({"clip_id": "deleted", "file_path": "missing.mp4", "decision": "reject"})
    review = root / "review.csv"
    write_csv(review, rows)
    monkeypatch.setattr("vn_av_data.data.export.probe", lambda _: {"duration_s": 4})
    return root, review, rows


def test_export_relocation_labels_and_rerun(reviewed, tmp_path):
    root, review, _ = reviewed
    labels = tmp_path / "annotations.jsonl"
    labels.write_text(
        json.dumps(
            {
                "clip_id": "clip0",
                "relation_annotations": {
                    "motion_speech": {"known": [[0, 4]], "positive": [[1, 2]]}
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "export"
    report = export_dataset(review, root, output, "fixture_v001", labels)
    assert report["clips"] == 4
    assert set(report["splits"]) == {"train", "validation", "test"}
    relocated = tmp_path / "other_machine" / "dataset"
    shutil.copytree(output, relocated)
    rows, _ = validate_bundle(relocated)
    # Split gán lúc export, ghi kèm split-lock để generate/part sau đọc lại.
    assert all(row["split"] and row["video"].startswith("clips/") for row in rows)
    lock = json.loads((relocated / "split-lock.json").read_text(encoding="utf-8"))
    assert {r["clip_id"]: r["split"] for r in lock["assignments"]} == {
        r["clip_id"]: r["split"] for r in rows
    }
    assert rows[0]["relation_annotations"]["motion_speech"]["positive"] == [[1, 2]]
    assert rows[0]["source_start_s"] == 10
    # Chạy lại: không đổi thì giữ bản cũ; duyệt thêm thì dựng lại cả part.
    assert export_dataset(review, root, output, "fixture_v001", labels)["unchanged"]
    rows_now = _review_rows(review)
    rows_now[2]["decision"] = "reject"
    write_csv(review, rows_now)
    report = export_dataset(review, root, output, "fixture_v001", labels)
    assert report["replaced"] and validate_bundle(output)[1]["clips"] == 3
    assert not (tmp_path / "export.old").exists() and not (tmp_path / "export.partial").exists()
    (relocated / rows[0]["video"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_bundle(relocated)


def test_export_caps_clips_per_speaker_and_warns_on_small_test(reviewed, tmp_path):
    root, review, rows = reviewed
    # 4 clip: clip0-2 cùng một speaker, clip3 speaker khác; thêm 2 clip nguồn riêng cho đủ nhóm.
    for row in rows[:3]:
        row["speaker_id"] = "alice"
    rows[3]["speaker_id"] = "bob"
    for i in (4, 5):
        path = root / f"clip{i}.mp4"
        path.write_bytes(f"fixture-video-{i}".encode())
        rows.append(
            {
                **rows[3],
                "clip_id": f"clip{i}",
                "file_path": path.name,
                "sha256": sha(path),
                "source_id": f"source{i}",
                "speaker_id": f"carol{i}",
            }
        )
    write_csv(review, rows)
    report = export_dataset(review, root, tmp_path / "capped", "test", max_clips_per_speaker=2)
    assert report["clips"] == 5 and len(report["clips_capped_per_speaker"]) == 1
    assert report["clips_capped_per_speaker"][0] in {"clip0", "clip1", "clip2"}
    # 60/15/25 mặc định; test ít hơn 5 nhóm độc lập chỉ cảnh báo, vẫn export.
    assert report["split_groups_by_split"]["test"] < 5 and "Test chỉ có" in report["warnings"][0]
    info = json.loads((tmp_path / "capped/dataset_info.json").read_text(encoding="utf-8"))
    assert info["max_clips_per_speaker"] == 2
    assert info["split"]["split_ratios"] == {"train": 0.6, "validation": 0.15, "test": 0.25}


def test_export_requires_sync_confirmation_and_detects_changed_review(reviewed, tmp_path):
    root, review, rows = reviewed
    rows[0]["sync_status"] = "unverified"
    write_csv(review, rows)
    with pytest.raises(ValueError, match="confirm synchrony"):
        export_dataset(review, root, tmp_path / "out", "test")
    rows[0]["sync_status"] = "reviewed_match"
    write_csv(review, rows)
    (root / "clip0.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="has changed"):
        export_dataset(review, root, tmp_path / "out", "test")


@pytest.mark.parametrize("bad_path", ["../outside.mp4", "/outside.mp4", "C:/outside.mp4"])
def test_consumer_rejects_unsafe_paths_even_with_updated_manifest_hash(
    reviewed, tmp_path, bad_path
):
    root, review, _ = reviewed
    output = tmp_path / "out"
    export_dataset(review, root, output, "test")
    manifest = output / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[0]["video"] = bad_path
    manifest.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    info = json.loads((output / "dataset_info.json").read_text())
    info["manifest_sha256"] = sha(manifest)
    write_json(output / "dataset_info.json", info)
    with pytest.raises(ValueError, match="Unsafe"):
        validate_bundle(output)


def test_dedup_and_invalid_annotation_do_not_silently_lose_labels(reviewed, tmp_path):
    root, review, rows = reviewed
    rows.insert(1, {**rows[0], "clip_id": "duplicate"})
    write_csv(review, rows)
    report = export_dataset(review, root, tmp_path / "dedup", "test")
    assert report["clips"] == 4
    assert report["duplicate_clips_skipped"] == ["duplicate"]
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps(
            {
                "clip_id": "duplicate",
                "relation_annotations": {"sequence": {"known": [[0, 4]], "positive": [[1, 2]]}},
            }
        )
    )
    with pytest.raises(ValueError, match="duplicate with annotations"):
        export_dataset(review, root, tmp_path / "bad", "test", labels)
