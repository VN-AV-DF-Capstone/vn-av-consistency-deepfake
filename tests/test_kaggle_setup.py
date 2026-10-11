"""Kiểm bootstrap không tải package thật: isolation, resume, failure và archive safety."""

import importlib.util
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "bootstrap_worker", Path(__file__).resolve().parents[1] / "environments/bootstrap_worker.py"
)
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


@pytest.fixture
def installer(tmp_path, monkeypatch):
    """Giả lập process con, giữ filesystem/receipt thật để kiểm khi bị ngắt."""
    monkeypatch.setattr(bootstrap.platform, "system", lambda: "Linux")
    monkeypatch.setattr(bootstrap.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(bootstrap, "micromamba", lambda path: path / "micromamba")
    monkeypatch.setattr(bootstrap.subprocess, "check_output", lambda *a, **k: "numpy==1.23.5\n")
    for worker, source in (("avhubert", "av_hubert"), ("musetalk", "MuseTalk")):
        for relative in (
            f"environments/{worker}.yml",
            f"external/{source}/vn_av_revision.json",
            f"external/{source}/requirements.txt",
            f"external/{source}/fairseq/setup.py",
            f"external/{source}/avhubert/hubert.py",
            f"external/{source}/musetalk/utils/utils.py",
        ):
            target = tmp_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("fixture", encoding="utf-8")
    commands = []

    def execute(command, cwd, log, env=None):
        command = [str(x) for x in command]
        commands.append((command, env))
        if "-p" in command:
            python = Path(command[command.index("-p") + 1]) / "bin/python"
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("fixture")
        if "-c" in command:
            prefix = Path(command[0]).parents[1]  # <env_root>/.venv-<worker>/bin/python
            worker = prefix.name.removeprefix(".venv-")
            path = prefix.parent / ".kaggle-tools" / worker / "probe.json"
            path.write_text(json.dumps({"cuda_available": True}))

    monkeypatch.setattr(bootstrap, "execute", execute)
    return tmp_path, commands, execute


@pytest.mark.parametrize("worker", ["avhubert", "musetalk"])
def test_worker_isolated_and_ready_resume_rechecks_without_reinstall(
    installer, monkeypatch, worker
):
    root, commands, _ = installer
    monkeypatch.setenv("PYTHONPATH", "notebook-only-packages")
    monkeypatch.setenv("PIP_TARGET", "do-not-write-here")
    monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda-12.8")
    python = bootstrap.setup_worker(root, worker)
    assert python == root / f".venv-{worker}/bin/python"
    assert all("PYTHONPATH" not in env and "PIP_TARGET" not in env for _, env in commands)
    # CUDA_HOME của Kaggle (12.x) làm fairseq build extension .cu lệch torch cu118.
    assert all("CUDA_HOME" not in env for _, env in commands)
    for command, _ in commands:
        if "pip" in command:
            assert command[0] == str(python)
    installed = "\n".join(" ".join(c) for c, _ in commands)
    assert ("mmpose==1.1.0" in installed) == (worker == "musetalk")
    assert ("fairseq" in installed) == (worker == "avhubert")
    receipt = root / ".kaggle-tools" / worker / "environment.json"
    assert json.loads(receipt.read_text())["status"] == "ready"
    commands.clear()
    bootstrap.setup_worker(root, worker)
    assert len(commands) == 2  # pip check và import/CUDA probe vẫn phải chạy lại.
    assert all("install" not in command for command, _ in commands)
    (root / "environments" / f"{worker}.yml").write_text("changed")
    with pytest.raises(ValueError, match="specification/source changed"):
        bootstrap.setup_worker(root, worker)


def test_env_root_keeps_environment_outside_checkout_and_drops_caches(installer):
    root, commands, _ = installer
    workers = root.parent / "kaggle-temp"
    cache = workers / ".kaggle-tools/mamba/pkgs/torch-2.0.1/lib.so"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"x" * 1000)
    python = bootstrap.setup_worker(root, "musetalk", env_root=workers)
    assert python == workers / ".venv-musetalk/bin/python"
    receipt = workers / ".kaggle-tools/musetalk/environment.json"
    assert json.loads(receipt.read_text())["status"] == "ready"
    # Checkout (Output Kaggle) không chứa env/receipt; cache gói conda bị xoá, pip không cache.
    assert not (root / ".venv-musetalk").exists() and not (root / ".kaggle-tools").exists()
    assert not cache.parent.parent.exists()
    assert all(env["PIP_NO_CACHE_DIR"] == "1" for _, env in commands)
    assert any(str(workers / ".kaggle-tools/mamba") in command for command, _ in commands)


def test_failed_install_resumes_but_failed_probe_is_not_ready(installer, monkeypatch):
    root, commands, real_execute = installer

    def fail(command, *args, **kwargs):
        if "--index-url" in command:
            raise RuntimeError("interrupted package download")
        return real_execute(command, *args, **kwargs)

    monkeypatch.setattr(bootstrap, "execute", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        bootstrap.setup_worker(root, "avhubert")
    receipt = root / ".kaggle-tools/avhubert/environment.json"
    assert json.loads(receipt.read_text())["status"] == "installing"
    monkeypatch.setattr(bootstrap, "execute", real_execute)
    commands.clear()
    bootstrap.setup_worker(root, "avhubert")
    assert "install" in commands[0][0]  # Môi trường đã tạo, tiếp tục dependency còn dở.

    def fail_probe(command, *args, **kwargs):
        if "-c" in command:
            raise RuntimeError("CUDA unavailable")
        return real_execute(command, *args, **kwargs)

    monkeypatch.setattr(bootstrap, "execute", fail_probe)
    with pytest.raises(RuntimeError, match="CUDA"):
        bootstrap.setup_worker(root, "avhubert")
    assert json.loads(receipt.read_text())["status"] != "ready"


def test_missing_source_unmanaged_environment_and_non_linux_fail_early(installer, monkeypatch):
    root, commands, _ = installer
    source = root / "external/av_hubert/fairseq/setup.py"
    source.unlink()
    with pytest.raises(FileNotFoundError, match="asset setup"):
        bootstrap.setup_worker(root, "avhubert")
    (root / ".venv-musetalk").mkdir()
    with pytest.raises(ValueError, match="unmanaged"):
        bootstrap.setup_worker(root, "musetalk")
    monkeypatch.setattr(bootstrap.platform, "system", lambda: "Windows")
    with pytest.raises(RuntimeError, match="Kaggle/Linux"):
        bootstrap.setup_worker(root, "musetalk")
    assert not commands


def test_archive_extraction_never_writes_other_members(tmp_path):
    archive, target = tmp_path / "package.tar.bz2", tmp_path / "micromamba"
    with tarfile.open(archive, "w:bz2") as package:
        for name in ("../escaped", "bin/micromamba"):
            member = tarfile.TarInfo(name)
            member.size = 7
            package.addfile(member, io.BytesIO(b"fixture"))
    bootstrap.extract_micromamba(archive, target)
    assert target.read_bytes() == b"fixture"
    assert not (tmp_path.parent / "escaped").exists()
    with tarfile.open(archive, "w:bz2") as package:
        member = tarfile.TarInfo("bin/micromamba")
        member.type, member.linkname = tarfile.SYMTYPE, "../elsewhere"
        package.addfile(member)
    with pytest.raises(ValueError, match="regular file"):
        bootstrap.extract_micromamba(archive, target)


@pytest.mark.parametrize("worker", ["avhubert", "musetalk"])
def test_generated_probe_is_valid_python(worker, tmp_path):
    compile(bootstrap.probe_code(worker, tmp_path, tmp_path / "probe.json", True), "probe", "exec")


@pytest.mark.parametrize("notebook,worker", [("generate", "musetalk"), ("prepare", "avhubert")])
@pytest.mark.parametrize("needed", [True, False])
def test_notebook_bootstrap_configures_only_its_worker(notebook, worker, tmp_path, needed):
    path = Path(__file__).resolve().parents[1] / "notebooks" / f"{notebook}.ipynb"
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    for cell in cells:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(path), "exec")
    setup = next(c for c in cells if c["id"] == notebook + "-worker-setup")
    commands = []
    cfg = {
        "musetalk": {"python": "unchanged"},
        "encoders": {"avhubert": {"python": "unchanged"}},
        "architectures": ["selected"],
        "methods": {"selected": {"encoder": "avhubert"}},
        "prepare_encoders": ["avhubert", "dinov2"] if needed else ["dinov2"],
        # Generator của phiên generate: chỉ cài worker MuseTalk khi phiên này sinh MuseTalk.
        "generation_generators": ["musetalk_1_5"] if needed else ["wav2lip_gan"],
        "generation": {
            "generators_by_split": {"test": ["musetalk_1_5" if needed else "wav2lip_gan"]}
        },
    }
    workers = tmp_path / "kaggle-temp"
    state = dict(
        ROOT=tmp_path,
        WORKERS=workers,
        cfg=cfg,
        settings=SimpleNamespace(),
        subprocess=SimpleNamespace(run=lambda args, **kwargs: commands.append((args, kwargs))),
        sys=SimpleNamespace(executable="notebook-python"),
    )
    exec(compile("".join(setup["source"]), "worker-cell", "exec"), state)
    if not needed:
        assert not commands
        assert cfg["musetalk"]["python"] == cfg["encoders"]["avhubert"]["python"] == "unchanged"
        return
    expected = str(workers / f".venv-{worker}/bin/python")
    assert commands == [
        (
            [
                "notebook-python",
                str(tmp_path / "environments/bootstrap_worker.py"),
                worker,
                "--root",
                str(tmp_path),
                "--env-root",
                str(workers),
            ],
            {"check": True},
        )
    ]
    assert cfg["musetalk"]["python"] == (expected if worker == "musetalk" else "unchanged")
    assert cfg["encoders"]["avhubert"]["python"] == (
        expected if worker == "avhubert" else "unchanged"
    )


def notebook_config_state(notebook, tmp_path, monkeypatch):
    """Thực thi cell cấu hình thật, thay các action download/train bằng recorder."""
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location("notebook_settings_fixture", root / "settings.py")
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    cells = json.loads((root / "notebooks" / f"{notebook}.ipynb").read_text(encoding="utf-8"))[
        "cells"
    ]
    sources = {c["id"]: "".join(c["source"]) for c in cells if c["cell_type"] == "code"}
    calls = []
    state = dict(
        ROOT=tmp_path,
        Path=Path,
        settings=settings,
        execute=lambda action, cfg: calls.append(action),
    )
    exec(sources[f"{notebook}-02"], state)
    return sources, state, calls


def part_folder(root, name="vn-av-df-data-part7", nested=True):
    """Part như Kaggle mount: gốc dataset (tên slug) chứa thư mục part có dataset_info.json."""
    folder = root / (f"slug-{name}" if nested else "") / name
    folder.mkdir(parents=True)
    (folder / "dataset_info.json").write_text("{}")
    return folder.parent if nested else folder


MAIN = [
    "avh_tcn",
    "p2_syncartifact",
    "tricue_mech",
    "tricue_w2l",
    "tricue_mech_w2l",
    "tricue_mech_no_lead",
    "tricue_mech_no_sync",
    "tricue_mech_no_art",
]
ANALYSIS = [
    "art_all",
    *[f"art_only_m{i}" for i in range(1, 6)],
    "art_m1_m5",
    *[f"art_drop_m{i}" for i in range(1, 6)],
]


@pytest.mark.parametrize(
    "selection,expected",
    [
        ("all", MAIN + ANALYSIS),
        (MAIN, MAIN),
        (["tricue_mech"], ["tricue_mech"]),
        (["art_all", "art_drop_m3"], ["art_all", "art_drop_m3"]),
    ],
)
def test_training_notebook_uses_edited_models_parts_and_hyperparameters(
    tmp_path, monkeypatch, selection, expected
):
    from vn_av_df import dataset

    monkeypatch.setattr(dataset, "training_dataset", lambda *args, **kwargs: ([], {"samples": 0}))
    sources, state, calls = notebook_config_state("train", tmp_path, monkeypatch)
    exec(sources["train-parameters"], state)
    # Đường dẫn đầy đủ copy từ Kaggle (gốc dataset): tự vào thư mục part bên trong.
    state.update(
        ARCHITECTURES=selection,
        SEEDS=[99],
        DATA_PARTS=[part_folder(tmp_path)],
        RUN_NAME="custom_run",
    )
    state["TRAINING"].update(epochs=3, lr=0.002, hidden=32)
    state["RECONSTRUCTION"]["epochs"] = 2
    state["SYNC"]["shift_frames"] = [2, 4]
    exec(sources["train-config"], state)
    cfg = state["cfg"]
    assert cfg["architectures"] == expected
    assert cfg["dataset_parts"] == ["vn-av-df-data-part7"] and cfg["seeds"] == [99]
    assert Path(cfg["dataset"]).name == "slug-vn-av-df-data-part7"
    assert Path(cfg["runs"]).name == "custom_run"
    assert cfg["training"]["lr"] == 0.002 and cfg["training"]["epochs"] == 3
    assert cfg["reconstruction"]["epochs"] == 2
    assert cfg["sync"]["shift_frames"] == [2, 4]
    assert calls == []  # Train chỉ đọc cache: không tải encoder, không cài worker.
    assert state["RUN_TEST"] is True  # Test ngay sau train.


@pytest.mark.parametrize(
    "selection,encoders,architectures",
    [
        ("all", ["avhubert", "dinov2"], ["avh_tcn", "tricue_mech"]),
        (["dinov2", "avhubert"], ["avhubert", "dinov2"], ["avh_tcn", "tricue_mech"]),
        (["avhubert"], ["avhubert"], ["avh_tcn"]),
    ],
)
def test_prepare_notebook_selects_encoders_and_feature_settings(
    tmp_path, monkeypatch, selection, encoders, architectures
):
    from vn_av_df import dataset

    monkeypatch.setattr(dataset, "training_dataset", lambda *args, **kwargs: ([], {"samples": 0}))
    sources, state, calls = notebook_config_state("prepare", tmp_path, monkeypatch)
    exec(sources["prepare-parameters"], state)
    parts = [
        part_folder(tmp_path, nested=False),
        part_folder(tmp_path / "other", "vn-av-df-data-part8"),
    ]
    state.update(ENCODERS=selection, DATA_PARTS=parts, DEVICE="cpu")
    state["DINO_FEATURES"]["crop_size"] = 112
    exec(sources["prepare-config"], state)
    cfg = state["cfg"]
    assert cfg["prepare_encoders"] == encoders and cfg["architectures"] == architectures
    # Nhiều part ở các dataset khác nhau: gốc chung, part tương đối với gốc đó.
    assert cfg["dataset_parts"] == [
        "vn-av-df-data-part7",
        "other/slug-vn-av-df-data-part8/vn-av-df-data-part8",
    ]
    assert cfg["prepare_devices"] is None
    assert cfg["encoders"]["dinov2"]["crop_size"] == 112
    assert calls == ["encoder_setup"]


def test_prepare_notebook_rejects_dinov2_without_avhubert_boxes(tmp_path, monkeypatch):
    sources, state, _ = notebook_config_state("prepare", tmp_path, monkeypatch)
    exec(sources["prepare-parameters"], state)
    state.update(ENCODERS=["dinov2"], DEVICE="cpu", DATA_PARTS=[part_folder(tmp_path)])
    with pytest.raises(ValueError, match="AV-HuBERT"):
        exec(sources["prepare-config"], state)


@pytest.mark.parametrize(
    "generators,session",
    [
        (
            "all",
            sorted(
                [
                    "wav2lip_gan",
                    "musetalk_1_5",
                    "diag_paste",
                    "diag_global",
                    *["mech_blend", "mech_lowres", "mech_vae", "mech_warp", "mech_temporal"],
                ]
            ),
        ),
        (["wav2lip_gan"], ["wav2lip_gan"]),
    ],
)
def test_generation_notebook_uses_edited_part_and_generator_settings(
    tmp_path, monkeypatch, generators, session
):
    sources, state, calls = notebook_config_state("generate", tmp_path, monkeypatch)
    clean = tmp_path / "slug-clean-part2" / "vn-av-df-data-part2"
    clean.mkdir(parents=True)
    (clean / "manifest.jsonl").write_text("")
    state.update(
        PART=2,
        GENERATION_NAME="vn-av-df-data-part2",
        CLIPS_PER_SPLIT=0,
        CLEAN_PART=clean.parent,  # Gốc dataset Kaggle: tự vào thư mục có manifest.jsonl.
        GENERATORS=generators,
        CRF=20,
        MUSETALK_BATCH_SIZE=2,
        PARTIAL_RANGE_S=[0.8, 0.8],
        DEVICE="cuda",  # Phiên Kaggle có GPU.
    )
    state["settings"].ROOT = tmp_path
    exec(sources["generate-config"], state)
    cfg = state["cfg"]
    assert cfg["data_part"] == 2 and cfg["clean_dataset"] == str(clean)
    assert Path(cfg["generated_dataset"]).name == "vn-av-df-data-part2"
    assert cfg["generation"]["clips_per_split"] == 0 and cfg["generation"]["crf"] == 20
    assert cfg["generation"]["partial_range_s"] == [0.8, 0.8]
    assert cfg["generation"]["fake_audio_modes_by_split"]["train"] == ["donor"]
    assert "audio_mode" not in cfg["generation"]
    assert cfg["musetalk"]["batch_size"] == 2
    assert cfg["generation_generators"] == session
    assert "split_ratios" not in cfg  # Split gán ở 05 export, không cấu hình ở generate.
    assert calls == ["validate", "plan", "generator_setup"]
    state["GENERATORS"] = ["sadtalker"]
    with pytest.raises(ValueError, match="GENERATORS"):
        exec(sources["generate-config"], state)
    # Phiên không GPU: chỉ MechFake không VAE / chẩn đoán.
    state.update(DEVICE="cpu", GENERATORS=["mech_blend", "mech_lowres", "mech_warp"])
    exec(sources["generate-config"], state)
    assert state["cfg"]["generation_generators"] == ["mech_blend", "mech_lowres", "mech_warp"]
    state["GENERATORS"] = ["mech_vae"]
    with pytest.raises(ValueError, match="Không có GPU"):
        exec(sources["generate-config"], state)


def test_generation_notebook_merges_parallel_session_outputs(tmp_path, monkeypatch):
    import vn_av_df.generation as generation

    sources, state, calls = notebook_config_state("generate", tmp_path, monkeypatch)
    clean = tmp_path / "clean" / "vn-av-df-data-part1"
    clean.mkdir(parents=True)
    (clean / "manifest.jsonl").write_text("")
    state.update(CLEAN_PART=clean, DEVICE="cuda", GENERATORS=["musetalk_1_5", "mech_vae"])
    state["settings"].ROOT = tmp_path
    previous = []
    for name in ("gpu", "cpu"):  # Output hai notebook chạy song song, cùng cấu trúc repo.
        folder = tmp_path / "input" / name / "vn-av-df-forensics"
        (folder / "generation/plans").mkdir(parents=True)
        (folder / "generation/plans/vn-av-df-data-part1.json").write_text("{}")
        (folder / "datasets/vn-av-df-data/vn-av-df-data-part1").mkdir(parents=True)
        previous.append(folder)
    merged = []
    monkeypatch.setattr(generation, "merge_outputs", lambda cfg, pairs: merged.append(pairs))
    state["PREVIOUS_OUTPUT"] = previous
    exec(sources["generate-config"], state)
    # Mỗi phiên một cặp (plan, output), đúng thứ tự PREVIOUS_OUTPUT.
    assert [[p.relative_to(folder) for p in pair] for folder, pair in zip(previous, merged[0])] == [
        [
            Path("generation/plans/vn-av-df-data-part1.json"),
            Path("datasets/vn-av-df-data/vn-av-df-data-part1"),
        ]
    ] * 2
    state["PREVIOUS_OUTPUT"] = [*previous, tmp_path / "missing"]
    with pytest.raises(AssertionError, match="PREVIOUS_OUTPUT"):
        exec(sources["generate-config"], state)
