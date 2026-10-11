"""MuseTalk 1.5: source/weights có provenance, worker sinh trung gian lossless."""

import json
import os
import sys
from pathlib import Path

from vn_av_df import worker
from vn_av_df.common.runtime import read_json, require_file, run, sha


def setup(cfg):
    """Tải tài sản upstream; dependency MuseTalk nằm ở env worker riêng."""
    from vn_av_df.assets import fetch_file, pinned_repository, pinned_snapshot

    repo = Path(cfg["musetalk"]["repo"])
    pinned_repository("https://github.com/TMElyralab/MuseTalk.git", repo)
    root = repo / "models"
    for model, target, patterns in (
        ("TMElyralab/MuseTalk", root, ["musetalkV15/*"]),
        (
            "stabilityai/sd-vae-ft-mse",
            root / "sd-vae",
            ["config.json", "diffusion_pytorch_model.bin"],
        ),
        (
            "openai/whisper-tiny",
            root / "whisper",
            ["config.json", "pytorch_model.bin", "preprocessor_config.json"],
        ),
        ("yzd-v/DWPose", root / "dwpose", ["dw-ll_ucoco_384.pth"]),
    ):
        pinned_snapshot(model, target, patterns)
    face = root / "face-parse-bisent"
    face.mkdir(parents=True, exist_ok=True)
    target = face / "79999_iter.pth"
    if not target.exists():
        import gdown

        temporary = target.with_suffix(".partial")
        if not gdown.download(
            id="154JgKpzCPW82qINcVieuPH3fZ2e0P812", output=str(temporary), quiet=False
        ):
            raise RuntimeError("MuseTalk face parser download failed; rerun setup")
        temporary.replace(target)
    fetch_file(
        "https://download.pytorch.org/models/resnet18-5c106cde.pth", face / "resnet18-5c106cde.pth"
    )
    return provenance(cfg)


def provenance(cfg):
    """Hash toàn bộ weight/config inference và commit, phát hiện đổi asset khi resume."""
    repo = Path(cfg["musetalk"]["repo"])
    require_file(repo / "musetalk/utils/utils.py", "MuseTalk source")
    checkpoint = require_file(repo / "models/musetalkV15/unet.pth", "MuseTalk 1.5 checkpoint")
    required = [
        "musetalkV15/musetalk.json",
        "sd-vae/config.json",
        "sd-vae/diffusion_pytorch_model.bin",
        "whisper/pytorch_model.bin",
        "whisper/preprocessor_config.json",
        "whisper/config.json",
        "dwpose/dw-ll_ucoco_384.pth",
        "face-parse-bisent/79999_iter.pth",
        "face-parse-bisent/resnet18-5c106cde.pth",
    ]
    hashes = {name: sha(require_file(repo / "models" / name, name)) for name in required}
    revision = run(["git", "rev-parse", "HEAD"], cwd=repo).strip()
    lock = read_json(repo / "vn_av_revision.json")
    if revision != lock["commit"]:
        raise ValueError("MuseTalk source changed")
    return dict(
        generator="musetalk_1_5",
        version=revision,
        checkpoint_sha256=sha(checkpoint),
        dependencies=hashes,
        worker_sha256=sha(Path(__file__).with_name("musetalk_worker.py")),
    )


def synthesize(cfg, video, audio_video, output, frames, width, height, gpu=None):
    """Cùng giao diện Wav2Lip; output FFV1/PCM, không encode fake thêm một lần lossy.

    Worker sống suốt lượt sinh (nạp VAE/UNet/Whisper/DWPose một lần), một worker mỗi GPU.
    """
    options = cfg["musetalk"]
    command = [
        options.get("python", sys.executable),
        "-B",
        Path(__file__).with_name("musetalk_worker.py"),
        "--serve",
        json.dumps(options, sort_keys=True),
    ]

    def start():
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        return worker.Worker(command, cwd=options["repo"], env=env)

    bridge = worker.get(("musetalk", gpu, *map(str, command)), start)
    bridge.call(
        dict(
            video=str(Path(video).resolve()),
            audio_video=str(Path(audio_video).resolve()),
            output=str(Path(output).resolve()),
            frames=frames,
            width=width,
            height=height,
        ),
        timeout=3600,
    )
