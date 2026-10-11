"""Worker AV-HuBERT tách môi trường fairseq khỏi transformers của môi trường chính."""

import json
import sys
from pathlib import Path

from vn_av_df import worker
from vn_av_df.common.runtime import fingerprint, read_json, require_file, sha


class AVHubertVideoEncoder:
    """Cache native 25 Hz có PTS/mask; strict signature cả upstream lẫn preprocessing."""

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.worker = None
        repo = Path(cfg["repo"])
        paths = {
            name: require_file(cfg[name], name) for name in ("checkpoint", "landmarks", "mean_face")
        }
        for name in (
            "hubert.py",
            "hubert_pretraining.py",
            "utils.py",
            "preparation/align_mouth.py",
        ):
            paths[name] = require_file(repo / "avhubert" / name, name)
        paths["worker"] = Path(__file__).with_name("avhubert_worker.py")
        paths["adapter"] = Path(__file__)
        paths["media"] = Path(__file__).parents[1] / "data/media.py"
        self.signature = fingerprint(
            dict(
                assets={k: sha(v) for k, v in paths.items()},
                options={
                    k: cfg.get(k, default)
                    for k, default in (
                        ("chunk_frames", 200),
                        ("overlap_frames", 50),
                        ("max_duration", 60),
                        ("max_side", 640),
                        ("output_stride", 5),
                    )
                },
                format="avhubert-native-v2",
            )
        )

    def extract(self, row, output):
        """Nhận media đã render; worker không tải model hoặc tự chỉnh lệch tiếng."""
        output = Path(output).resolve()
        video = require_file(row["video"], "video")
        source = fingerprint({"assets": {"video": sha(video)}, "variant": {"kind": "clean"}})
        if output.is_file() and output.with_suffix(".json").is_file():
            meta = read_json(output.with_suffix(".json"))
            if (
                meta.get("feature_signature") == self.signature
                and meta.get("source_fingerprint") == source
                and meta.get("feature_sha256") == sha(output)
            ):
                return meta
        output.parent.mkdir(parents=True, exist_ok=True)
        # Worker sống suốt lượt trích (nạp fairseq/checkpoint/dlib một lần), mỗi encoder một
        # worker; worker chết thì lần sau tự mở lại.
        if self.worker is None or not self.worker.alive:
            self.worker = worker.Worker(
                [
                    self.cfg.get("python", sys.executable),
                    "-B",
                    Path(__file__).with_name("avhubert_worker.py"),
                    "--serve",
                    json.dumps(self.cfg, sort_keys=True),
                ]
            )
        self.worker.call(
            dict(
                video=str(video),
                output=str(output),
                signature=self.signature,
                source_fingerprint=source,
            ),
            timeout=3600,
        )
        return read_json(output.with_suffix(".json"))

    def close(self):
        if self.worker is not None:
            self.worker.close()
            self.worker = None
