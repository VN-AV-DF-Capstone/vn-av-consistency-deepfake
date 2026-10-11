"""One public output: video suspicion and scored temporal intervals."""

import tempfile
from pathlib import Path

import torch

from vn_av_df.common.runtime import read_json, write_json
from vn_av_df.experiment import load_model, resolve_checkpoint
from vn_av_df.features.registry import assemble, combined_signature, make_encoder
from vn_av_df.metrics import intervals
from vn_av_df.models import output_valid, pool_score


def chosen_checkpoint(cfg):
    if cfg.get("checkpoint"):
        return Path(cfg["checkpoint"])
    # Demo dùng DEMO_METHOD + seed đầu trong training.json; không xếp hạng model.
    runs = read_json(Path(cfg["runs"]) / "training.json")["runs"]
    selected = next(
        (
            r
            for r in runs
            if r["architecture"] == cfg["demo_method"] and r["seed"] == cfg["seeds"][0]
        ),
        None,
    )
    if selected is None:
        raise ValueError("DEMO_METHOD/seed has no completed checkpoint; set CHECKPOINT explicitly")
    return resolve_checkpoint(cfg, selected["checkpoint"])


class Analyzer:
    """Inference dùng đúng encoder/config/mask và threshold đã chọn trên validation."""

    def __init__(self, cfg, encoder=None, artifact_encoder=None):
        self.cfg = cfg
        self.model, self.state = load_model(chosen_checkpoint(cfg), cfg["device"])
        self.architecture = self.state["model_config"]["architecture"]
        kind = self.state["encoder_config"].get("kind")
        encoder_cfg = cfg.get("encoders", {}).get(kind, cfg.get("encoder"))
        self.encoder = encoder or make_encoder(encoder_cfg)
        signature = self.encoder.signature
        self.artifact_encoder = None
        recorded = self.state.get("artifact_encoder_config")
        if recorded:
            # P2 cần thêm DINOv2 miệng; cấu hình local phải trùng lúc train.
            self.artifact_encoder = artifact_encoder or make_encoder(
                cfg.get("encoders", {}).get(recorded["kind"], recorded)
            )
            signature = combined_signature(signature, self.artifact_encoder.signature)
        if signature != self.state["feature_signature"]:
            raise ValueError(
                "Encoder/preprocessing differs from training; use the same assets and settings"
            )

    @torch.no_grad()
    def analyze(self, video, output=None, cache_folder=None):
        """Một clip; cache_folder cho demo nhiều head dùng lại features cùng backbone."""
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(cache_folder or folder) / (self.encoder.signature + ".npz")
            meta = self.encoder.extract({"video": str(video)}, cache)
            artifact = None
            if self.artifact_encoder is not None:
                extra = cache.with_name(self.artifact_encoder.signature + ".npz")
                extra_meta = self.artifact_encoder.extract(
                    {"video": str(video), "boxes": str(cache)}, extra
                )
                artifact = (extra, extra_meta)
            arrays, times, meta = assemble(self.architecture, (cache, meta), artifact)
            tensors = [
                None if x is None else torch.as_tensor(x, device=self.cfg["device"]) for x in arrays
            ]
            scores = self.model(*tensors).sigmoid()
            valid = output_valid(self.model, tensors[2])
            score = pool_score(scores, valid, self.state["top_fraction"])
            threshold = self.state["thresholds"]["temporal"]
            # Điểm từng nhánh P2 là bằng chứng phụ, chưa có ngưỡng riêng nên không xuất interval.
            evidence = {
                name: value.sigmoid()
                for name, value in getattr(self.model, "outputs", {}).items()
                if value is not None
            }
            result = {
                "video_score": float(score) if score is not None else None,
                "intervals": intervals(
                    scores.cpu().numpy(),
                    valid.cpu().numpy(),
                    times,
                    meta["step_s"],
                    meta["duration_s"],
                    threshold,
                )
                if threshold is not None
                else [],
            }
            if evidence:
                result["evidence"] = {
                    name: float(value) if value is not None else None
                    for name, value in (
                        (name, pool_score(values, valid, self.state["top_fraction"]))
                        for name, values in evidence.items()
                    )
                }
            if self.cfg.get("demo_details", False):
                if evidence:
                    result["evidence_scores"] = {k: v.cpu().tolist() for k, v in evidence.items()}
                result.update(
                    method=self.state.get("method_name", self.architecture),
                    coverage=float(valid.float().mean()),
                    thresholds=self.state["thresholds"],
                    scores=scores.cpu().tolist(),
                    times_s=times.tolist(),
                    valid=valid.cpu().tolist(),
                    dataset_manifest_sha256=self.state["manifest_sha256"],
                    seed=self.state["seed"],
                )
        if output:
            write_json(output, result)
        return result
