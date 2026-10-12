"""Numbered scripts call these functions; no CLI arguments needed."""

import json
from pathlib import Path


def execute(action, cfg, resume=False):
    if action == "validate":
        from vn_av_data.contract import validate_bundle

        from vn_av_df.data.media import probe

        rows, info = validate_bundle(cfg["clean_dataset"], probe=probe)
        from collections import Counter

        if not info["splits"]:
            raise ValueError(
                "Clean part has no split; run 05_export again (split is assigned there)"
            )
        # Split đã gán ở 05_export; ở đây chỉ báo lại số clip và speaker theo split.
        result = {
            **info,
            "speakers": {
                split: sorted({r.get("speaker_id") or "-" for r in rows if r["split"] == split})
                for split in ("train", "validation", "test")
            },
            "speaker_clips": dict(Counter(r.get("speaker_id") for r in rows)),
        }
    elif action == "generator_setup":
        from vn_av_df.generators import setup_generators

        result = setup_generators(cfg)
    elif action == "encoder_setup":
        from vn_av_df.assets import setup_assets

        result = setup_assets(cfg)
    elif action == "plan":
        from vn_av_df.generation import make_plan

        result = make_plan(cfg)
    elif action == "generate":
        from vn_av_df.generation import generate

        result = generate(cfg)
    elif action == "merge":
        from vn_av_df.merge import merge_sessions, session_sources

        sources = session_sources(
            cfg["root"], cfg["plan"], cfg["generated_dataset"], cfg["merge_inputs"]
        )
        result = merge_sessions(cfg, sources)
    elif action == "finalize":
        from vn_av_df.generation import finalize

        result = finalize(cfg)
    elif action == "review":
        import uvicorn

        from vn_av_df.web import review_app

        uvicorn.run(review_app(cfg), host="127.0.0.1", port=8002)
        return
    elif action == "prepare":
        from vn_av_df.experiment import prepare

        result = prepare(cfg)
    elif action == "train":
        from vn_av_df.experiment import train_selected

        result = train_selected(cfg, resume)
    elif action == "evaluate":
        from vn_av_df.experiment import evaluate

        result = evaluate(cfg)
    elif action == "demo":
        import uvicorn

        from vn_av_df.web import demo_app

        uvicorn.run(demo_app(cfg), host="127.0.0.1", port=8000)
        return
    elif action == "export":
        import shutil

        from vn_av_df.dataset import load_dataset

        # Export chỉ part vừa review, không ZIP lại toàn bộ collection để train.
        dataset = Path(cfg["generated_dataset"])
        load_dataset(dataset, verify_media=True)
        result = {"zip": shutil.make_archive(str(dataset), "zip", dataset.parent, dataset.name)}
    else:
        raise ValueError(action)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
