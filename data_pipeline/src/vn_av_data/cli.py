"""Independent data preparation CLI. No Torch imports."""

import argparse
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="vn-av-data")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in (
        "setup",
        "collect",
        "download",
        "cut",
        "review",
        "export",
        "validate",
    ):
        p = sub.add_parser(name)
        if name in ("setup", "cut"):
            p.add_argument("--config", default="configs/data.yaml")
        if name in ("collect", "download"):
            p.add_argument(
                "--cookies-from-browser",
                choices=("firefox", "chrome", "edge", "brave"),
                help="Use a browser session when YouTube rejects anonymous requests",
            )
            p.add_argument("--force-ipv4", action="store_true")
        if name == "collect":
            p.add_argument("--input", required=True)
            p.add_argument("--output", required=True)
        elif name == "download":
            p.add_argument("--sources", required=True)
            p.add_argument("--output", default="data/raw")
            p.add_argument("--limit", type=int, default=0)
            p.add_argument("--dry-run", action="store_true")
        elif name == "cut":
            p.add_argument("--manifest", default="data/raw/sources.jsonl")
            p.add_argument("--output", required=True)
        elif name in ("review", "export"):
            p.add_argument("--review", required=True)
            p.add_argument("--root", required=True)
            if name == "review":
                p.add_argument("--port", type=int, default=8001)
            else:
                p.add_argument("--output", required=True)
                p.add_argument("--dataset-id", required=True)
                p.add_argument("--annotations", help="JSONL: clip_id and relation_annotations")
                p.add_argument("--part", type=int, default=1)
                p.add_argument("--split-seed", type=int, default=42)
                p.add_argument("--split-ratios", help='JSON, vd. {"train":0.6,...}')
                p.add_argument("--min-test-groups", type=int, default=5, help="Ít hơn: cảnh báo")
                p.add_argument(
                    "--max-clips-per-speaker", type=int, default=0, help="0 = không giới hạn"
                )
                p.add_argument(
                    "--split-history", nargs="*", default=[], help="split-lock part trước"
                )
        elif name == "validate":
            p.add_argument("--dataset", required=True)
    args = parser.parse_args(argv)
    if hasattr(args, "config"):
        from vn_av_data.common.runtime import load_config

        cfg = load_config(args.config)
    if args.command == "collect":
        from vn_av_data.data.collect import collect_part

        result = collect_part(
            args.input,
            args.output,
            cookies_from_browser=args.cookies_from_browser,
            force_ipv4=args.force_ipv4,
        )
    elif args.command == "setup":
        from vn_av_data.assets import setup_assets

        result = setup_assets(cfg)
    elif args.command == "download":
        from vn_av_data.data.acquisition import download_sources

        result = download_sources(
            args.sources,
            args.output,
            args.cookies_from_browser,
            args.force_ipv4,
            args.limit,
            args.dry_run,
        )
    elif args.command == "cut":
        from vn_av_data.data.curation import curate_sources

        result = curate_sources(args.manifest, args.output, cfg)
    elif args.command == "review":
        import uvicorn

        from vn_av_data.serving.review import create_review_app

        uvicorn.run(create_review_app(args.review, args.root), host="127.0.0.1", port=args.port)
        return 0
    elif args.command == "export":
        from vn_av_data.data.export import export_dataset

        result = export_dataset(
            args.review,
            args.root,
            args.output,
            args.dataset_id,
            args.annotations,
            part=args.part,
            seed=args.split_seed,
            ratios=json.loads(args.split_ratios) if args.split_ratios else None,
            history=args.split_history,
            min_test_groups=args.min_test_groups,
            max_clips_per_speaker=args.max_clips_per_speaker or None,
        )
    else:
        from vn_av_data.contract import validate_bundle

        _, result = validate_bundle(args.dataset)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def entrypoint():
    try:
        return main()
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
