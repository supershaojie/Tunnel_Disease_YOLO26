"""Prepare DRSC detection candidates, then render a human-selected five-column figure."""

import argparse
import json
from pathlib import Path


def read_json(path):
    """Read a UTF-8 JSON file, including files saved by Windows editors."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    """Keep inference, CPU review and final layout available as separate operations."""
    base = "/root/autodl-tmp/projects/Tunnel_Disease_YOLO26"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("all", "prepare", "candidates", "render", "package"))
    parser.add_argument("--base", type=Path, default=Path(base))
    parser.add_argument("--out", type=Path, default=Path(base) / "runs/visualization/drsc_detection_v1")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--device", default="0", help="GPU passed to each existing model environment in turn")
    parser.add_argument("--overrides", type=Path, help="Small per-model JSON overrides for changed paths")
    parser.add_argument("--reuse-only", action="store_true", help="Report incomplete caches without starting inference")
    parser.add_argument("--limit", type=int, default=80, help="Maximum proposed candidates, with no filler cases")
    parser.add_argument("--bundle", type=Path, help="Existing full bundle.json or unpacked candidate bundle.json")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--ids", help="Five comma-separated candidate IDs or canonical image IDs")
    selection.add_argument("--cases", type=Path, help="Selection JSON with optional common original-coordinate crops")
    args = parser.parse_args()
    if not 1 <= args.limit <= 80:
        parser.error("--limit must be between 1 and 80")
    args.out.mkdir(parents=True, exist_ok=True)
    bundle_path = args.bundle or args.out / "bundle.json"
    if args.command in {"all", "prepare"}:
        from .cache import prepare

        data = args.data or args.base / "datasets/Tunnel_Crack_AugFirst_Diverse5x_RandomSplit_7_2_1_seed42/data.yaml"
        bundle = prepare(
            data,
            args.base,
            args.out,
            args.device,
            overrides=read_json(args.overrides) if args.overrides else None,
            reuse_only=args.reuse_only,
        )
        bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        (args.out / "sources.json").write_text(json.dumps(bundle["sources"], indent=2), encoding="utf-8")
    else:
        bundle = read_json(bundle_path)
    for image in bundle["images"]:
        image["path"] = str((bundle_path.parent / image["path"]).resolve())
    if args.command in {"all", "candidates"}:
        from .analyze import rank_candidates
        from .render import package_candidates, previews

        candidates = rank_candidates(bundle, args.out, args.limit)
        previews(bundle, candidates, args.out)
        package_candidates(bundle, candidates, args.out)
    elif args.command == "render":
        from .render import package_final, render_final

        if args.cases:
            cases = read_json(args.cases)
        elif args.ids:
            candidates_path = bundle_path.parent / "candidates.json"
            candidates = read_json(candidates_path) if candidates_path.exists() else []
            mapping = {row["case_id"]: row["image_id"] for row in candidates}
            cases = {
                "status": "user_selected",
                "cases": [{"case_id": key, "image_id": mapping.get(key, key)} for key in args.ids.split(",")],
            }
        else:
            parser.error("render requires --ids or --cases; all never publishes an automatic final selection")
        render_final(bundle, cases, args.out)
        package_final(args.out)
    elif args.command == "package":
        from .render import package_candidates

        package_candidates(bundle, read_json(args.out / "candidates.json"), args.out)
    print(f"Completed {args.command}: {args.out.resolve()}")


if __name__ == "__main__":
    main()
