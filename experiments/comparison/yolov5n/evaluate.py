"""Reload a trusted trained checkpoint in a fresh process; export unified val/test predictions and figures."""

import copy
from pathlib import Path

from common import UniqueStore, bootstrap, configuration, parser, print_plan, protocol, verify_sources
from comparison_io import digest, now, read_json, status, trusted_load, verify_weights, write_json


def main():
    p = parser(__doc__)
    p.add_argument("--split", action=UniqueStore, choices=("val", "test"))
    p.add_argument(
        "--result-dir",
        action=UniqueStore,
        type=Path,
        help="Run directory; outputs are evaluation/val or evaluation/test",
    )
    p.add_argument(
        "--threshold-from",
        action=UniqueStore,
        type=Path,
        help="Frozen threshold from evaluation of the same weights on val",
    )
    args = p.parse_args()
    cfg, audit = configuration(args)
    verify_sources()
    if args.dry_run:
        print_plan(cfg, audit)
        return
    from data import loader, manifest, save_manifest
    from reporting import counts, figures, save_exports, summary
    from comparisons import comparisons
    from runtime import environment, infer

    data = manifest(cfg["data"])
    provenance = verify_weights(cfg["weights"])
    if provenance["kind"] != "comparison_checkpoint":
        raise ValueError("Evaluate must reload this experiment's trained best, not original pretrained weights")
    imports = bootstrap(cfg["cache"])
    checkpoint = trusted_load(cfg["weights"])
    info = checkpoint.get("comparison", {})
    if info.get("model") != "yolov5n":
        raise ValueError("Wrong checkpoint backend identity")
    current = protocol(cfg, data)
    # An explicit evaluation variant is allowed, but data and training augmentation identity must still match.
    if any(info["public"][key] != current["public"][key] for key in ("data", "augment")):
        raise ValueError("Evaluation data/training augmentation identity mismatch with trained checkpoint")
    split = cfg["eval"]["split"]
    run = args.result_dir.resolve() if args.result_dir else Path(cfg["weights"]).resolve().parent.parent
    directory = run / "evaluation" / split
    threshold = None
    if split == "test":
        threshold_path = args.threshold_from or run / "evaluation/val/val_threshold.json"
        threshold = read_json(threshold_path)
        if (
            threshold.get("split") != "val"
            or threshold.get("weights_sha256") != digest(cfg["weights"])
            or threshold.get("protocol_sha256") != current["protocol_sha256"]
        ):
            raise ValueError("Test threshold must come from val with exactly the same weights/data/evaluation protocol")
    directory.mkdir(parents=True, exist_ok=False)
    started = now()
    status(directory, "evaluate", "RUNNING", started=started, purpose=info["purpose"])
    try:
        from utils.torch_utils import select_device

        device = select_device(cfg["train"]["device"], batch_size=cfg["eval"]["batch"])
        model = (
            copy.deepcopy(checkpoint["ema"] if checkpoint.get("ema") is not None else checkpoint["model"])
            .float()
            .to(device)
            .eval()
        )
        unfused = sum(p.numel() for p in model.parameters())
        if model.model[-1].nc != 1 or model.model[-1].__class__.__name__ != "Detect":
            raise ValueError("Expected original nc=1 anchor Detect head")
        if cfg["eval"]["fuse"]:
            model.fuse()
        dataloader, dataset = loader(data["splits"][split]["records"], cfg, split)
        records = data["splits"][split]["records"]
        values, curves, predictions, targets, _, actual = infer(model, dataloader, cfg, records)
        if split == "val":
            if curves is None:
                raise ValueError("Cannot select a val threshold without GT")
            threshold = {
                "split": "val",
                "conf": max(values["legacy_curve_peak"]["confidence"], cfg["eval"]["conf"]),
                "selection": "val frozen smoothed interpolated F1 argmax; exact counts evaluated separately",
                "weights_sha256": digest(cfg["weights"]),
                "protocol_sha256": current["protocol_sha256"],
                "data_split_sha256": data["splits"]["val"]["sha256"],
            }
            write_json(directory / "val_threshold.json", threshold)
        values["val_frozen"] = {**counts(predictions, targets, threshold["conf"]), "threshold_source": threshold}
        values.update(
            model="yolov5n",
            split=split,
            images=len(records),
            boxes=sum(len(t) for t in targets),
            predictions=sum(len(p) for p in predictions),
            weights=str(Path(cfg["weights"]).resolve()),
            weight_sha256=digest(cfg["weights"]),
            purpose=info["purpose"],
            **current,
            actual=actual,
            parameters={"unfused": unfused, "fused_or_evaluated": sum(p.numel() for p in model.parameters())},
            units_percent="multiply fraction values by 100; CSV stores fraction",
            main_table_PR_policy="UNVERIFIED historical policy; use clearly named fixed/val-selected/legacy fields",
        )
        save_exports(directory, predictions, targets, records)
        write_json(
            directory / "image_index.json",
            [{k: v for k, v in r.items() if k not in ("image", "label")} for r in records],
        )
        write_json(directory / "metrics.json", values)
        write_json(directory / "requested_config.json", cfg)
        write_json(directory / "config_audit.json", audit)
        write_json(directory / "environment.json", environment(imports))
        save_manifest(directory, data)
        summary(directory / "summary.csv", values)
        figures(directory, curves, values)
        comparisons(
            directory,
            [{"image_id": r["image_id"], "predictions": p, "gt": t} for r, p, t in zip(records, predictions, targets)],
            cfg,
            data,
        )
        status(directory, "evaluate", "COMPLETED", started=started, ended=now(), images=len(records))
        print({k: values[k] for k in ("split", "images", "AP50", "AP75", "mAP50-95", "purpose")})
        print(f"Results: {directory}")
    except BaseException as error:
        status(directory, "evaluate", "FAILED", started=started, ended=now(), reason=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
