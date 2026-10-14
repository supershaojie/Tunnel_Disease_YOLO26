"""Check configuration, exact data bytes, backend imports, environment and weight provenance."""

from common import bootstrap, configuration, parser, print_plan, protocol, verify_sources
from comparison_io import verify_weights


def main():
    args = parser(__doc__).parse_args()
    cfg, audit = configuration(args)
    verify_sources()
    if args.dry_run:
        print_plan(cfg, audit)
        return
    from data import manifest
    from runtime import environment

    data = manifest(cfg["data"])
    weights = verify_weights(cfg["weights"])
    imports = bootstrap(cfg["cache"])
    value = {
        "state": "PREFLIGHT_PASSED",
        "weights": weights,
        "environment": environment(imports),
        "protocol": protocol(cfg, data),
        "data": {
            k: {"images": v["images"], "boxes": v["boxes"], "sha256": v["sha256"]} for k, v in data["splits"].items()
        },
        "historical_yaml_bytes_match": data["historical_yaml_bytes_match"],
        "baseline_reference": audit["baseline_reference"],
        "historical_optional_albumentations": "UNVERIFIED; chosen policy is explicit in augmentation protocol",
        "server_training": "NOT_STARTED",
    }
    import json

    print(json.dumps(value, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
