"""Train original YOLOv5n with explicit configuration and isolated comparison outputs."""

import contextlib
import sys
import traceback
from pathlib import Path

from common import configuration, parser, print_plan, verify_sources
from comparison_io import now, read_json, status, verify_weights, write_json


class Tee:
    def __init__(self, terminal, stream):
        self.terminal, self.stream = terminal, stream

    def write(self, value):
        self.terminal.write(value)
        self.stream.write(value)
        self.stream.flush()
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.stream.flush()

    def __getattr__(self, key):
        return getattr(self.terminal, key)


def main():
    args = parser(__doc__).parse_args()
    cfg, audit = configuration(args)
    verify_sources()
    if args.dry_run:
        print_plan(cfg, audit)
        return
    from data import manifest, save_manifest
    from runtime import train_native, validate_resume

    data_manifest = manifest(cfg["data"])
    initial = verify_weights(cfg["weights"])
    if initial["kind"] != "official_pretrained":
        raise ValueError(
            "New training initializes only from verified official COCO YOLOv5n weights; --resume is explicit"
        )
    directory = (Path(cfg["project"]) / cfg["name"]).resolve()
    restored = None
    if cfg["train"]["resume"]:
        restored = validate_resume(cfg, data_manifest, directory)
        previous = read_json(directory / "initialization.json")
        if initial["sha256"] != previous["sha256"]:
            raise ValueError("Resume cannot change the original official initialization provenance")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "initialization.json", initial)
        write_json(directory / "raw_user_config.json", audit["raw_user_config"])
        write_json(directory / "official_defaults.json", audit["official_defaults"])
        write_json(directory / "initial_requested_config.json", cfg)
        write_json(directory / "initial_config_audit.json", audit)
        save_manifest(directory, data_manifest)
    started = now()
    write_json(directory / "requested_config.json", cfg)
    write_json(directory / "config_audit.json", audit)
    write_json(directory / "backend.lock.json", read_json(Path(__file__).parent / "backend.lock.json"))
    status(
        directory,
        "train",
        "RUNNING",
        started=started,
        purpose="SMOKE" if cfg["name"].startswith("SMOKE") else "EXPERIMENT",
    )
    try:
        with (directory / "train.log").open("a", encoding="utf-8") as stream, contextlib.redirect_stdout(
            Tee(sys.stdout, stream)
        ), contextlib.redirect_stderr(Tee(sys.stderr, stream)):
            train_native(cfg, audit, data_manifest, directory, restored)
        status(directory, "train", "COMPLETED", started=started, ended=now())
        print(f"Completed: {directory}")
    except BaseException as error:
        status(directory, "train", "FAILED", started=started, ended=now(), reason=f"{type(error).__name__}: {error}")
        with (directory / "train.log").open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        raise


if __name__ == "__main__":
    main()
