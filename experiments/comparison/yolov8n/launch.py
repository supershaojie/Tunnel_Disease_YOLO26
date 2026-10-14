"""Independent tmux sessions with collision protection and Python pipeline exit codes."""

import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from common import ROOT, UniqueStore, parser, resolve, result_directory


def main():
    """Launch one model without polling, waiting for or stopping another GPU process."""
    p = parser("tmux launch")
    p.add_argument("--session", action=UniqueStore, default="cmp-y8n")
    p.add_argument("--mode", action=UniqueStore, choices=["train", "evaluate", "benchmark"], default="train")
    args = p.parse_args()
    cfg = resolve(args)
    if args.dry_run:
        print(json.dumps({"session": args.session, "mode": args.mode, "status": "DRY_RUN", "name": cfg["name"]}))
        return
    if subprocess.run(["tmux", "has-session", "-t", args.session], capture_output=True, check=False).returncode == 0:
        raise FileExistsError(f"tmux session exists: {args.session}; choose an explicit new --session")
    forwarded, i = [], 0
    while i < len(sys.argv[1:]):
        item = sys.argv[1:][i]
        if item in ["--session", "--mode"]:
            i += 2
        elif item.startswith(("--session=", "--mode=")):
            i += 1
        else:
            forwarded.append(item)
            i += 1
    log_root = Path(cfg["cache_dir"]).resolve() / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log = log_root / (args.session + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".log")
    if args.mode == "train":
        result = (
            Path(cfg["train"]["resume"]).resolve().parent.parent
            if cfg["train"]["resume"]
            else Path(cfg["project"]).resolve() / cfg["name"]
        )
    else:
        if not cfg["weights"]:
            raise ValueError(f"{args.mode} requires --weights <this-experiment-best.pt>")
        run = Path(cfg["weights"]).resolve().parent.parent
        result = result_directory(run, args, args.mode, cfg["eval"]["split"])
    command = shlex.join([sys.executable, "-u", str(Path(__file__).parent / (args.mode + ".py")), *forwarded])
    shell = (
        f"set -o pipefail\n{command} 2>&1 | tee {shlex.quote(str(log))}\n"
        "python_status=${PIPESTATUS[0]}\n"
        f"if [ -d {shlex.quote(str(result))} ]; then cp {shlex.quote(str(log))} "
        f"{shlex.quote(str(result / (args.mode + '.log')))}; fi\n"
        'exit "$python_status"\n'
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", args.session, "-c", str(ROOT), "bash", "-lc", shell], check=True)
    print(
        json.dumps(
            {
                "status": "SESSION_CREATED",
                "session": args.session,
                "log": str(log),
                "training_status": "NOT_PROVEN_BY_SESSION_CREATION",
                "result": str(result),
            }
        )
    )


if __name__ == "__main__":
    main()
