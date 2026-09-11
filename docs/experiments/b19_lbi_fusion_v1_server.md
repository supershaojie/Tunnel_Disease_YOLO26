# LBI-Fusion v1 server procedure

Implementation does not start or stop AutoDL training. The following commands are for the user's later deployment.
The expected native runtime is read and checked: Python3.12.3, PyTorch2.8.0+cu128, Ultralytics8.4.98 and RTX4090.
Do not reinstall or change CUDA/TF32/CUBLAS to bypass a mismatch. Formal training is **NOT STARTED** by this task.

## Deploy a verified commit, without touching active worktrees

Copy the full final Commit SHA from `deployment_report.txt` into TARGET below. Do not copy SHA from an image.
The remote must be the user's repository; no upstream or main push is used. Existing worktrees must be clean and idle.
The same lock protects stage execution and deployment. No task/session is killed by these commands.

```bash
set -euo pipefail
BASE=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_LBI_Fusion_v1
BRANCH=codex/exp-yolo26n-b19-lbi-fusion-v1
TARGET=PASTE_FULL_VERIFIED_COMMIT_SHA_HERE
[[ "$TARGET" =~ ^[0-9a-f]{40}$ ]] || { echo 'A full verified Git SHA is required'; exit 2; }
[[ "$(git -C "$BASE" remote get-url origin)" == https://github.com/supershaojie/Tunnel_Disease_YOLO26.git ]] || exit 2
exec 9>"$BASE/.yolo26n_b19_lbi_fusion_v1.lock"
flock -n 9 || { echo 'LBI stage/deployment is active; code unchanged'; exit 3; }
if pgrep -af 'python.*(run_b19_lbi_fusion|verify_b19_lbi_fusion|finish_b19_lbi_fusion)'; then
    echo 'Residual LBI process found; inspect it before deployment'; exit 3
fi
/root/miniconda3/bin/python - "$WORK" <<'PY'
import os
import sys
from pathlib import Path
work = Path(sys.argv[1]).resolve()
active = []
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name) in {os.getpid(), os.getppid()}:
        continue
    try:
        command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        executable = (proc / 'exe').resolve().name.lower()
        cwd = (proc / 'cwd').resolve()
    except (OSError, RuntimeError):
        continue
    if ('python' in executable or 'yolo' in executable) and (
        cwd == work or work in cwd.parents or str(work) in command
    ):
        active.append((proc.name, str(cwd), command))
if active:
    print('Active processes may be reading WORK; deployment refused:', active)
    raise SystemExit(3)
PY
if ! git -C "$BASE" cat-file -e "$TARGET^{commit}" 2>/dev/null; then
    # One bounded attempt; no tags, no retry loop, no stale FETCH_HEAD on failure.
    timeout 90s git -C "$BASE" -c http.version=HTTP/1.1 \
        -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=20 \
        fetch --no-tags origin "refs/heads/$BRANCH" || { echo 'FETCH FAILED; stop'; exit 4; }
    GOT="$(git -C "$BASE" rev-parse FETCH_HEAD)"
    if [[ "$GOT" != "$TARGET" ]]; then
        /root/miniconda3/bin/python - "$TARGET" "$GOT" <<'PY'
import sys
a, b = sys.argv[1:]
first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
print('expected=', repr(a), 'length=', len(a))
print('actual=', repr(b), 'length=', len(b), 'first_difference=', first)
PY
        exit 4
    fi
fi
git -C "$BASE" cat-file -e "$TARGET^{commit}"
if [[ -e "$WORK" ]]; then
    [[ "$(git -C "$WORK" rev-parse --show-toplevel)" == "$WORK" ]] || exit 3
    [[ -z "$(git -C "$WORK" status --porcelain)" ]] || { echo 'Worktree is dirty; preserve it'; exit 3; }
    git -C "$WORK" checkout --detach "$TARGET"
else
    git -C "$BASE" worktree add --detach "$WORK" "$TARGET"
fi
[[ "$(git -C "$WORK" rev-parse HEAD)" == "$TARGET" ]]
flock -u 9
exec 9>&-
```

This checks the shared lock, ordinary named LBI processes and Python/YOLO processes bound to WORK by cwd or command.
Do not update code used by another process hidden by operating-system process visibility restrictions.
No fetch is needed when the exact user-verified target object is already local. On any failed fetch, mismatch or
activity check, stop; never train from a residual FETCH_HEAD or previous HEAD.

## Start only when explicitly desired

Inspect existing tmux sessions with `tmux ls` and `tmux list-panes -a -F '#S:#I.#P #{pane_pid} #{pane_current_command} #{pane_dead}'`.
Do not reuse or kill an existing session containing active work. After confirming a new session name is free:

```bash
tmux new-session -d -s lbi_fusion_v1 \
    'cd /root/autodl-tmp/projects/Tunnel_Disease_YOLO26_LBI_Fusion_v1 && bash tools/experiments/server_b19_lbi_fusion_v1.sh train'
```

Outside tmux use `tmux attach -t lbi_fusion_v1`; inside tmux use `tmux switch-client -t lbi_fusion_v1`.
Detach while preserving work with Ctrl+B, then D. A dead pane does not execute commands typed into it.
The session ending is not evidence of training completion; inspect stage status and native epoch logs.

The normal interface is exactly:

```bash
bash tools/experiments/server_b19_lbi_fusion_v1.sh train
```

It prints experiment, stage, WORK, actual HEAD, interpreter, device visibility and the per-attempt log path before any
large Python import. It then audits source/runtime/complete b19 args, starts an independent native B32/640/MuSGD/AMP
preflight and only on a matching PASS creates a fresh formal trainer. The verifier prints each native batch's progress.
The final interpreter import location is in provenance and must point into WORK. `B19_PYTHON` may select an existing
interpreter; default is `/root/miniconda3/bin/python`. A runtime mismatch fails without changing the recipe.

Attempts live under `runs/detect/.yolo26n_b19_lbi_fusion_v1.<stage>.attempt.*` with `console.log`, command and status
receipts. Full preflight evidence is in its own `*_preflight_*` directory; it is copied into formal provenance only
after PASS. Tracebacks and SIGINT/SIGTERM records remain. Python's failure code is preserved across tee; a zero Python
code plus tee failure also fails. A process merely existing does not prove it reached epoch1. There is no fixed
seconds-without-output failure rule. Existing formal output is never overwritten or silently renamed with a `2` suffix.

## Later stages

```bash
bash tools/experiments/server_b19_lbi_fusion_v1.sh preflight  # development diagnosis only; no formal training
bash tools/experiments/server_b19_lbi_fusion_v1.sh test       # fixed val/test and separate b19 comparison
bash tools/experiments/server_b19_lbi_fusion_v1.sh diagnose   # fixed validation IDs only
bash tools/experiments/server_b19_lbi_fusion_v1.sh package    # requires complete canonical results
```

Preflight failure blocks train and leaves evidence. It does not authorize batch/LR changes or another architecture.
If the formal run already exists, inspect it; this v1 entry does not resume or replace it. If evaluation failed midway,
its partial output is retained and must be understood before retrying. A valid completed stage receipt can be reused
only while all bound files, source and weights still match.

The archive contains only this run, its required results and committed source closure, with a manifest and SHA256
sidecar. Export it only after successful read-back verification. The implementation's local fixture package is a
packaging test, not a formal result archive, and cannot authorize the server run.
