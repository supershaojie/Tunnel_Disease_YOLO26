#!/usr/bin/env bash
# Start a persistent interactive shell before dispatching the independent task script.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash experiments/dtr_sce/launch_tmux.sh {train|evaluate} [Python entry arguments...]
       bash experiments/dtr_sce/launch_tmux.sh --help

train forwards its arguments to train.py; evaluate forwards to validate.py.
Use train --help or evaluate --help for the actual Python CLI.
Defaults: /root/miniconda3/bin/python, sessions dtr-sce-b19 / dtr-sce-eval.
Overrides: DTR_SCE_PYTHON (absolute interpreter), DTR_SCE_SESSION (new session name),
           DTR_SCE_LOG_DIR (launcher log directory, outside the formal run).
Existing sessions are preserved and rejected. GPU0 may be shared with other experiments.
HELP
}

case "${1:---help}" in
    --help|-h) usage; exit 0 ;;
    train) mode=train; entry=train.py; default_session=dtr-sce-b19 ;;
    evaluate) mode=evaluate; entry=validate.py; default_session=dtr-sce-eval ;;
    *) usage >&2; exit 2 ;;
esac
shift
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
root=$(cd -- "$script_dir/../.." && pwd -P)
python=${DTR_SCE_PYTHON:-/root/miniconda3/bin/python}
session=${DTR_SCE_SESSION:-$default_session}
if [[ "$python" != /* || ! -x "$python" ]]; then
    printf 'Python must be an existing absolute executable: %s\n' "$python" >&2
    exit 2
fi
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    exec "$python" "$script_dir/$entry" "$@"
fi
[[ "$session" =~ ^[a-zA-Z0-9_-]+$ ]] || { printf 'Invalid session name: %s\n' "$session" >&2; exit 2; }
command -v tmux >/dev/null || { printf 'tmux is required.\n' >&2; exit 2; }
if tmux has-session -t "=$session" 2>/dev/null; then
    printf 'Session already exists; no task was sent. View: tmux attach -t %q\n' "$session" >&2
    exit 2
fi
umask 077
log_root=${DTR_SCE_LOG_DIR:-$root/artifacts/dtr_sce/tmux}
mkdir -p -- "$log_root"
log_root=$(cd -- "$log_root" && pwd -P)
job_dir=$(mktemp -d -- "$log_root/$session.XXXXXX")
job=$job_dir/task.sh
{
    printf '#!/usr/bin/env bash\nset -u\nset -o pipefail\n'
    printf 'cd -- %q || exit 2\n' "$root"
    printf 'export YOLO_AUTOINSTALL=false ULTRALYTICS_SAFE_LOAD=true\n'
    printf 'command=('
    printf '%q ' "$python" -u "$script_dir/$entry" "$@"
    printf ')\n'
    printf 'summary_command=('
    printf '%q ' "$python" - "$mode" "$@"
    printf ')\n'
    printf 'log=%q\nexit_file=%q\n' "$job_dir/output.log" "$job_dir/exit_codes.txt"
    cat <<'TASK'
"${command[@]}" 2>&1 | tee "$log"
pipeline_status=("${PIPESTATUS[@]}")
python_rc=${pipeline_status[0]}
tee_rc=${pipeline_status[1]}
printf 'python=%s\ntee=%s\n' "$python_rc" "$tee_rc" > "$exit_file"
"${summary_command[@]}" <<'PY'
import json, sys
from pathlib import Path
from experiments.dtr_sce.train import parser as train_parser
from experiments.dtr_sce.validate import parser as eval_parser
mode = sys.argv[1]
args = (train_parser() if mode == 'train' else eval_parser()).parse_args(sys.argv[2:])
target = Path(args.project).expanduser().resolve() / args.name
if getattr(args, 'dry_run', False):
    print('Dry-run: temporary construction only; formal output was not reserved.')
    print('Audit report:', args.report)
else:
    summary = target / ('dtr_sce_audit.json' if mode == 'train' else 'metrics_exact.json')
    print('Result directory:', target)
    print('Final summary:', summary)
    if summary.is_file():
        data = json.loads(summary.read_text(encoding='utf-8'))
        if mode == 'train':
            print(json.dumps({k: data.get(k) for k in ('formal_training', 'last_epoch', 'final_metrics')}, ensure_ascii=False, indent=2))
        else:
            metrics = data['metrics']
            print(' '.join(f'{k}={metrics[k]:.8f}' for k in ('precision', 'recall', 'f1', 'AP50', 'AP75', 'mAP50_95')))
PY
printf '\nPython exit=%s; tee exit=%s\nLog: %s\nExit record: %s\nTask ended; this shell remains open.\n' "$python_rc" "$tee_rc" "$log" "$exit_file"
if (( python_rc != 0 )); then
    exit "$python_rc"
fi
exit "$tee_rc"
TASK
} > "$job"
# A seconds-fast task failure cannot destroy the interactive parent shell.
tmux new-session -d -s "$session" -c "$root" 'bash --noprofile --norc -i'
tmux set-option -t "=$session" remain-on-exit on
printf -v job_command 'bash %q' "$job"
tmux send-keys -t "=$session" -l "$job_command"
tmux send-keys -t "=$session" C-m
printf 'Started session: %s\nView: tmux attach -t %q\nLog: %s\n' "$session" "$session" "$job_dir/output.log"
