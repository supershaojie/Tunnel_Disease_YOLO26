#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 2 ]; then
    printf 'Usage: bash launch_tmux.sh SESSION train|evaluate [entry arguments...]\n' >&2
    exit 2
fi
session="$1"
entry="$2"
shift 2
case "$session" in
    *[!a-zA-Z0-9_-]*|'') printf 'Use a simple explicit tmux session name.\n' >&2; exit 2 ;;
esac
case "$entry" in
    train|evaluate) ;;
    *) printf 'Entry must be train or evaluate.\n' >&2; exit 2 ;;
esac
if tmux has-session -t "=$session" 2>/dev/null; then
    tmux display-message -p -t "=$session" '#{session_name} #{pane_current_command} #{pane_dead}' >&2
    printf 'Session already exists; use an explicitly different name.\n' >&2
    exit 1
fi
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
python_executable="${COMPARISON_PYTHON:-$(command -v python)}"
printf -v command '%q ' "$python_executable" -u "$root/experiments/comparison/yolov5n/$entry.py" "$@"
printf -v shell_command 'tmux set-option -t %q remain-on-exit on && cd %q && %s; result=$?; printf "Python exit code: %%s\\n" "$result"; exit "$result"' "=$session" "$root" "$command"
tmux new-session -d -s "$session" "bash -lc $(printf '%q' "$shell_command")"
printf 'Started independent session %s. Inspect: tmux attach -t %s\n' "$session" "$session"
