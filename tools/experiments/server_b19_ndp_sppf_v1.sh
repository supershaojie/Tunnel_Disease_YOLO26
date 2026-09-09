#!/usr/bin/env bash
# Only this experiment is locked; all other experiments retain their processes and output.
set -euo pipefail
WORK="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BASE=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26
NAME=yolo26n_b19_ndp_sppf_v1
PROJECT="$WORK/runs/detect"
RUN="$PROJECT/$NAME"
STAGE="${1:-train}"
B19_PYTHON="${B19_PYTHON:-/root/miniconda3/bin/python}"
export PYTHONPATH="$WORK${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_AUTOINSTALL=false YOLO_OFFLINE=true NO_ALBUMENTATIONS_UPDATE=1
cd -- "$WORK"
case "$STAGE" in preflight|train|test|diagnose|package) ;; *)
    printf 'Usage: bash %s {preflight|train|test|diagnose|package}\n' "$0" >&2; exit 2 ;;
esac
mkdir -p -- "$PROJECT"
mkdir -p -- "$BASE/.experiment-locks"
exec 9>"$BASE/.experiment-locks/${NAME}.lock"
if ! flock -n 9; then
    printf 'NDP is already running; no duplicate started: %s\n' "$BASE/.experiment-locks/${NAME}.lock" >&2
    exit 3
fi
if [[ "$STAGE" == train && -e "$RUN" ]]; then
    printf 'No new attempt started; preserving existing NDP run: %s\n' "$RUN" >&2
    exit 3
fi
ATTEMPT="$(mktemp -d "$PROJECT/${NAME}_${STAGE}.attempt.XXXXXXXX")"
# A current exit_status exists only after the current attempt exits.
for suffix in exit_status process_status.json; do
    current="$PROJECT/${NAME}_${STAGE}.$suffix"
    if [[ -f "$current" ]]; then mv -- "$current" "$ATTEMPT/previous.$suffix"; fi
done
printf '%s\n' "$ATTEMPT" > "$PROJECT/${NAME}_${STAGE}.current_attempt"
printf '%s\n' "$$" > "$ATTEMPT/shell.pid"
date -Is > "$ATTEMPT/started.txt"
git -c "safe.directory=$WORK" rev-parse HEAD > "$ATTEMPT/commit.txt"
printf '{"state":"running","shell_pid":%s}\n' "$$" > "$ATTEMPT/process_status.json"
cp -- "$ATTEMPT/process_status.json" "$PROJECT/${NAME}_${STAGE}.process_status.json"
finish() {
    local status=$?
    trap - EXIT
    printf '%s\n' "$status" > "$ATTEMPT/exit_status"
    printf '{"state":"exited","shell_pid":%s,"exit_status":%s}\n' "$$" "$status" > "$ATTEMPT/process_status.json"
    date -Is > "$ATTEMPT/finished.txt"
    cp -- "$ATTEMPT/process_status.json" "$PROJECT/${NAME}_${STAGE}.process_status.json"
    cp -- "$ATTEMPT/exit_status" "$PROJECT/${NAME}_${STAGE}.exit_status"
    printf 'Stage %s exited %s. Attempt: %s\n' "$STAGE" "$status" "$ATTEMPT"
    exit "$status"
}
trap finish EXIT
python_pid= tee_pid= signal_status=
forward_signal() {
    local signal="$1"
    signal_status="$2"
    if [[ -n "$python_pid" ]]; then kill -s "$signal" -- "-$python_pid" 2>/dev/null || true; fi
}
trap 'forward_signal INT 130' INT
trap 'forward_signal TERM 143' TERM
COMMAND=("$B19_PYTHON" -u)
case "$STAGE" in
    preflight|train)
        COMMAND+=("$WORK/tools/experiments/run_b19_ndp_sppf_v1.py"
            --baseline-root "$BASE"
            --baseline-args "$BASE/runs/detect/b19_y26n_diverse5x_e200_i640_b32_musgd_b8b9hybrid_s42/args.yaml"
            --pretrained "$BASE/yolo26n.pt"
            --pretrained-sha256 9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef
            --baseline-launcher "$WORK/tools/experiments/b19_launcher_expanded.txt"
            --stage "$STAGE" --name "$NAME" --project "$PROJECT")
        ;;
    test|diagnose|package)
        COMMAND+=("$WORK/tools/experiments/finish_b19_ndp_sppf_v1.py" --stage "$STAGE")
        ;;
esac
printf '%q ' "${COMMAND[@]}" > "$ATTEMPT/command.txt"
printf '\n' >> "$ATTEMPT/command.txt"
export B19_STAGE_ATTEMPT="$ATTEMPT" B19_STAGE="$STAGE"
mkfifo "$ATTEMPT/console.pipe"
tee -a "$PROJECT/${NAME}_${STAGE}.console.log" "$ATTEMPT/console.log" < "$ATTEMPT/console.pipe" &
tee_pid=$!
# All descendants share this new process group; signals reach preflight and formal subprocesses too.
setsid --wait "${COMMAND[@]}" > "$ATTEMPT/console.pipe" 2>&1 &
python_pid=$!
printf '%s\n' "$python_pid" > "$ATTEMPT/supervisor.pid"
set +e
wait "$python_pid"
python_status=$?
if [[ -n "$signal_status" ]]; then
    wait "$python_pid"
    python_status="$signal_status"
fi
wait "$tee_pid"
tee_status=$?
set -e
printf '{"python":%s,"tee":%s}\n' "$python_status" "$tee_status" > "$ATTEMPT/pipeline_status.json"
status="$python_status"
if [[ "$tee_status" -ne 0 ]]; then status="$tee_status"; fi
exit "$status"
