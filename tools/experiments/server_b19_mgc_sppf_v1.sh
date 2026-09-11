#!/usr/bin/env bash
# Independent MGC v1 worktree entry. Run inside an interactive tmux shell with remain-on-exit enabled.
set -euo pipefail
WORK="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BASE=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26
NAME=yolo26n_b19_mgc_sppf_v1
PROJECT="$WORK/runs/detect"
RUN="$PROJECT/$NAME"
STAGE="${1:-train}"
B19_PYTHON="${B19_PYTHON:-/root/miniconda3/bin/python}"
export PYTHONPATH="$WORK"
export YOLO_AUTOINSTALL=false YOLO_OFFLINE=true NO_ALBUMENTATIONS_UPDATE=1
cd -- "$WORK"
case "$STAGE" in preflight|train|test|diagnose|package) ;; *)
    printf 'Usage: bash %s {preflight|train|test|diagnose|package}\n' "$0" >&2; exit 2 ;;
esac
mkdir -p -- "$PROJECT"
exec 9>"$BASE/.yolo26n_b19_mgc_sppf_v1.lock"
if ! flock -n 9; then
    printf 'A MGC v1 command already holds %s; no duplicate started.\n' "$BASE/.yolo26n_b19_mgc_sppf_v1.lock" >&2
    exit 3
fi
if [[ "$STAGE" == train && -e "$RUN" ]]; then
    printf 'Preserving existing MGC v1 run: %s\n' "$RUN" >&2
    exit 3
fi
ATTEMPT="$(mktemp -d "$PROJECT/${NAME}_${STAGE}.attempt.XXXXXXXX")"
status=1
python_status=null
tee_status=null
finish_attempt() {
    local exit_code=$?
    trap - EXIT
    printf '{"state":"finished","pid":%s,"python":%s,"tee":%s,"exit_code":%s}\n' "$$" "$python_status" "$tee_status" "$exit_code" > "$ATTEMPT/process_status.json"
    printf '%s\n' "$exit_code" > "$ATTEMPT/exit_status"
    date -Is > "$ATTEMPT/finished.txt"
    if [[ "$STAGE" != preflight && -d "$RUN/provenance" ]]; then
        mkdir -p -- "$RUN/logs/${STAGE}_${ATTEMPT##*.}"
        cp -a -- "$ATTEMPT/." "$RUN/logs/${STAGE}_${ATTEMPT##*.}/"
    fi
    printf 'Stage %s exited %s. Preserved attempt: %s\n' "$STAGE" "$exit_code" "$ATTEMPT"
    exit "$exit_code"
}
trap finish_attempt EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
date -Is > "$ATTEMPT/started.txt"
printf '%s\n' "$B19_PYTHON" > "$ATTEMPT/interpreter.txt"
printf '%s\n' "$ATTEMPT" > "$PROJECT/${NAME}_${STAGE}.current_attempt"
printf '%s\n' "$$" > "$ATTEMPT/pid"
printf '%q ' "$BASH" "$WORK/tools/experiments/server_b19_mgc_sppf_v1.sh" "$STAGE" > "$ATTEMPT/command.txt"
printf '{"state":"running","pid":%s,"exit_code":null}\n' "$$" > "$ATTEMPT/process_status.json"
ln -sfn -- "$ATTEMPT/console.log" "$PROJECT/${NAME}_${STAGE}.console.log"
git rev-parse HEAD > "$ATTEMPT/commit.txt"
printf 'experiment=%s stage=%s WORK=%s HEAD=%s interpreter=%s log=%s\n' "$NAME" "$STAGE" "$WORK" "$(cat "$ATTEMPT/commit.txt")" "$B19_PYTHON" "$ATTEMPT/console.log"
printf 'CUDA_VISIBLE_DEVICES=%s (recipe device=0)\n' "${CUDA_VISIBLE_DEVICES:-unset}"
git ls-files -z | xargs -0 sha256sum > "$ATTEMPT/source_sha256.txt"
export B19_ATTEMPT="$ATTEMPT"
# The trace records each expanded Python command and actual interpreter import before execution.
export PS4='+ stage command: '
run_stage() {
    case "$STAGE" in
        preflight|train)
            "$B19_PYTHON" -u "$WORK/tools/experiments/run_b19_mgc_sppf.py" \
                --baseline-root "$BASE" \
                --baseline-args "$BASE/runs/detect/b19_y26n_diverse5x_e200_i640_b32_musgd_b8b9hybrid_s42/args.yaml" \
                --pretrained "$BASE/yolo26n.pt" \
                --pretrained-sha256 9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef \
                --baseline-launcher "$WORK/tools/experiments/b19_launcher_expanded.txt" \
                --stage "$STAGE" --name "$NAME" --project "$PROJECT"
            ;;
        test|diagnose|package)
            "$B19_PYTHON" -u "$WORK/tools/experiments/finish_b19_mgc_sppf.py" --stage "$STAGE" --run "$RUN"
            ;;
    esac
}
set +e
{ set -x; "$B19_PYTHON" -c 'import sys, ultralytics; print(sys.executable); print(ultralytics.__file__)' && run_stage; } 2>&1 | tee "$ATTEMPT/console.log"
codes=("${PIPESTATUS[@]}")
set -e
python_status="${codes[0]}"
tee_status="${codes[1]}"
status="$python_status"
if [[ "$status" -eq 0 && "$tee_status" -ne 0 ]]; then status="$tee_status"; fi
exit "$status"
