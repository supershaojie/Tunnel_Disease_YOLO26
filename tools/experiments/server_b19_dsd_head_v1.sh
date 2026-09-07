#!/usr/bin/env bash
# Independent DSD v1 worktree entry. Run inside an interactive tmux shell with remain-on-exit enabled.
set -euo pipefail
WORK="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BASE=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26
NAME=yolo26n_b19_dsd_head_v1
ENTRY=dsd_head
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
exec 9>"$PROJECT/${NAME}.lock"
if ! flock -n 9; then
    printf 'A DSD v1 command already holds %s; no duplicate started.\n' "$PROJECT/${NAME}.lock" >&2
    exit 3
fi
if [[ "$STAGE" == train && -e "$RUN" ]]; then
    printf 'Preserving existing DSD v1 run: %s\n' "$RUN" >&2
    exit 3
fi
ATTEMPT="$(mktemp -d "$PROJECT/${NAME}_${STAGE}.attempt.XXXXXXXX")"
for suffix in exit_status process_status.json; do
    current="$PROJECT/${NAME}_${STAGE}.$suffix"
    if [[ -f "$current" ]]; then mv -- "$current" "$ATTEMPT/previous.$suffix"; fi
done
printf '%s\n' "$ATTEMPT" > "$PROJECT/${NAME}_${STAGE}.current_attempt"
date -Is > "$ATTEMPT/started.txt"
git -c "safe.directory=$WORK" rev-parse HEAD > "$ATTEMPT/commit.txt"
run_stage() {
    case "$STAGE" in
        preflight|train)
            "$B19_PYTHON" -u "$WORK/tools/experiments/run_b19_${ENTRY}.py" \
                --baseline-root "$BASE" \
                --baseline-args "$BASE/runs/detect/b19_y26n_diverse5x_e200_i640_b32_musgd_b8b9hybrid_s42/args.yaml" \
                --pretrained "$BASE/yolo26n.pt" \
                --pretrained-sha256 9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef \
                --baseline-launcher "$WORK/tools/experiments/b19_launcher_expanded.txt" \
                --stage "$STAGE" --name "$NAME" --project "$PROJECT"
            ;;
        test|diagnose|package)
            "$B19_PYTHON" -u "$WORK/tools/experiments/finish_b19_${ENTRY}.py" --stage "$STAGE" --run "$RUN"
            ;;
    esac
}
set +e
run_stage 2>&1 | tee -a "$PROJECT/${NAME}_${STAGE}.console.log" "$ATTEMPT/console.log"
codes=("${PIPESTATUS[@]}")
set -e
status="${codes[0]}"
if [[ "${codes[1]}" -ne 0 ]]; then status="${codes[1]}"; fi
printf '{"python":%s,"tee":%s}\n' "${codes[0]}" "${codes[1]}" > "$ATTEMPT/process_status.json"
printf '%s\n' "$status" > "$ATTEMPT/exit_status"
date -Is > "$ATTEMPT/finished.txt"
cp -- "$ATTEMPT/process_status.json" "$PROJECT/${NAME}_${STAGE}.process_status.json"
cp -- "$ATTEMPT/exit_status" "$PROJECT/${NAME}_${STAGE}.exit_status"
printf 'Stage %s exited %s. Preserved attempt: %s\n' "$STAGE" "$status" "$ATTEMPT"
exit "$status"
