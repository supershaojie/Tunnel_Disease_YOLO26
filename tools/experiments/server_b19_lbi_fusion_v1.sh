#!/usr/bin/env bash
# Reuse native b19 offline entry, per-attempt logs and a lock shared with deployment.
set -euo pipefail
WORK="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BASE=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26
NAME=yolo26n_b19_lbi_fusion_v1
PROJECT="$WORK/runs/detect"
RUN="$PROJECT/$NAME"
STAGE="${1:-preflight}"
B19_PYTHON="${B19_PYTHON:-/root/miniconda3/bin/python}"
case "$STAGE" in preflight|train|test|diagnose|package) ;; *)
    printf 'Usage: bash %s {preflight|train|test|diagnose|package}\n' "$0" >&2; exit 2 ;;
esac
cd -- "$WORK"
mkdir -p -- "$PROJECT"
ATTEMPT="$(mktemp -d "$PROJECT/.${NAME}.${STAGE}.attempt.XXXXXX")"
LOG="$ATTEMPT/console.log"
HEAD="$(git rev-parse HEAD)"
printf 'experiment=%s stage=%s\nWORK=%s\nHEAD=%s\npython=%s\nlog=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
    "$NAME" "$STAGE" "$WORK" "$HEAD" "$B19_PYTHON" "$LOG" "${CUDA_VISIBLE_DEVICES-<unset>}" | tee "$LOG"
signal=none
finish() {
    local status="$1"
    printf '{"stage":"%s","exit_code":%s,"signal":"%s","head":"%s"}\n' \
        "$STAGE" "$status" "$signal" "$HEAD" > "$ATTEMPT/status.json"
}
trap 'finish "$?"' EXIT
trap 'signal=INT; printf "Interrupted: SIGINT\n" >> "$LOG"; exit 130' INT
trap 'signal=TERM; printf "Interrupted: SIGTERM\n" >> "$LOG"; exit 143' TERM
exec 9>"$BASE/.${NAME}.lock"
flock -n 9 || { echo 'LBI stage or deployment already active.' | tee -a "$LOG" >&2; exit 3; }
export PYTHONPATH="$WORK${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_AUTOINSTALL=false YOLO_OFFLINE=true NO_ALBUMENTATIONS_UPDATE=1
if [[ "$STAGE" == train && -e "$RUN" ]]; then
    printf 'Preserving existing formal run: %s\n' "$RUN" | tee -a "$LOG" >&2; exit 3
fi
if [[ "$STAGE" == train || "$STAGE" == preflight ]]; then
    COMMAND=("$B19_PYTHON" -u tools/experiments/run_b19_lbi_fusion.py --stage "$STAGE" --baseline-root "$BASE"
        --baseline-args "$BASE/runs/detect/b19_y26n_diverse5x_e200_i640_b32_musgd_b8b9hybrid_s42/args.yaml"
        --pretrained "$BASE/yolo26n.pt" --project "$PROJECT")
else
    COMMAND=("$B19_PYTHON" -u tools/experiments/finish_b19_lbi_fusion.py --stage "$STAGE" --run "$RUN")
fi
printf '%q ' "${COMMAND[@]}" > "$ATTEMPT/command.txt"
printf '\n' >> "$ATTEMPT/command.txt"
set +e
"${COMMAND[@]}" 2>&1 | tee -a "$LOG"
codes=("${PIPESTATUS[@]}")
set -e
printf '{"python":%s,"tee":%s}\n' "${codes[0]}" "${codes[1]}" > "$ATTEMPT/pipeline.json"
status="${codes[0]}"
if [[ "$status" -eq 0 && "${codes[1]}" -ne 0 ]]; then status="${codes[1]}"; fi
finish "$status"
# Only a launched command owns canonical log publication; rejected locks/existing runs stay untouched.
if [[ -d "$RUN" && "$STAGE" != preflight ]]; then
    RECORD="$RUN/logs/$(basename "$ATTEMPT" | sed 's/\.attempt\./.record./')"
    mkdir -p -- "$RECORD"
    cp -- "$LOG" "$ATTEMPT/status.json" "$ATTEMPT/command.txt" "$ATTEMPT/pipeline.json" "$RECORD/" || {
        if [[ "$status" -eq 0 ]]; then status=1; fi
    }
fi
exit "$status"
