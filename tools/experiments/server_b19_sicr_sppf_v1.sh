#!/usr/bin/env bash
# Reuse the fixed b19 stage lock and native runner; retain only regular canonical logs.
set -euo pipefail
WORK="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BASE=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26
NAME=yolo26n_b19_sicr_sppf_v1
PROJECT="$WORK/runs/detect"
RUN="$PROJECT/$NAME"
STAGE="${1:-preflight}"
B19_PYTHON="${B19_PYTHON:-/root/miniconda3/bin/python}"
case "$STAGE" in preflight|train|test|diagnose|package|all) ;; *)
    printf 'Usage: bash %s {preflight|train|test|diagnose|package|all}\n' "$0" >&2; exit 2 ;;
esac
exec 9>"$BASE/.${NAME}.lock"
flock -n 9 || { echo 'SICR stage or deployment already active.' >&2; exit 3; }
export PYTHONPATH="$WORK${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_AUTOINSTALL=false YOLO_OFFLINE=true NO_ALBUMENTATIONS_UPDATE=1
cd -- "$WORK"
mkdir -p -- "$PROJECT"
if [[ "$STAGE" == train && -e "$RUN" ]]; then
    printf 'Preserving existing run: %s\n' "$RUN" >&2; exit 3
fi
LOG="$PROJECT/${NAME}_${STAGE}_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
if [[ "$STAGE" == train || "$STAGE" == preflight ]]; then
    COMMAND=("$B19_PYTHON" -u tools/experiments/run_b19_sicr_sppf.py --stage "$STAGE" --baseline-root "$BASE"
        --baseline-args "$BASE/runs/detect/b19_y26n_diverse5x_e200_i640_b32_musgd_b8b9hybrid_s42/args.yaml"
        --pretrained "$BASE/yolo26n.pt" --project "$PROJECT")
else
    COMMAND=("$B19_PYTHON" -u tools/experiments/finish_b19_sicr_sppf.py --stage "$STAGE" --run "$RUN")
fi
printf '%q ' "${COMMAND[@]}" > "$LOG.command.txt"
printf '\n' >> "$LOG.command.txt"
git rev-parse HEAD > "$LOG.commit.txt"
set +e
"${COMMAND[@]}" 2>&1 | tee "$LOG"
codes=("${PIPESTATUS[@]}")
set -e
status="${codes[0]}"
if [[ "${codes[1]}" -ne 0 ]]; then status="${codes[1]}"; fi
printf '{"stage":"%s","python":%s,"tee":%s}\n' "$STAGE" "${codes[0]}" "${codes[1]}" > "$LOG.status.json"
if [[ -d "$RUN" ]]; then
    mkdir -p -- "$RUN/logs"
    cp -- "$LOG" "$LOG.command.txt" "$LOG.commit.txt" "$LOG.status.json" "$RUN/logs/"
fi
exit "$status"
