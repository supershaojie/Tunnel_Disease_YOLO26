#!/usr/bin/env bash
# Isolated b19 SIR-SPPF entry; all source paths follow this script's actual worktree.
set -euo pipefail
WORK="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BASE=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26
NAME=yolo26n_b19_d1_sir_sppf_v1
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
    printf 'An SIR command already holds %s; no second command started.\n' "$PROJECT/${NAME}.lock" >&2
    exit 3
fi
if [[ "$STAGE" == train && -e "$RUN" ]]; then
    printf 'SIR run already exists: %s\n' "$RUN" >&2
    [[ ! -f "$PROJECT/${NAME}_train.exit_status" ]] || cat "$PROJECT/${NAME}_train.exit_status" >&2
    exit 3
fi
case "$STAGE" in
    preflight|train)
        # The Python runner restores the attributed user snapshot if the historical file is absent.
        set +e
        "$B19_PYTHON" -u "$WORK/tools/experiments/run_b19_sir_sppf.py" \
            --baseline-root "$BASE" \
            --baseline-args "$BASE/runs/detect/b19_y26n_diverse5x_e200_i640_b32_musgd_b8b9hybrid_s42/args.yaml" \
            --pretrained "$BASE/yolo26n.pt" \
            --pretrained-sha256 9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef \
            --baseline-launcher /root/autodl-tmp/experiment_backups/records/b19_launcher_expanded.txt \
            --stage "$STAGE" --name "$NAME" --project "$PROJECT" \
            2>&1 | tee -a "$PROJECT/${NAME}_${STAGE}.console.log"
        codes=("${PIPESTATUS[@]}")
        set -e
        status="${codes[0]}"
        if [[ "${codes[1]}" -ne 0 ]]; then status="${codes[1]}"; fi
        printf '{"python":%s,"tee":%s}\n' "${codes[0]}" "${codes[1]}" > "$PROJECT/${NAME}_${STAGE}.process_status.json"
        printf '%s\n' "$status" > "$PROJECT/${NAME}_${STAGE}.exit_status"
        exit "$status"
        ;;
    test|diagnose|package)
        "$B19_PYTHON" -u "$WORK/tools/experiments/finish_b19_sir_sppf.py" --stage "$STAGE" --run "$RUN"
        ;;
esac
