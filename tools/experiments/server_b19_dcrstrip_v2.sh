#!/usr/bin/env bash
# Run in the existing b19 Python environment; do not change its dependencies.
set -euo pipefail
BASE=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_DCRStrip_v2
NAME=yolo26n_b19_a1_dcrstrip_v2
RUN="$WORK/runs/detect/$NAME"
STAGE="${1:-train}"
B19_PYTHON="${B19_PYTHON:-python}"
cd "$WORK"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_AUTOINSTALL=false YOLO_OFFLINE=true

case "$STAGE" in
    preflight|train)
        LAUNCHER=/root/autodl-tmp/experiment_backups/records/b19_launcher_expanded.txt
        if [ ! -e "$LAUNCHER" ]; then
            mkdir -p "$(dirname "$LAUNCHER")"
            # User-provided historical record, with its source attribution retained.
            cp -n tools/experiments/b19_launcher_expanded.txt "$LAUNCHER"
        fi
        "$B19_PYTHON" -c 'import sys, torch, ultralytics; from pathlib import Path; assert Path(ultralytics.__file__).resolve().parent == Path.cwd()/"ultralytics"; print(sys.executable, sys.version, ultralytics.__file__, ultralytics.__version__, torch.__version__, torch.version.cuda)'
        mkdir -p "$WORK/runs/detect"
        if "$B19_PYTHON" -u tools/experiments/run_b19_dcrstrip_v2.py \
            --baseline-root "$BASE" \
            --baseline-args "$BASE/runs/detect/b19_y26n_diverse5x_e200_i640_b32_musgd_b8b9hybrid_s42/args.yaml" \
            --pretrained "$BASE/yolo26n.pt" \
            --pretrained-sha256 9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef \
            --baseline-launcher "$LAUNCHER" \
            --stage "$STAGE" --name "$NAME" 2>&1 | tee -a "$WORK/runs/detect/${NAME}_${STAGE}.console.log"; then
            status=0
        else
            status=$?
        fi
        printf '%s\n' "$status" > "$WORK/runs/detect/${NAME}_${STAGE}.exit_status"
        exit "$status"
        ;;
    test|diagnose|package)
        "$B19_PYTHON" -u tools/experiments/finish_b19_dcrstrip_v2.py --stage "$STAGE" --run "$RUN"
        ;;
    *)
        printf 'Usage: bash %s {preflight|train|test|diagnose|package}\n' "$0" >&2
        exit 2
        ;;
esac
