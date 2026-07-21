#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26
NAME=yolo26n_baseline_e200_e150aug_img640_b32_seed42
CFG=$ROOT/tunnel_project/configs/${NAME}.yaml
LOG=$ROOT/logs/train/${NAME}.log

mkdir -p "$ROOT/logs/train"
cd "$ROOT"

yolo detect train cfg="$CFG" 2>&1 | tee "$LOG"
