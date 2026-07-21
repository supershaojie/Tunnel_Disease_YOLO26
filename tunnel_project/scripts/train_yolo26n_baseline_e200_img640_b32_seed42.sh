#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/root/autodl-tmp/projects/Tunnel_Disease_YOLO26"
CONFIG="$ROOT/tunnel_project/configs/yolo26n_baseline_e200_img640_b32_seed42.yaml"
TRAINER="$ROOT/tunnel_project/scripts/train_yolo26n_baseline_e200_img640_b32_seed42.py"
DATA_YAML="$ROOT/datasets/Tunnel_Crack_AugFirst_5x/data_server.yaml"
DATASET_ROOT="$ROOT/datasets/Tunnel_Crack_AugFirst_5x"
MODEL="$ROOT/yolo26n.pt"

RUN_DIR="$ROOT/runs/tunnel_crack/yolo26n_baseline_e200_img640_b32_seed42"
LOG="$ROOT/logs/train/yolo26n_baseline_e200_img640_b32_seed42.log"
RECORDS="$ROOT/logs/train/yolo26n_baseline_e200_img640_b32_seed42_records"

EXPECTED_BRANCH="exp-yolo26n-baseline-e200-crack"

cd "$ROOT"

[[ "$(git branch --show-current)" == "$EXPECTED_BRANCH" ]] || {
    echo "错误：当前分支不是 $EXPECTED_BRANCH"
    exit 1
}

[[ -z "$(git status --porcelain)" ]] || {
    echo "错误：Git工作区不干净。"
    git status -sb
    exit 1
}

for path in "$CONFIG" "$TRAINER" "$DATA_YAML" "$MODEL"; do
    [[ -f "$path" ]] || {
        echo "错误：缺少文件 $path"
        exit 1
    }
done

[[ ! -e "$RUN_DIR" ]] || {
    echo "错误：结果目录已经存在：$RUN_DIR"
    exit 1
}

[[ ! -e "$LOG" ]] || {
    echo "错误：日志已经存在：$LOG"
    exit 1
}

[[ ! -e "$RECORDS" ]] || {
    echo "错误：复现记录已经存在：$RECORDS"
    exit 1
}

export YOLO_CONFIG_DIR="/root/autodl-tmp/.config/Ultralytics"

mkdir -p "$YOLO_CONFIG_DIR"
mkdir -p "$(dirname "$LOG")"
mkdir -p "$RECORDS"

date -Is > "$RECORDS/start_time.txt"
git branch --show-current > "$RECORDS/git_branch.txt"
git rev-parse HEAD > "$RECORDS/git_commit.txt"
git status -sb > "$RECORDS/git_status_before.txt"
git log -5 --oneline > "$RECORDS/git_log.txt"

cp "$CONFIG" "$RECORDS/config_at_launch.yaml"
cp "$TRAINER" "$RECORDS/trainer_at_launch.py"
cp "$DATA_YAML" "$RECORDS/data_server_at_launch.yaml"

python - <<'PYENV' > "$RECORDS/python_environment.txt"
import platform
import sys
import torch
import ultralytics

print("platform:", platform.platform())
print("python:", sys.version)
print("python_executable:", sys.executable)
print("ultralytics:", ultralytics.__version__)
print("ultralytics_path:", ultralytics.__file__)
print("torch:", torch.__version__)
print("cuda_runtime:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PYENV

python -m pip freeze > "$RECORDS/pip_freeze.txt"
nvidia-smi > "$RECORDS/nvidia_smi_before.txt"

sha256sum "$CONFIG" "$TRAINER" "$DATA_YAML" "$MODEL" \
    > "$RECORDS/input_sha256.txt"

find "$DATASET_ROOT" -type f -printf '%P\t%s\n' |
    LC_ALL=C sort > "$RECORDS/dataset_inventory.tsv"

sha256sum "$RECORDS/dataset_inventory.tsv" \
    > "$RECORDS/dataset_inventory_sha256.txt"

printf '%s\n' \
    "python -u $TRAINER" > "$RECORDS/command.txt"

echo "正式训练开始：$(date -Is)"
echo "结果目录：$RUN_DIR"
echo "训练日志：$LOG"

set +e
python -u "$TRAINER" 2>&1 | tee "$LOG"
pipeline_status=("${PIPESTATUS[@]}")
set -e

train_status="${pipeline_status[0]}"
tee_status="${pipeline_status[1]:-1}"

{
    echo "train_exit_code=$train_status"
    echo "tee_exit_code=$tee_status"
} > "$RECORDS/exit_codes.txt"

date -Is > "$RECORDS/end_time.txt"
nvidia-smi > "$RECORDS/nvidia_smi_after.txt"

if [[ -d "$RUN_DIR" ]]; then
    find "$RUN_DIR" -maxdepth 2 -type f -printf '%P\t%s\n' |
        LC_ALL=C sort > "$RECORDS/result_inventory.tsv"
fi

if [[ "$train_status" -ne 0 || "$tee_status" -ne 0 ]]; then
    echo "训练或日志保存异常，退出码已经记录。"
    exit 1
fi

echo "训练正常完成：$(date -Is)"
