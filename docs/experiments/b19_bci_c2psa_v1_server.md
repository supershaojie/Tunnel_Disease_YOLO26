# BCI-C2PSA v1 服务器操作

以下步骤在 AutoDL 终端人工执行，本次 Codex 未 SSH 或启动服务器训练。
按小节逐步执行，上一小节失败时停止；保留同一终端中的变量。
本文件的 `__DELIVERY_SHA__` 由交付时的 `artifacts/delivery/BCI_SERVER_DEPLOY.md` 替换为已推送并核对的完整 SHA；实际部署请复制该交付文件。

## 1. 设定身份并只读检查

```bash
set -euo pipefail
export BASE=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26
export WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_BCI_C2PSA_v1
export BRANCH=codex/exp-yolo26n-b19-bci-c2psa-v1
export SHA=__DELIVERY_SHA__
export NAME=yolo26n_b19_bci_c2psa_v1
export PROJECT="$WORK/runs/detect"
export B19_PYTHON=/root/miniconda3/bin/python
git -C "$BASE" worktree list
git -C "$BASE" status --short
tmux list-sessions || true
nvidia-smi
if pgrep -af 'python.*(run_b19_bci_c2psa|finish_b19_bci_c2psa)'; then
    printf 'BCI Python process exists; stop deployment and inspect its current attempt.\n' >&2
    exit 1
fi
if [[ -d "$WORK" ]]; then
    test "$(git -C "$WORK" rev-parse --show-toplevel)" = "$WORK"
    test "$(realpath "$(git -C "$WORK" rev-parse --path-format=absolute --git-common-dir)")" = "$(realpath "$BASE/.git")"
    git -C "$WORK" status --short
    git -C "$WORK" diff --quiet
    git -C "$WORK" diff --cached --quiet
    if [[ -e "$PROJECT/$NAME" && "$(git -C "$WORK" rev-parse HEAD)" != "$SHA" ]]; then
        printf 'Existing formal run belongs to another HEAD; preserving code/results.\n' >&2
        exit 1
    fi
    mkdir -p "$PROJECT"
    flock -n "$PROJECT/${NAME}.lock" true
fi
```

检查 GPU 上其他实验占用，不停止它们。已有正确 HEAD 时下步不会重复联网。
正式入口会核验 Python/PyTorch/GPU 与 b19 的记录，环境冲突会保留诊断并停止，不自动升级环境。

## 2. 仅在缺少指定提交时 fetch

```bash
if ! git -C "$BASE" cat-file -e "$SHA^{commit}" 2>/dev/null; then
    GIT_TERMINAL_PROMPT=0 git -C "$BASE" \
        -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 \
        fetch --progress origin "refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
fi
git -C "$BASE" cat-file -e "$SHA^{commit}"
git -C "$BASE" show -s --format='%H %s' "$SHA"
```

没有额外 60 秒总超时；fetch 失败即停止。不得 reset、clean、强推或删除 runs。

## 3. 创建或切换独立 worktree

```bash
if [[ ! -e "$WORK" ]]; then
    git -C "$BASE" worktree add --detach "$WORK" "$SHA"
elif [[ "$(git -C "$WORK" rev-parse HEAD)" != "$SHA" ]]; then
    git -C "$WORK" switch --detach "$SHA"
fi
test "$(git -C "$WORK" rev-parse HEAD)" = "$SHA"
test -f "$BASE/runs/detect/b19_y26n_diverse5x_e200_i640_b32_musgd_b8b9hybrid_s42/args.yaml"
test -f /root/autodl-tmp/experiment_backups/records/b19_launcher_expanded.txt
test -f "$BASE/yolo26n.pt"
cd "$WORK"
PYTHONPATH="$WORK" "$B19_PYTHON" -c 'import ultralytics; import ultralytics.nn.modules.bci_c2psa as m; print(ultralytics.__file__); print(m.__file__)'
```

两个 import 路径必须位于 BCI 工作树。源代码有未提交变化时入口拒绝运行。
若需要更新到另一个交付 SHA，先重新执行第 1 步检查；已有正式 run 时不能将它与新代码混用。

## 4. 只启动一次 train（内部自动独立预检）

```bash
test ! -e "$PROJECT/$NAME"
SESSION="y26_bci_v1_train_$(date +%Y%m%d_%H%M%S)_$$"
tmux new-session -d -s "$SESSION" \
    "cd '$WORK'; export B19_PYTHON='$B19_PYTHON'; bash tools/experiments/server_b19_bci_c2psa_v1.sh train; rc=\$?; printf '\ntrain exit=%s; window retained\n' \"\$rc\"; exec bash"
printf 'Session: %s\n' "$SESSION"
tmux attach -t "$SESSION"
```

`Ctrl-b d` 退出查看但保留进程；会话在失败后也保留。会话存在不等于训练仍在运行，以当前 attempt 和进程状态为准。
正常只运行一次 train，不手工拼 `preflight && train`。需要单独排障时才运行 `bash tools/experiments/server_b19_bci_c2psa_v1.sh preflight`。
预检最多 128 个真实 batch，不降 batch、不关 AMP、不调高 LR；失败保留全部 attempt。已有正式 run 时拒绝覆盖与静默 resume。

## 5. 查看当前 attempt、日志和退出码

```bash
STAGE=train  # 也可设 preflight、test、diagnose、package
ATTEMPT="$(cat "$PROJECT/${NAME}_${STAGE}.current_attempt")"
printf 'Current attempt: %s\n' "$ATTEMPT"
cat "$ATTEMPT/process_status.json"
cat "$ATTEMPT/commit.txt" 2>/dev/null || true
cat "$ATTEMPT/python.pid" 2>/dev/null || true
find "$ATTEMPT" -maxdepth 1 -name 'process_*.json' -exec cat {} \;
if [[ -f "$ATTEMPT/python.pid" ]]; then
    ps -p "$(cat "$ATTEMPT/python.pid")" -o pid,ppid,lstart,etime,args || true
fi
if [[ -f "$ATTEMPT/exit_status" ]]; then
    printf 'Recorded exit code: '; cat "$ATTEMPT/exit_status"
else
    printf 'No exit code yet; inspect process_status, PID and log together.\n'
fi
tail -n 80 "$ATTEMPT/console.log"
```

实时查看用 `tail -f "$ATTEMPT/console.log"`，Ctrl-C 只停止 tail。
自动 preflight 的当前目录同样由 `${NAME}_preflight.current_attempt` 指向；`audit_path.txt` 给出 checks/resolved/failed 证据目录。
正式成功需退出码 0、`$PROJECT/$NAME/completed.json` 及对应 best.pt；不要用旧 console.log 中的成功文字判断新 attempt。

## 6. 训练成功后 test → diagnose → package

分别执行下面三段，每段返回 0 后才执行下一段。长评估也可以使用新的 `y26_bci_v1_test_<时间>` tmux 会话运行。

```bash
test "$(cat "$PROJECT/${NAME}_train.exit_status")" = 0
cd "$WORK"
bash tools/experiments/server_b19_bci_c2psa_v1.sh test
```

```bash
test "$(cat "$PROJECT/${NAME}_test.exit_status")" = 0
bash tools/experiments/server_b19_bci_c2psa_v1.sh diagnose
```

```bash
test "$(cat "$PROJECT/${NAME}_diagnose.exit_status")" = 0
bash tools/experiments/server_b19_bci_c2psa_v1.sh package
```

test 对固定 best.pt 依次运行独立 FP32 val/test；仅 val 选 Recall 补充阈值并冻结给 test。
若基础目录存在 b19 best.pt，补齐相同口径的基线评估；没有基线权重时记录 unavailable，不重训。
diagnose 的 16 张图和通道统计不替代完整评估。主筛选看 val mAP50–95，同时检查 AP50 与同 Precision/FPPI 的 Recall。

## 7. 校验与下载

```bash
SHORT_SHA="$(git -C "$WORK" rev-parse --short=12 HEAD)"
PACKAGE="$WORK/artifacts/experiments/${NAME}_${SHORT_SHA}.tar.gz"
test -f "$PACKAGE"
test -f "$PACKAGE.sha256"
cd "$WORK/artifacts/experiments"
sha256sum -c "$(basename "$PACKAGE.sha256")"
gzip -t "$PACKAGE"
tar -tzf "$PACKAGE" > "$PACKAGE.contents.txt"
stat -c '%n %s bytes' "$PACKAGE"
sha256sum "$PACKAGE"
```

实际下载目录：`/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_BCI_C2PSA_v1/artifacts/experiments/`。
下载 `.tar.gz` 与 `.tar.gz.sha256`；入口同时打印真实绝对路径、大小和 SHA256。
缺少成功阶段、图数/实例数不符或任何身份/哈希不一致时 package 会报错，不会生成伪完成包。
