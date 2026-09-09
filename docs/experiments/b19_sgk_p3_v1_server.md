# SGK-P3 v1 服务器操作

本文件是部署模板；最终交付的 `artifacts/sgk_local/server_commands_<SHA>.md` 已将 `DELIVERY_SHA` 代入真实完整提交。
不需要手工先 preflight 再 train。不会自动 SSH；以下由用户在 AutoDL 终端逐段执行，任何一段失败先处理该段。

## 1. 固定版本并检查现场

```bash
export SGK_SHA=DELIVERY_SHA
export SGK_BRANCH=codex/exp-yolo26n-b19-sgk-p3-v1
export BASE=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26
export WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_SGK_P3_v1
export RUN=yolo26n_b19_sgk_p3_v1
export B19_PYTHON="${B19_PYTHON:-/root/miniconda3/bin/python}"
git -C "$BASE" worktree list
nvidia-smi
pgrep -af 'python.*(run_b19_sgk_p3|finish_b19_sgk_p3)' || true
```

先确认没有本实验正在运行的 Python 进程。其他实验不用停止，不删除任何 runs 或 tmux 会话。

## 2. 获取提交（已有正确提交时不联网）

```bash
(
    set -euo pipefail
    if git -C "$BASE" cat-file -e "${SGK_SHA}^{commit}" 2>/dev/null; then
        printf 'Pinned commit already exists: %s\n' "$SGK_SHA"
    else
        GIT_TERMINAL_PROMPT=0 git -C "$BASE" \
            -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 \
            fetch --progress --no-tags origin "$SGK_BRANCH"
    fi
    git -C "$BASE" cat-file -e "${SGK_SHA}^{commit}"
    git -C "$BASE" show --no-patch --format=fuller "$SGK_SHA"
)
```

此处没有60秒总超时。fetch 的进度会直接显示；失败不进入部署阶段。

## 3. 建立/更新独立工作树

```bash
(
    set -euo pipefail
    if [[ -e "$WORK" ]]; then
        test "$(git -C "$WORK" rev-parse --show-toplevel)" = "$WORK"
        test "$(git -C "$WORK" rev-parse --path-format=absolute --git-common-dir)" = \
             "$(git -C "$BASE" rev-parse --path-format=absolute --git-common-dir)"
        test -z "$(git -C "$WORK" status --porcelain --untracked-files=no)"
        if [[ -e "$WORK/runs/detect/${RUN}.lock" ]]; then
            exec 9>"$WORK/runs/detect/${RUN}.lock"
            flock -n 9
        fi
        "$B19_PYTHON" - "$WORK" <<'PY'
import os, pathlib, sys
work = pathlib.Path(sys.argv[1]).resolve()
for proc in pathlib.Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name) == os.getpid():
        continue
    try:
        command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        cwd = (proc / 'cwd').resolve()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if ('python' in command and (cwd == work or str(work) in command)) or ('server_b19_sgk_p3_v1.sh' in command):
        raise SystemExit(f'Active SGK process: {proc.name} {command}')
PY
        if [[ "$(git -C "$WORK" rev-parse HEAD)" != "$SGK_SHA" ]]; then
            test ! -e "$WORK/runs/detect/$RUN"
            git -C "$WORK" switch --detach "$SGK_SHA"
        fi
    else
        git -C "$BASE" worktree add --detach "$WORK" "$SGK_SHA"
    fi
    test "$(git -C "$WORK" rev-parse HEAD)" = "$SGK_SHA"
    cd "$WORK"
    PYTHONPATH="$WORK" "$B19_PYTHON" -c \
      'import ultralytics, ultralytics.nn.modules.sgk_p3 as s; print(ultralytics.__file__); print(s.__file__)'
)
```

使用 detached HEAD 固定已交付版本，远端来源分支是上文 SGK 专属分支。工作区有修改、活动进程或不同版本的现有正式 run 时停止，保留现场。
不在此处执行 pip install，也不更改任何已有 editable install。

## 4. 独立 tmux 启动一次 train

```bash
(
    set -euo pipefail
    test "$(git -C "$WORK" rev-parse HEAD)" = "$SGK_SHA"
    test ! -e "$WORK/runs/detect/$RUN"
    SESSION="y26_sgk_v1_${SGK_SHA:0:8}_$(date +%Y%m%d_%H%M%S)"
    tmux new-session -d -s "$SESSION" -c "$WORK"
    tmux set-option -t "$SESSION" remain-on-exit on
    tmux send-keys -t "$SESSION" -l "export B19_PYTHON='$B19_PYTHON'; bash tools/experiments/server_b19_sgk_p3_v1.sh train"
    tmux send-keys -t "$SESSION" Enter
    printf 'SGK tmux session: %s\n' "$SESSION"
)
```

`train` 自动启动独立预检子进程，通过后才从原始权重开始全新正式训练。预检失败保留完整记录，不能降低 batch 或关闭 AMP。
只在单独排查时使用 `bash "$WORK/tools/experiments/server_b19_sgk_p3_v1.sh" preflight`；正常启动不需要此命令。
同名旧闲置 tmux 会话不表示正在训练，应看下面的当前 attempt/PID/退出码。

## 5. 查看当前 attempt、进程与日志

```bash
STAGE=train
ATTEMPT="$(cat "$WORK/runs/detect/${RUN}_${STAGE}.current_attempt")"
printf 'Current attempt: %s\n' "$ATTEMPT"
cat "$ATTEMPT/started.txt" "$ATTEMPT/commit.txt" "$ATTEMPT/command.txt"
cat "$ATTEMPT/process_status.json"
ps -p "$(cat "$ATTEMPT/pid")" -o pid,etime,args
if [[ -f "$ATTEMPT/exit_status" ]]; then cat "$ATTEMPT/exit_status"; fi
tail -n 80 "$ATTEMPT/console.log"
```

无退出码且 PID 已消失表示运行被中断，需要检查日志，不能当作成功。
预检详细记录在 `$WORK/runs/detect/${RUN}_preflight/`。

```bash
CHECK_ROOT="$WORK/runs/detect/${RUN}_preflight"
INVOCATION="$(cat "$CHECK_ROOT/preflight.current_invocation")"
cat "$INVOCATION/process_status.json"
cat "$INVOCATION/checks.json"
```

预检 checks 会指向实际 report_dir，其中记录最多128批的任务梯度、真实成功 step、AMP scale/跳步、参数更新和失败阶段。
正式训练日志同时保存在 `$WORK/runs/detect/$RUN/train.log`。

## 6. 训练完成后逐段收尾

先确认当前 train attempt `exit_status=0`，且正式目录有 `completed.json` 与 `weights/best.pt`。

```bash
bash "$WORK/tools/experiments/server_b19_sgk_p3_v1.sh" test
```

test 完成全量独立 FP32 val/test，并冻结 val 的 R@P≥.85 阈值后报告 test；若原生 b19 best 可访问，还会用相同入口生成 baseline 对比。
该阶段失败时不执行后续阶段。

```bash
bash "$WORK/tools/experiments/server_b19_sgk_p3_v1.sh" diagnose
```

诊断失败时先查看 `diagnose.current_attempt` 及日志，不绕过。

```bash
bash "$WORK/tools/experiments/server_b19_sgk_p3_v1.sh" package
```

缺少必要阶段或身份不匹配会拒绝打包。每个阶段都可将第5节的 STAGE 换成 `test`、`diagnose`、`package` 来查看当前状态。

## 7. 核验并下载结果包

```bash
(
    set -euo pipefail
    SHORT="$(git -C "$WORK" rev-parse --short=12 HEAD)"
    cd "$WORK/artifacts/experiments"
    PACKAGE="${RUN}_${SHORT}.tar.gz"
    sha256sum -c "${PACKAGE}.sha256"
    gzip -t "$PACKAGE"
    ls -lh "$PACKAGE" "${PACKAGE}.sha256"
    realpath "$PACKAGE"
)
```

下载目录固定为 `/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_SGK_P3_v1/artifacts/experiments/`。
实际包的绝对路径、大小和 SHA256 由成功的 package 阶段打印；当前尚未在服务器生成包。
下载 `.tar.gz` 与 `.tar.gz.sha256` 后，本地也应核对哈希。原始数据、整个 Conda、`.git` 与其他实验结果不会进入包。
