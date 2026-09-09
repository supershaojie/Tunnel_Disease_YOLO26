# NDP-SPPF v1 服务器操作

这里不会代用户 SSH 或启动长时训练。使用交付时另给的固定 full SHA 部署代码块；仓库内只通过 `git rev-parse HEAD` 读取当前版本，避免文档包含自身最终 SHA 的循环。不要把移动分支名当最终部署版本。

## 固定版本部署

交付附带的固定 SHA 引导脚本会核对 origin、获取共享实验锁；本地已有目标提交时跳过联网，否则仅 fetch 本实验分支。网络参数 HTTP/1.1、lowSpeedLimit1024、lowSpeedTime120；120 秒是低速持续判定，并非 fetch 总超时。失败退出码保留。

仓库中的完整实现为 `tools/experiments/deploy_b19_ndp_sppf_v1.sh FULL_SHA`。若目标 WORK 已存在且不是指定 SHA，或有修改，它会展示实际状态并停止，不静默 checkout/reset；不会触碰其他 worktree。

路径固定：

- BASE：`/root/autodl-tmp/projects/Tunnel_Disease_YOLO26`
- WORK：`/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_NDP_SPPF_v1`
- BRANCH：`codex/exp-yolo26n-b19-ndp-sppf-v1`
- RUN：`$WORK/runs/detect/yolo26n_b19_ndp_sppf_v1`
- Python：`/root/miniconda3/bin/python`，必要时用 `B19_PYTHON` 显式指定同环境解释器。

部署完成后，以下各块自行定义变量，可独立复制执行。

## 正常启动一次 train

`train` 自动执行独立 preflight，成功后再启动全新正式训练进程。不需要先手动 preflight。与 b19 环境或完整配置冲突、OOM、无有效任务更新等情况都会失败保留现场，不能通过改 batch/训练参数放行。

```bash
set -euo pipefail
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_NDP_SPPF_v1
SESSION=ndp_sppf_v1_train
B19_PYTHON=${B19_PYTHON:-/root/miniconda3/bin/python}
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "会话已存在，未启动重复训练：$SESSION" >&2
    exit 3
fi
# 先建立可保留的会话，再送入命令，避免进程立即失败时窗口消失。
tmux new-session -d -s "$SESSION" -c "$WORK"
tmux set-option -t "$SESSION" remain-on-exit on
printf -v CMD 'cd %q && B19_PYTHON=%q bash tools/experiments/server_b19_ndp_sppf_v1.sh train' "$WORK" "$B19_PYTHON"
tmux send-keys -t "$SESSION" "$CMD" C-m
tmux attach-session -t "$SESSION"
```

离开窗口但保持训练：先按 Ctrl+b，松开后按 d。失败后窗口和日志保留，不要删除 run 或自动 resume 重试。

## 查看当前记录和进程

```bash
set -euo pipefail
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_NDP_SPPF_v1
PROJECT="$WORK/runs/detect"
NAME=yolo26n_b19_ndp_sppf_v1
for STAGE in train preflight; do
    POINTER="$PROJECT/${NAME}_${STAGE}.current_attempt"
    if [[ -f "$POINTER" ]]; then
        ATTEMPT="$(cat "$POINTER")"
        printf '\n%s: %s\n' "$STAGE" "$ATTEMPT"
        if [[ -f "$ATTEMPT/process_status.json" ]]; then cat "$ATTEMPT/process_status.json"; fi
        if [[ -f "$ATTEMPT/exit_status" ]]; then
            printf 'exit_status='; cat "$ATTEMPT/exit_status"
        else
            echo '尚未记录退出码；请结合进程判断，不能据此宣称正常运行。'
        fi
        if [[ -f "$ATTEMPT/python.pid" ]]; then ps -fp "$(cat "$ATTEMPT/python.pid")" || true; fi
        if [[ -f "$ATTEMPT/console.log" ]]; then tail -n 60 "$ATTEMPT/console.log"; fi
    fi
done
ps -eo pid,ppid,pgid,etime,args | grep -E '[r]un_b19_ndp_sppf_v1|[f]inish_b19_ndp_sppf_v1'
```

```bash
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_NDP_SPPF_v1
NAME=yolo26n_b19_ndp_sppf_v1
ATTEMPT="$(cat "$WORK/runs/detect/${NAME}_train.current_attempt")"
tail -n 80 -F "$ATTEMPT/console.log"
```

进入已有窗口：

```bash
tmux attach-session -t ndp_sppf_v1_train
```

## 正式训练成功后 test → diagnose → package

```bash
set -euo pipefail
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_NDP_SPPF_v1
B19_PYTHON=${B19_PYTHON:-/root/miniconda3/bin/python}
export B19_PYTHON
cd "$WORK"
bash tools/experiments/server_b19_ndp_sppf_v1.sh test
bash tools/experiments/server_b19_ndp_sppf_v1.sh diagnose
bash tools/experiments/server_b19_ndp_sppf_v1.sh package
```

每个命令获得同一实验锁且保存独立 attempt。任何一步失败都会停止后续命令，保留已产生的结果。test 成功后会把权重/源代码/数据哈希绑定到 val/test 报告；diagnose 使用同一 best.pt；package 拒绝混用失败 attempt、其他 run 或未验证产物。

```bash
set -euo pipefail
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_NDP_SPPF_v1
NAME=yolo26n_b19_ndp_sppf_v1
SHORT="$(git -C "$WORK" rev-parse --short=12 HEAD)"
cd "$WORK/artifacts/experiments"
sha256sum -c "${NAME}_${SHORT}.tar.gz.sha256"
ls -lh "${NAME}_${SHORT}.tar.gz"*
```

该路径只是成功后的产物规则；未执行完整训练/评估/诊断前，不存在最终结果包。

## 独立故障诊断预检

只有需要排查时才单独执行，之后的正常 train 仍会做自己的全新预检，不消费旧预检模型或状态。

```bash
set -euo pipefail
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_NDP_SPPF_v1
B19_PYTHON=${B19_PYTHON:-/root/miniconda3/bin/python}
cd "$WORK"
B19_PYTHON="$B19_PYTHON" bash tools/experiments/server_b19_ndp_sppf_v1.sh preflight
```

检查当前 attempt 的 `audit_path.txt` 所指目录：`resolved.json` 是完整配置/数据/预训练审计；`structural.json`/`initialization.json` 是状态与初始化；`preflight/checks.json` 记录真实任务梯度、MuSGD 更新/反事实重放、AMP、累积、峰值显存、耗时及失败证据。`failed.json` 和 `console.log` 保存异常。没有 `passed.json` 或进程退出非零时，不能认为预检通过。

环境缺项须恢复真实 b19 环境后再运行，不能改 v1 的公式或训练条件。单个 FP32 patches 132 MiB 与少量点卷积 GFLOPs 都不是整网显存/延迟结论。

## 离线恢复结果包

先核对外部 `.sha256`，再解压到新目录；包内 `source.tar` 是交付 SHA 的完整已提交源码，包含模块注册与 YAML。解开 source.tar，切到该目录并用 b19 对应依赖环境，确认 `ultralytics.__file__` 指向该源码后，才能用 YOLO 读取包内 best/last。数据集不在包中，按记录的数据配置及图像/标签哈希恢复原数据。包中的 `checksums.json` 对应每个成员内容；`.manifest.json` 是其外部副本。
