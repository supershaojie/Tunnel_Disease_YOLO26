# BDI-P3 v1 服务器操作

固定 WORK：`/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_BDI_P3_v1`。
固定 run：`yolo26n_b19_bdi_p3_v1`。Python 默认为 `/root/miniconda3/bin/python`，可用 `B19_PYTHON` 指定同一环境。
本次交付未 SSH、未启动训练；本地 RTX 2060 的小批验证不等于服务器预检。

## 部署固定提交

使用最终交付的 `deploy_bdi_p3_v1.sh` 完整代码块。它由已提交的
`tools/experiments/deploy_b19_bdi_p3_v1.sh` 生成，并填入已经和远端核对的 40 位 SHA。
该文件在提交后生成，因此不参与该提交本身的哈希。

受版本控制的部署脚本接收一个完整 SHA。若已在经过核对的该实验 checkout 内，重验部署可运行：

```bash
set -euo pipefail
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_BDI_P3_v1
SHA="$(git -C "$WORK" rev-parse HEAD)"
bash "$WORK/tools/experiments/deploy_b19_bdi_p3_v1.sh" "$SHA"
```

初次部署请使用交付文件中填好 SHA 的自包含脚本，不用移动分支名替代 SHA。
部署核实基础仓库 origin 为 `https://github.com/supershaojie/Tunnel_Disease_YOLO26.git`；不改 remote。
已有目标 commit 即跳过联网；缺少才以 HTTP/1.1 fetch 指定分支，显示 progress。
`lowSpeedLimit=1024`、`lowSpeedTime=120` 是低速检测，不是整个 fetch 的总超时；失败保留退出码并停止。

新目标建立独立 detached worktree 来固定提交；不改基础 checkout 的分支或文件。
已存在目标若 SHA 不同或有未提交文件，展示实际状态并停止，不自动切换/重置。
部署与所有阶段共用基础仓库 `.experiment-locks/yolo26n_b19_bdi_p3_v1.lock`，防止运行期间更新代码。
保留其他实验、旧窗口和结果。

## 一次 train，自动预检

以下命令用独立 tmux 窗口运行；已有同名会话会停止而不覆盖。窗口命令结束后仍保留，便于查看错误。

```bash
set -euo pipefail
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_BDI_P3_v1
SESSION=b19_bdi_p3_v1
if tmux has-session -t "$SESSION" 2>/dev/null; then
    printf 'Existing session preserved: %s\n' "$SESSION" >&2
    exit 3
fi
tmux new-session -d -s "$SESSION" -c "$WORK"
tmux set-option -w -t "$SESSION:0" remain-on-exit on
tmux send-keys -t "$SESSION:0" -l "bash '$WORK/tools/experiments/server_b19_bdi_p3_v1.sh' train"
tmux send-keys -t "$SESSION:0" Enter
tmux attach-session -t "$SESSION"
```

离开而不停止：先按 `Ctrl+b`，松开，再按 `d`。
正式 run 已存在时入口拒绝再次 train；失败 run/attempt 均保留，不自动 resume、删除或覆盖。
成功创建 tmux 不等于训练成功，必须看真实日志/进程。

train 自动完成以下步骤：核对全部 b19 配方与数据、原始预训练哈希/架构 → 独立 preflight → 子进程成功退出 → 全新 Python 正式训练。
预检保持真实 batch32/640、在线增强、MuSGD、AMP 和原生 warmup/累积/裁剪，最多观察 128 个 batch。
梯度、参数更新、P2 可达、非零残差、Validator、EMA/重载/fuse 或显存失败时停止，不降低 batch、不放宽判据。
不需要先手动运行 preflight；单独子命令仅供定位失败：

```bash
set -euo pipefail
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_BDI_P3_v1
bash "$WORK/tools/experiments/server_b19_bdi_p3_v1.sh" preflight
```

服务器环境需与已归档的 Python 3.12.3 / torch 2.8.0+cu128 / Ultralytics 8.4.98 / RTX 4090 一致；不自动升级依赖。
原始 b19 args 默认从基础仓库固定 run 读取，data 从该配置解析。不能改数据目录猜测路径、换初始化或关闭增强。
完整差异见每次 audit 的 `resolved.json` 和正式 run 的 `provenance/effective_config.json`。

## 监控

```bash
set -euo pipefail
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_BDI_P3_v1
NAME=yolo26n_b19_bdi_p3_v1
PROJECT="$WORK/runs/detect"
for STAGE in train preflight; do
    POINTER="$PROJECT/${NAME}_${STAGE}.current_attempt"
    if [[ -f "$POINTER" ]]; then
        ATTEMPT="$(cat "$POINTER")"
        printf '\n%s: %s\n' "$STAGE" "$ATTEMPT"
        cat "$ATTEMPT/process_status.json"
        if [[ -f "$ATTEMPT/exit_status" ]]; then cat "$ATTEMPT/exit_status"; else echo 'exit_status 尚未记录'; fi
        if [[ -f "$ATTEMPT/python.pid" ]]; then
            PID="$(cat "$ATTEMPT/python.pid")"
            ps -p "$PID" -o pid,ppid,etime,stat,args || true
        fi
        tail -n 30 "$ATTEMPT/console.log"
    fi
done
nvidia-smi
```

没有 exit_status 只表示尚未记录，不能单凭这一点判断正在正常训练。PID 可能退出或复用，结合命令、时间及日志检查。

```bash
set -euo pipefail
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_BDI_P3_v1
NAME=yolo26n_b19_bdi_p3_v1
ATTEMPT="$(cat "$WORK/runs/detect/${NAME}_train.current_attempt")"
tail -F "$ATTEMPT/console.log"
```

按 Ctrl+C 仅退出上述 tail。返回窗口：

```bash
SESSION=b19_bdi_p3_v1
tmux attach-session -t "$SESSION"
```

每个阶段保留独立 attempt：开始时间、full SHA、实际命令、shell/Python PID、console、process_status、退出码。
入口锁覆盖整个前后台子进程生命周期；TERM/INT 转发给独立进程组，Python/tee 退出码分别记录，失败不会被上层返回 0 掩盖。
如操作系统强制杀死全部进程，退出记录可能缺失，必须按未知/未完成处理。

## 正式结束后 test → diagnose → package

以下代码块各自可独立复制，长时间评估可在上面的 tmux 会话内执行。任何一步失败先读该阶段 current_attempt，保留现场。

```bash
set -euo pipefail
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_BDI_P3_v1
bash "$WORK/tools/experiments/server_b19_bdi_p3_v1.sh" test
```

test 对验证选择的 best.pt 执行全量 FP32 val/test，各自独立目录。保存完整 args、指标、逐 IoU AP、曲线、混淆矩阵、预测/匹配统计。
置信度操作点在 val 上选择后冻结到 test；如 b19 best.pt 存在，则按相同评估设置补评，不重训。

```bash
set -euo pipefail
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_BDI_P3_v1
bash "$WORK/tools/experiments/server_b19_bdi_p3_v1.sh" diagnose
```

固定 16 张 val 图片，同权重开/关 R，保存文件清单、U/D/V3/R 数值、TP/FP/FN、补回/丢失目标、定位变化和 off/on 可视化。
conf=.25、IoU 匹配 .5/.75；不使用 test 挑图或改变结构。

```bash
set -euo pipefail
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_BDI_P3_v1
bash "$WORK/tools/experiments/server_b19_bdi_p3_v1.sh" package
```

只有同源 preflight/train/val/test/diagnose 全部成功且必要文件齐全才生成包。
位置为 `$WORK/artifacts/experiments/`，命名规则为 `yolo26n_b19_bdi_p3_v1_<提交前12位>.tar.gz`，
附 `.sha256`、`.manifest.json`、`.json` 收据。最终完整路径/大小/SHA256 由成功命令打印。
未执行正式生命周期前，不会存在可交付的正式训练结果包。

```bash
set -euo pipefail
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_BDI_P3_v1
NAME=yolo26n_b19_bdi_p3_v1
SHA="$(git -C "$WORK" rev-parse --short=12 HEAD)"
cd "$WORK/artifacts/experiments"
sha256sum -c "${NAME}_${SHA}.tar.gz.sha256"
tar -tzf "${NAME}_${SHA}.tar.gz" | sed -n '1,20p'
```

包内 `source.tar` 是该训练提交的源码快照，含正常模块注册、YAML、runner、验证器和两份文档。
恢复时在新的空目录解开 source.tar；从该目录运行 Python，确认 `ultralytics.__file__` 指向该目录，再使用
`YOLO('.../weights/best.pt')`。不把它覆盖到正在使用的基础仓库。包内没有数据集全量图片，数据仍需原 b19 数据配置和内容指纹核验。
