# CCA-Fusion v2 服务器操作

最终交付另附本文件的固定完整SHA版本；以下仓库模板通过 `CCA_RELEASE_SHA` 接收同一个交付SHA。
不能使用分支最新HEAD替代固定SHA。每个代码块独立定义变量。在服务器执行，不会自动SSH启动训练。
Python默认 `/root/miniconda3/bin/python`；如需显式改路径，在每段修改 `B19_PYTHON`，运行时记录解释器和模块导入路径。

## 固定SHA部署

```bash
(
set -euo pipefail
SHA="${CCA_RELEASE_SHA:?请使用最终交付的完整SHA}"
BASE=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_CCA_Fusion_v2
BRANCH=codex/exp-yolo26n-b19-cca-fusion-v2
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]]
test "$(git -C "$BASE" remote get-url origin)" = https://github.com/supershaojie/Tunnel_Disease_YOLO26.git
exec 9>"$BASE/.yolo26n_b19_cca_fusion_v2.lock"
flock -n 9 || { echo 'CCA正在运行或部署'; exit 3; }
if ! git -C "$BASE" cat-file -e "$SHA^{commit}" 2>/dev/null; then
  GIT_TERMINAL_PROMPT=0 git -C "$BASE" -c http.version=HTTP/1.1 \
    -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 \
    fetch --progress --no-tags origin "$BRANCH"
fi
git -C "$BASE" cat-file -e "$SHA^{commit}"
if [[ -e "$WORK" ]]; then
  test "$(git -C "$WORK" rev-parse HEAD)" = "$SHA"
  test -z "$(git -C "$WORK" status --porcelain --untracked-files=no)"
  ! git -C "$WORK" symbolic-ref -q HEAD
  if pgrep -af '[p]ython.*(run_b19_cca_fusion|finish_b19_cca_fusion)'; then exit 3; fi
  git -C "$WORK" status --short
else
  git -C "$BASE" worktree add --detach "$WORK" "$SHA"
fi
test -f "$WORK/tools/experiments/server_b19_cca_fusion_v2.sh"
bash -n "$WORK/tools/experiments/server_b19_cca_fusion_v2.sh"
test "$(git -C "$WORK" rev-parse HEAD)" = "$SHA"
printf '部署完成: %s\n%s\n' "$WORK" "$SHA"
)
```

已有commit就跳过联网。lowSpeedTime=60是持续低速检测，不是整个fetch的60秒总超时。
现有工作树若SHA不符、存在源码改动或活动进程就停止；不reset/clean、不删除结果、不自动迁移旧实验。
部署和所有阶段使用BASE下同一把实验锁。未跟踪结果只展示并保留；入口拒绝已存在正式run。

## tmux只启动一次train

```bash
(
set -euo pipefail
SHA="${CCA_RELEASE_SHA:?请使用最终交付的完整SHA}"
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_CCA_Fusion_v2
B19_PYTHON=/root/miniconda3/bin/python
SESSION=y26_cca_v2_train
test "$(git -C "$WORK" rev-parse HEAD)" = "$SHA"
test ! -e "$WORK/runs/detect/yolo26n_b19_cca_fusion_v2"
if tmux has-session -t "$SESSION" 2>/dev/null; then echo 'tmux会话已存在，未重复启动'; exit 3; fi
tmux new-session -d -s "$SESSION"
tmux set-option -t "$SESSION" remain-on-exit on
tmux send-keys -t "$SESSION" "cd '$WORK' && B19_PYTHON='$B19_PYTHON' bash tools/experiments/server_b19_cca_fusion_v2.sh train; rc=\$?; printf '\\ntrain shell exit=%s\\n' \"\$rc\"; exit \$rc" C-m
tmux attach -t "$SESSION"
)
```

Ctrl+B，再按D，离开查看但保持任务运行。正常只调用一次train：它自动在独立Python子进程预检，
成功后重置seed、重新构建原始预训练模型、优化器和数据迭代器，再正式训练，不继承预检状态。
每次train都重新预检，不凭旧passed.json放行。单独preflight只供失败排查，不是正常流程的额外必跑步骤。

```bash
(
set -euo pipefail
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_CCA_Fusion_v2
B19_PYTHON=/root/miniconda3/bin/python
cd "$WORK"
B19_PYTHON="$B19_PYTHON" bash tools/experiments/server_b19_cca_fusion_v2.sh preflight
)
```

## 进度、进程与真实退出码

```bash
(
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_CCA_Fusion_v2
PROJECT="$WORK/runs/detect"
RUN=yolo26n_b19_cca_fusion_v2
tmux list-panes -t y26_cca_v2_train -F '#{session_name} dead=#{pane_dead} exit=#{pane_dead_status} pid=#{pane_pid}'
tmux capture-pane -pt y26_cca_v2_train -S -35
for STAGE in train test diagnose package; do
  POINTER="$PROJECT/${RUN}_${STAGE}.current_attempt"
  if [[ -f "$POINTER" ]]; then
    ATTEMPT="$(cat "$POINTER")"
    printf '\n%s attempt: %s\n' "$STAGE" "$ATTEMPT"
    cat "$ATTEMPT/process_status.json"
    if [[ -f "$ATTEMPT/exit_status" ]]; then cat "$ATTEMPT/exit_status"; else echo '尚无真实退出码'; fi
    PID="$(cat "$ATTEMPT/pid")"
    ps -p "$PID" -o pid,lstart,etime,args || true
    tail -n 12 "$ATTEMPT/console.log"
  fi
done
PREFLIGHT="$PROJECT/${RUN}_preflight/preflight.current_invocation"
if [[ -f "$PREFLIGHT" ]]; then
  ATTEMPT="$(cat "$PREFLIGHT")"
  printf '\n当前预检: %s\n' "$ATTEMPT"
  cat "$ATTEMPT/process_status.json"
  tail -n 30 "$ATTEMPT/checks.json"
fi
pgrep -af '[p]ython.*(run_b19_cca_fusion|finish_b19_cca_fusion)' || true
nvidia-smi
if [[ -f "$PROJECT/$RUN/results.csv" ]]; then tail -n 6 "$PROJECT/$RUN/results.csv"; fi
)
```

进程活性和退出状态分开看：没有exit_status不能判成功；SIGKILL或机器掉电可能留下running记录，
须结合PID启动时间、tmux和Python进程判断。不要用旧attempt的成功或失败替代当前attempt。
每阶段记录开始/结束时间、commit、源码摘要、展开命令、PID、解释器、console.log、Python/tee与最终退出码。

## train成功后test → diagnose → package

```bash
(
set -euo pipefail
SHA="${CCA_RELEASE_SHA:?请使用最终交付的完整SHA}"
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_CCA_Fusion_v2
B19_PYTHON=/root/miniconda3/bin/python
RUN="$WORK/runs/detect/yolo26n_b19_cca_fusion_v2"
test "$(git -C "$WORK" rev-parse HEAD)" = "$SHA"
ATTEMPT="$(cat "${RUN}_train.current_attempt")"
test "$(cat "$ATTEMPT/exit_status")" = 0
cd "$WORK"
for STAGE in test diagnose package; do
  B19_PYTHON="$B19_PYTHON" bash tools/experiments/server_b19_cca_fusion_v2.sh "$STAGE"
done
"$B19_PYTHON" - "$RUN/package_result.json" <<'PY'
import json, pathlib, sys
record = json.loads(pathlib.Path(sys.argv[1]).read_text())
print('实际归档绝对路径:', record['archive'])
print('字节数:', record['size_bytes'])
print('SHA256:', record['sha256'])
PY
ARCHIVE="$("$B19_PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["archive"])' "$RUN/package_result.json")"
cd "$(dirname "$ARCHIVE")"
sha256sum -c "$(basename "$ARCHIVE").sha256"
gzip -t "$ARCHIVE"
tar -tzf "$ARCHIVE" >/dev/null
)
```

任一步失败停止后续，保留结果。test使用训练机制按val选择的best.pt，分别独立进程执行完整FP32 val和test，
640/batch32/workers8/device0/conf0.001/iou0.7/max_det300/rect=True/augment=False；本版本用quantize=None表达FP32。
运行时核验实际参数dtype、forward autocast和原生end-to-end，记录TF32、速度和显存。
v2要求b19 best经相同入口复测并验证固定权重摘要；缺失时test阶段失败，package也拒绝缺少同条件b19比较的结果。
R@P≥0.85只从val选阈值，分数并列一起保留，冻结后读取test实际P/R/FPPI；不可达如实记录。
常规P/R工作点与固定阈值混淆矩阵分别标注。

diagnose使用路径字典序前16张验证图，保存清单/哈希、Q/K范数、R/U比、按角/边/内部区分的中心权重与熵、
四相位分布差异、对应权重期望位置、残差开关的原始one-to-one差异及IoU0.5/0.75 TP/FP/FN与目标匹配变化。
同权重关闭R通过临时输出hook完成，不改权重文件。这16张图不能替代完整val/test。

package只接受本次成功preflight/train/val/test/diagnose及一致的权重/代码/配置/数据身份。
真实路径由 `package_result.json` 给出，不猜文件名。归档内有best、配置、正式日志、指标/预测/图、预检更新证据、
源码tar与补丁、依赖版本和每文件大小/SHA256；临时attempt目录及链接不纳入主包；不包含原始数据、Conda、.git或其他实验。
用FileZilla下载输出的绝对路径及同名 `.sha256`；Windows本地可用 `Get-FileHash -Algorithm SHA256 -LiteralPath '已下载文件绝对路径'`
与服务器打印摘要核对。没有本次训练成功记录时，不存在可宣称已生成的正式结果包。

## 独立 val 与 test 命令

正常流程执行 server 的 test 阶段，会依次在独立进程运行 val、冻结 val 阈值、test，并运行同条件 b19 比较。
如需单独核查阶段，以下命令复用已完成且身份相同的结果，不重新选择阈值；完整 evaluation.json 由 test 阶段汇总。

```bash
(
set -euo pipefail
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_CCA_Fusion_v2
RUN="$WORK/runs/detect/yolo26n_b19_cca_fusion_v2"
B19_PYTHON=/root/miniconda3/bin/python
cd "$WORK"
"$B19_PYTHON" -m tools.experiments.finish_b19_cca_fusion_v2 --stage test --run "$RUN" --split val
"$B19_PYTHON" -m tools.experiments.finish_b19_cca_fusion_v2 --stage test --run "$RUN" --split test
B19_PYTHON="$B19_PYTHON" bash tools/experiments/server_b19_cca_fusion_v2.sh test
)
```

v2 diagnostics 在原16图诊断中额外保存 corner/edge/interior 的 center、noncenter、熵、归一化熵、C、G、
K_valid、对应期望位移、raw/actual residual 与 U 的范数比，以及 0.50\*G 的 mean/std/p50/p90/max。
对应位移是粗尺度单位下加权期望位置的偏移，不是 GT alignment error。
两个 residual ratio 分别比较同一 v2 correspondence 的未门控残差，以及去掉 prior 的同投影 v1 理论残差。
后者也改变了对应分布，不能把它要求为固定0.5或更小。
主包会校验全部普通成员的长度与SHA256并读到gzip尾部验证CRC；package_result.json列出关键成员。
服务器真实batch32/640预检及Linux tmux/flock行为仍须实际执行通过。当前任务未连接服务器、未启动正式训练。
