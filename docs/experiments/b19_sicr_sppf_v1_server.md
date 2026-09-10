# SICR-SPPF v1 AutoDL 操作

只运行固定v1：200epoch/640/batch32/MuSGD/seed42。以下命令由用户在b19原环境执行；本次实施没有远程启动训练。

## 同步和固定版本

将最终交付的完整40位提交填入 `SHA`。从 b19 主工程只 fetch，然后在新的独立 detached worktree 部署，不修改b19工作文件，也不叠加其他实验分支。

```bash
set -euo pipefail
BASE=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_SICR_SPPF_v1
BRANCH=codex/exp-yolo26n-b19-sicr-sppf-v1
SHA=REPLACE_WITH_DELIVERED_40_CHARACTER_COMMIT
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]]
git -C "$BASE" -c http.version=HTTP/1.1 fetch --no-tags origin "$BRANCH"
test "$(git -C "$BASE" rev-parse FETCH_HEAD)" = "$SHA"
DEPLOY=$(mktemp /tmp/deploy-sicr-v1.XXXXXX.sh)
git -C "$BASE" show "$SHA:tools/experiments/deploy_b19_sicr_sppf_v1.sh" > "$DEPLOY"
bash "$DEPLOY" "$SHA"
rm -- "$DEPLOY"
cd -- "$WORK"
test "$(git rev-parse HEAD)" = "$SHA"
git status --short
export B19_PYTHON=/root/miniconda3/bin/python
"$B19_PYTHON" -c 'import sys,torch; print(sys.version); print(torch.__version__); print(torch.cuda.get_device_name(0))'
```

正式入口检查 Python3.12.3、torch2.8.0+cu128、Ultralytics8.4.98、RTX4090。不要为跳过检查升级框架或改参数；先恢复原 b19 环境。

必须已存在原b19 `args.yaml`、数据和原始 `yolo26n.pt`。展开命令证据已纳入源码，仍会校验它能重现args。训练初始化不使用任何best.pt。

## 预检与正式启动

```bash
cd /root/autodl-tmp/projects/Tunnel_Disease_YOLO26_SICR_SPPF_v1
export B19_PYTHON=/root/miniconda3/bin/python
bash tools/experiments/server_b19_sicr_sppf_v1.sh preflight
bash tools/experiments/server_b19_sicr_sppf_v1.sh train
```

`train`自动在新进程再执行当前代码/数据的synthetic B32/640预检，随后重新从原始预训练和seed初始化正式训练。没有batch16、低batch数据集小训练或复用预检权重。

需要长连接后台运行时，用现有tmux：

```bash
tmux new-session -d -s y26_sicr_v1 \
  'cd /root/autodl-tmp/projects/Tunnel_Disease_YOLO26_SICR_SPPF_v1 && bash tools/experiments/server_b19_sicr_sppf_v1.sh train'
tmux attach -t y26_sicr_v1
```

前台正式启动与tmux命令选一种执行；共享flock会拒绝并发阶段或部署。已存在正式run时拒绝覆盖/自动resume。日志在 `runs/detect/yolo26n_b19_sicr_sppf_v1_<stage>_<time>_<pid>.log`，成功/失败及python和tee退出码均保留；训练run内拷贝常规日志文件。

## 收尾

```bash
cd /root/autodl-tmp/projects/Tunnel_Disease_YOLO26_SICR_SPPF_v1
bash tools/experiments/server_b19_sicr_sppf_v1.sh test
bash tools/experiments/server_b19_sicr_sppf_v1.sh diagnose
bash tools/experiments/server_b19_sicr_sppf_v1.sh package
```

也可首次执行 `bash tools/experiments/server_b19_sicr_sppf_v1.sh all` 顺序完成三个阶段。代码和训练提交必须一致；test/diagnose只评估和观察。test同时评估正式b19 best.pt，文件哈希必须为 `d0b2ca5a5d30de9ed002c64c9238b182ddeccce644c055ae5f2dba566878bd5e`。

结果位置：

```text
runs/detect/yolo26n_b19_sicr_sppf_v1/
  weights/best.pt, weights/last.pt, args.yaml, results.csv, results.png
  provenance/{resolved.json,weights.json,optimizer.json,preflight/checks.json}
  completed.json
  baseline_comparison/{sicr_val,sicr_test,b19_test}/
  baseline_comparison/comparison.{json,md}
  baseline_comparison/diagnostics/sicr_sppf_diagnostics.{json,md}
artifacts/experiments/yolo26n_b19_sicr_sppf_v1_<12位提交>.tar.gz
artifacts/experiments/yolo26n_b19_sicr_sppf_v1_<12位提交>.tar.gz.sha256
```

归档后在输出目录运行 `sha256sum -c <文件名>.sha256`；也可 `tar -tzf <文件名>.tar.gz` 列出成员。恢复源码时把包内 `run/source.tar` 解压到新的空目录，让该目录优先于已安装Ultralytics后再加载权重。

package失败只重试package，不把它归为test失败。已成功的test会验证其绑定哈希及产物后复用。诊断已存在时拒绝覆盖；保留旧目录后再显式处理，不能静默替换历史证据。目录 `.attempt.*` 及链接不是正式归档来源。

如果写包中途失败留下同名文件，默认入口会保护它并拒绝覆盖。保留失败包作为排查证据，在确认没有其他阶段运行后指定新的输出名重试：

```bash
"${B19_PYTHON:-/root/miniconda3/bin/python}" tools/experiments/finish_b19_sicr_sppf.py \
  --stage package --output artifacts/experiments/sicr_sppf_v1_package_retry1.tar.gz
```

## 本地结构验证

以下仅适合CPU开发验证，不是正式训练命令：

```bash
python tools/experiments/verify_b19_sicr_sppf.py --local \
  --baseline-root /path/to/original/b19/project \
  --baseline-args /path/to/archived/b19/train_run/args.yaml \
  --output artifacts/local_preflight
python -m pytest tests/test_sicr_sppf_v1.py -q
```

`--local`同样核对原始checkpoint和全量数据manifest，synthetic检测反向只使用B1/640，并清楚写入 `local_only=true`；这个记录不能用于宣称AutoDL正式预检通过。
