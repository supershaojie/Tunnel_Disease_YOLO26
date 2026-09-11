# MGC-SPPF v1：服务器部署和操作

本页是供用户后续执行的说明。本次实施不登录启动/停止 AutoDL 训练，formal training = **NOT STARTED**。
必须使用最终实施报告中的完整 Branch、Commit SHA 和已核验远程 SHA，不从图片抄写。

## 部署

固定路径：

```bash
BASE=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_MGC_SPPF_v1
BRANCH=codex/exp-yolo26n-b19-mgc-sppf-v1
SHA='替换为最终报告的40位Commit SHA'
```

从本次交付代码取得 `tools/experiments/deploy_b19_mgc_sppf_v1.sh` 后执行：

```bash
bash /path/to/delivered/deploy_b19_mgc_sppf_v1.sh "$SHA"
```

该工具验证 origin 仓库、完整 SHA、b19 ancestry 和目标入口文件，取得与 server 相同的独占锁。
检查 MGC 入口进程及 WORK 中的 Python 进程；发现活动即拒绝更新。
本地已有目标 Git 对象就不联网；缺少时仅 fetch 指定实验分支，使用 HTTP/1.1、no-tags、
30秒 lowSpeed 和120秒总超时，单次请求，无无限重试。只有 fetch 成功才读 FETCH_HEAD 并比较 SHA。
字符串不符时输出双方原文、长度和首个差异位置，拒绝部署。
现有 worktree 必须干净且 detached；否则保留。工具只更新本实验独立 detached worktree，不操作其他实验。
不会执行 reset/clean/force push 或启动训练。

若需要自行初次获取交付代码，先按同样的有限 fetch 流程核对远程 SHA，再使用
`git worktree add --detach "$WORK" "$SHA"`；已存在 WORK 时使用部署脚本中的完整活动检查流程，
不要直接 checkout。网络失败立即停止，不用遗留 FETCH_HEAD 或旧 HEAD 开始训练。

## 正常入口

用户后续明确决定启动时，只需：

```bash
cd /root/autodl-tmp/projects/Tunnel_Disease_YOLO26_MGC_SPPF_v1
bash tools/experiments/server_b19_mgc_sppf_v1.sh train
```

`train` 自动进行 runtime/source/args 审计，然后在独立进程和独立目录执行完整 preflight。
只接受当前源码/配方一致且 native B32/640/MuSGD/AMP 通过的 receipt。
PASS 后重新按 seed42 创建正式模型、原始预训练加载、optimizer、EMA、GradScaler、RNG 和 DataLoader。
不复用预检权重或 batch，不将预检计入正式 epoch。200 epochs 保留原 patience60。
已存在正式 RUN 会拒绝，不覆盖、不自动创建同名 `...2`。

默认解释器 `/root/miniconda3/bin/python`，保留原 `B19_PYTHON` override，显式设置 PYTHONPATH 到当前 WORK。
大导入前立即打印 experiment/stage/WORK/实际 HEAD/解释器/日志路径及 CUDA 可见性。
导入后打印并验证 `ultralytics.__file__` 属于此 WORK。正式环境与 b19 证据逐项比对：
Python3.12.3、PyTorch2.8.0+cu128、Ultralytics8.4.98、RTX4090、device0。
不能以本机历史文字代替真实读取结果；不重装环境、不修改 CUDA/TF32/CUBLAS 设置、不假设 GPU1 可用。

## 状态和日志

每次调用使用新的 `runs/detect/${NAME}_${STAGE}.attempt.XXXXXXXX/`。
其中有 `console.log`、`command.txt`、`commit.txt`、`source_sha256.txt`、`process_status.json` 和退出状态。
长预检逐阶段和逐 batch 输出；失败写 checks.json/traceback 和原生 step/batch ledger。
Python `-u`、stderr/stdout tee 保留；Python 非零优先保留，tee 失败也使入口失败。
SIGINT/SIGTERM 记录退出状态。仅存在进程表示已启动，不表示进入正式 1/200；不能用固定8秒无输出判失败。
正式 RUN 的 `logs/` 保存完成调用的正规日志副本，打包不依赖 `.attempt.*` symlink。

开发诊断可单独调用：

```bash
bash tools/experiments/server_b19_mgc_sppf_v1.sh preflight
```

正常启动无需手工先跑一次 preflight。正式训练后按阶段处理：

```bash
bash tools/experiments/server_b19_mgc_sppf_v1.sh test
bash tools/experiments/server_b19_mgc_sppf_v1.sh diagnose
bash tools/experiments/server_b19_mgc_sppf_v1.sh package
```

`test` 固定选定 best，生成 MGC val/test 和同条件 b19 test 对照，不据 test 调 epoch/阈值/TTA。
`diagnose` 只读固定 val 清单，输出 JSON/CSV/manifest。`package` 要求完整正式材料，缺关键文件即失败。
未训练时不能把 dry-run 合成包当作正式结果包。

## tmux

先用 `tmux list-sessions` 查看已有会话，已有同名会话先查看状态，不自动杀会话或发送 Ctrl+C。
需要创建会话时可由用户运行 `tmux new-session -s mgc_sppf_v1`，在其中执行正常入口。
建议 `tmux set-option -t mgc_sppf_v1 remain-on-exit on` 保留退出日志。
退出查看并保持任务运行用 **Ctrl+B，D**。

在 tmux 外查看已有会话用 `tmux attach-session -t mgc_sppf_v1`；在 tmux 内切换用
`tmux switch-client -t mgc_sppf_v1`。死 pane 不接收命令，不在死 pane 输入 exit 期待执行。
活动进程、锁、会话冲突时停止并由用户确认状态，不批量删锁、不停止别的实验。
