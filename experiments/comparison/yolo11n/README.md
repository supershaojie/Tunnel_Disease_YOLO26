# YOLO11n / comparison-b19-v1

本实验复用 `codex/compare-yolov8n-b19` 的
`fab360ab2ada13b4da1f7e74a055bcf5d453c734`，新增官方 YOLO11n 对比模型。
正式 200 轮训练及全量独立 test **NOT_STARTED**。开发验证结果见 `validation.json`。

## 模型、依赖与复用

- 直接复用 `../yolov8n/` 的运行时、Dataset/Trainer 适配、独立评测、测速和精简打包；
  `../shared/` 的 B19 在线增强、关闭日程、匹配和 AP 实现继续共用。
- 私有源码仍是 Ultralytics v8.3.0，commit
  `6e43d1e1e5db72afbf686dee6745669bcb124b0a`，位于
  `../yolov8n/vendor/ultralytics_source/`，保留 LICENSE 和原有源码锁。
  本目录 `backend.lock.json` 只提供模型与官方权重身份，运行时合并完整源码锁并记录到结果中。
- 官方 COCO80 检测预训练为 [yolo11n.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt)，
  使用原生 Trainer 适配 `nc=1`。保留 C3k2、SPPF、C2PSA、Detect、DFL/TAL、损失和 EMA；
  不使用 v8n 或其他实验训练权重初始化。
- 正式 Trainer 重建后验证 nano 尺度和原生模块，检查结果及权重继承记录随初始化报告保存。
  `prepare.py` 首次确认官方资产及本地 SHA256，之后复用缓存。
- `requirements.txt` 覆盖实际检测、公共匹配、绘图和测速导入链，明确包含 `scipy==1.17.1`。
  独立 venv 使用宿主 Torch/Torchvision；不安装另一份 Ultralytics。
  普通 pip 安装保留已经满足版本要求的科学计算包，不使用会漏掉传递依赖的 `--no-deps`。

默认 epochs200/imgsz640/batch32/seed42/workers8/device0/AMP，显式
SGD/lr0=0.01/momentum=0.9/warmup_bias_lr=0。它沿用当前 v8n 项目设置，
不宣称与官方 auto 完全等价。其余优化、调度、早停字段来自锁定后端，
不把 B19 的 MuSGD 或整份 args 覆盖进来。实际值、覆盖来源、完成轮数与停止原因都会保存。

所有入口保留 `--config --data --weights --project --name --device --resume --set`。
例如 `--set train.batch=16 --set train.epochs=150`；恢复使用
`--resume /absolute/run/weights/last.pt`。新运行目录已存在时明确报错，不覆盖历史实验。

## 服务器准备

在母仓库拉取交付中的固定提交，独立 worktree 中运行。首次创建示例：

```bash
# GitHub 超时才执行：source /etc/network_turbo
M=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26
W="$M/.worktrees/compare-yolo11n-b19"
git -C "$M" fetch origin codex/compare-yolo11n-b19
# 将交付消息中的完整 SHA 赋给 PIN，再执行：
git -C "$M" worktree add -b codex/run-yolo11n-b19 "$W" "$PIN"
cd "$W"
python -m venv --system-site-packages .venv-y11n
.venv-y11n/bin/python -m pip install -r experiments/comparison/yolo11n/requirements.txt
.venv-y11n/bin/python experiments/comparison/yolo11n/prepare.py \
  --config experiments/comparison/yolo11n/configs/default.yaml
```

固定读取母目录已有数据 YAML（8414/2404/1202 张）和 B19 真实 args；缺失时应修正路径，
不下载替代数据。权重缓存默认 `$M/weights/pretrained/yolo11n.pt`，
本模型数据缓存位于 worktree 的 `experiments/comparison/yolo11n/cache/`。
已经创建的 worktree/venv 可直接复用，不必重复创建。

## 交互 tmux 训练

```bash
tmux new-session -s cmp-y11n
```

进入会话后直接运行：

```bash
cd /root/autodl-tmp/projects/Tunnel_Disease_YOLO26/.worktrees/compare-yolo11n-b19
.venv-y11n/bin/python -W ignore::FutureWarning -u \
  experiments/comparison/yolo11n/train.py \
  --config experiments/comparison/yolo11n/configs/default.yaml
```

每轮 tqdm 在终端内原地刷新；完成或异常后返回交互 shell，保留窗口和错误。
不接外部 `tee`，不使用 v8n 的 detached launcher，也不把会话创建当作训练开始。
训练 CSV、配置、指标和状态仍由现有运行时保存。
只屏蔽 FutureWarning，真实异常继续显示并返回非零退出码。

离开但保持运行：按 **Ctrl+b**，松开后按 **d**。
再次进入：`tmux attach-session -t cmp-y11n`；查看：`tmux list-sessions`。
可按显存、CPU、内存和 I/O 资源在 GPU0 同卡并行多个实验，包括 3–4 个；
没有固定实验数量上限、GPU 独占锁、等空卡队列或自动降低 batch/imgsz/AMP。

## 独立评测、参数/FLOPs、测速和打包

每轮仅验证 val，以公共 val mAP50–95 选 best；test 在训练完成后单独运行。
默认独立评测使用 FP32、640、batch32、conf0.001、NMS IoU0.7、max_det300、
rect=true、augment=false、原生 fuse，保留 YOLO11 原生解码与 NMS。

```bash
cd /root/autodl-tmp/projects/Tunnel_Disease_YOLO26/.worktrees/compare-yolo11n-b19
X=experiments/comparison/yolo11n
PY=.venv-y11n/bin/python
R=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26/runs/comparison/yolo11n/compare_yolo11n_b19_e200_i640_b32_s42
"$PY" "$X/evaluate.py" --weights "$R/weights/best.pt" --split val
"$PY" "$X/evaluate.py" --weights "$R/weights/best.pt" --split test \
  --val-metrics "$R/evaluate_val/metrics.json"

# 正式公平单图测速须在并发训练结束后单独执行。
"$PY" "$X/benchmark.py" --weights "$R/weights/best.pt"
"$PY" "$X/package.py" --result-dir "$R/evaluate_test" --output "$R/yolo11n_core.zip"
```

`summary.csv` 的 `curve_peak_P/curve_peak_R/curve_peak_F1` 对应各条公共置信度曲线的峰值，
供既有消融表取数；`metrics.json/curve_peaks` 保存各自置信度，三者不对应同一组 TP/FP/FN。
`legacy_curve_peak` 仍表示平滑 F1 最大处同阈值的插值 P/R/F1；
`val_frozen` 和 `fixed_confidence` 另名保存真实阈值计数。
CSV 原 `P/R/F1` 列继续指 val 冻结阈值，不能混写成各曲线峰值。
完整曲线、float32 test 预测、GT 和图像索引均保存。

benchmark 报告 nc1 未融合/融合参数，FLOPs 统一640，并记录计数覆盖范围，
单图前处理/forward/NMS/总耗时分别保存；并发资源竞争结果标记 CONTENDED，不能作正式 FPS。
核心 ZIP 严格小于12,000,000 bytes，复用已有 collector，包含指标、配置、CSV、曲线数值、
预测/索引、摘要和少量预览，不重复放 best/last、数据、环境或缓存；原始权重完整保留。
打包命令实际打印输出路径、bytes 和 SHA256，旁边写清单。

## 有限开发验证

`verify.py` 只做少量真实图像检测损失更新与保存后新进程重载，使用独立 SMOKE 输出目录。
其结果不能代表200轮收敛精度、全量 test、服务器4090环境或正式 FPS。
`validation.json` 记录开发机环境、官方预训练继承、实际 nano 结构、损失与重载结果，
以及实际验证包位置与大小。

开发验证可选命令（输出必须是新目录，不是正式训练的前置步骤）：

```bash
.venv-y11n/bin/python experiments/comparison/yolo11n/verify.py \
  --data /root/autodl-tmp/projects/Tunnel_Disease_YOLO26/datasets/Tunnel_Crack_AugFirst_Diverse5x_RandomSplit_7_2_1_seed42/data.yaml \
  --pretrained-cache /root/autodl-tmp/projects/Tunnel_Disease_YOLO26/weights/pretrained \
  --output experiments/comparison/yolo11n/cache/SMOKE_real_cuda640 --device 0
```

正式输出：

```text
/root/autodl-tmp/projects/Tunnel_Disease_YOLO26/runs/comparison/yolo11n/
  compare_yolo11n_b19_e200_i640_b32_s42/
    weights/{best,last}.pt
    evaluate_val/
    evaluate_test/
    benchmark/
    yolo11n_core.zip
```
