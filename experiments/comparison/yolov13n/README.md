# YOLOv13n / B19 对比实验

本实验基于 `fab360ab2ada13b4da1f7e74a055bcf5d453c734`，复用公共配置、B19 增强、指标、评测、测速和精简打包。
公共实现迁入 `../shared/`；v8 入口保留等价薄包装。其他实验的 checkout、训练和历史结果不受影响。
本次仅完成开发验证，**没有启动服务器 200 轮训练**。

## 作者模型与私有环境

- 作者：[iMoonLab/yolov13](https://github.com/iMoonLab/yolov13)，tag `yolov13`，完整源码提交
  `70f23ede45ee00a30cf6139c3d1ea7abe3df4eec`，内部包版本 `8.3.63`。
- `vendor/ultralytics_source/` 保留作者包、许可证与模型配置。`backend.lock.json` 记录实际取得的源码压缩包、
  原文件和私有补丁后的逐文件 SHA256。
- 官方 `yolov13n.pt` 为 10,570,688 bytes，实际 SHA256
  `6653035017b0f111f80ec11ed914874ea85699b104aeac1e46e517d16889d6b7`，与 GitHub release asset digest 一致。
  prepare 首次下载并核对，之后复用缓存，不安装 pip ultralytics。
- 保留 DSC3k2、A2C2f、HyperACE、FullPAD_Tunnel 与三尺度 Detect；逐层核对实际模块及 `from` 连线和
  `n=[0.50,0.25,1024]`。nc=1 由作者原生 Trainer 从 COCO 权重迁移，保留原生损失、TAL、EMA、解码与 NMS。
- 私有补丁删除自动 Flash 探测、不可达的 Flash 分支及失效开关，直接使用作者已有的
  QK → 稳定 softmax → AV。AMP 与 FP32 使用同一路径；不安装 cp311/Torch2.2 Flash wheel，不降级宿主 Torch。
- 其他必要补丁：可信旧 checkpoint 加载点显式 `weights_only=False`；删除全局 torch.load 替换、遥测、
  外部集成注册、pip 更新与字体下载；NumPy2 积分 API；缓存移到实验目录、JPEG 验证只读；增强关闭处刷新
  loader worker；原生保存返回同次序列化的轻量元数据，恢复训练尊重剩余 epoch 预算；删除 NMS 的整批超时截断，
  保证慢图不导致后续图像漏预测；AMP 探针失败明确抛错，不静默退回 FP32。详见 lock 的 patches。
- `dependencies.lock.txt` 为实际导入依赖，明确包含 `scipy==1.17.1` 和作者顶层导入需要的
  `huggingface-hub==0.23.2`。独立 venv 通过 `--system-site-packages` 复用服务器 Torch/Torchvision。

## 配置与产物口径

默认使用指定的 8414/2404/1202 划分、640、batch32、200epochs、seed42、workers8、GPU0、AMP，显式
SGD/lr0=0.01/momentum=0.9/warmup_bias_lr=0。其余调度、早停及损失来自固定作者默认值；不宣称与 native auto 等价。
修改 `configs/default.yaml` 或追加 `--set train.batch=16` 等逐字段参数；CLI 优先，不在内部改回默认值。
显式恢复使用 `--resume /绝对路径/last.pt`，原生恢复会覆盖的冲突配置会报错，不自动查找最新训练。

B19 增强复用 `../shared/provider.py` 和冻结核心：真实 Mosaic/MixUp/CutMix，原生重复随机增强被替换；
通过 Dataset/Instances/Format/collate 适配。关闭时间由 `epochs-close_mosaic` 的原生训练事件触发。
每轮只验证 val，以公共 val mAP50–95 选 best。默认独立评测为 FP32、rect=true、640、batch32、
conf0.001、IoU0.7、max_det300、无 TTA、原生 fuse。

`summary.csv` 的 P/R/F1 使用公共 B19 曲线的平滑 F1 峰处口径，`P_R_F1_rule=legacy_curve_peak`；
val 冻结阈值及精确计数单列在 `metrics.json` 的 `threshold_selection` / `val_frozen` 中。
峰值统计与阈值冻结统计不能混用。完整曲线数值为 `curves.npz`，完整预测为 `predictions.npz`，包含图像索引。
精简包保存这些数值、配置、训练曲线、来源、SVG 图与少量固定预览；不含数据、环境或权重。
best.pt/last.pt 完整留在服务器，供后续检测图、热图和恢复使用。

THOP 的 hook 计数不覆盖 HyperACE/注意力的 matmul、softmax、einsum、FullPAD 元素融合等操作，输出明确标记
`MEASURED_PARTIAL`；不能把未覆盖操作当作零成本。参数同时报告训练和原生融合后的推理口径。
公平单图测速应在 GPU 资源合适时另行执行；并发训练期间的延迟不能作为正式 FPS。

## 服务器操作

在服务器母仓库中执行。将 `Y13_SHA` 设为交付消息中的完整提交 SHA；使用固定提交建立独立 checkout，
不切换正在运行的 v5/v8 worktree。GitHub 超时时可先 `source /etc/network_turbo` 再重试失败命令。

```bash
cd /root/autodl-tmp/projects/Tunnel_Disease_YOLO26
Y13_SHA=填入交付的完整提交SHA
git fetch origin codex/compare-yolov13n-b19
git worktree add --detach ../Tunnel_Disease_YOLO26-y13n "$Y13_SHA"
cd /root/autodl-tmp/projects/Tunnel_Disease_YOLO26-y13n

python -m venv --system-site-packages .venv-y13n
.venv-y13n/bin/python -m pip install -r experiments/comparison/yolov13n/dependencies.lock.txt
.venv-y13n/bin/python experiments/comparison/yolov13n/prepare.py \
  --config experiments/comparison/yolov13n/configs/default.yaml

tmux new-session -s cmp-y13n
```

进入这个交互窗口后，直接运行以下命令。tmux pane 自身保存终端日志，Python 的 isatty 和回车刷新保持有效。
不接外部 tee；每轮原地从 0% 到 100%，下一轮换行。仅屏蔽已知 FutureWarning，异常照常抛出。

```bash
cd /root/autodl-tmp/projects/Tunnel_Disease_YOLO26-y13n
mkdir -p experiments/comparison/yolov13n/cache/logs
export YOLO_COMPARISON_LOG="$PWD/experiments/comparison/yolov13n/cache/logs/train_$(date +%Y%m%d_%H%M%S).log"
tmux pipe-pane -o "cat >> '$YOLO_COMPARISON_LOG'"
.venv-y13n/bin/python -W ignore::FutureWarning -u \
  experiments/comparison/yolov13n/train.py \
  --config experiments/comparison/yolov13n/configs/default.yaml
train_exit=$?
printf '\nYOLOv13n train exit=%s\n' "$train_exit"
```

训练成功或失败后仍回到该 shell，窗口保留。`Ctrl-b d` 分离；其他终端使用：

```bash
tmux attach-session -t cmp-y13n
tmux capture-pane -p -S -80 -t cmp-y13n
```

允许 GPU0 同卡并行多个实验，无独占锁、空卡等待、轮询队列或固定数量上限。按实际显存、CPU、内存和 I/O
自行安排；OOM 如实退出，不自动改变 batch、imgsz 或 AMP。更改参数时建议显式 `--name` 指定新 run 名。

训练后，从独立进程执行 val/test。以下 RUN 对应未经覆盖的默认配置；参数变化时用实际 run 路径。

```bash
cd /root/autodl-tmp/projects/Tunnel_Disease_YOLO26-y13n
RUN=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26/runs/comparison/yolov13n/compare_yolov13n_b19_e200_i640_b32_s42
PY=.venv-y13n/bin/python
EXP=experiments/comparison/yolov13n
$PY -W ignore::FutureWarning -u "$EXP/evaluate.py" \
  --weights "$RUN/weights/best.pt" --split val
$PY -W ignore::FutureWarning -u "$EXP/evaluate.py" \
  --weights "$RUN/weights/best.pt" --split test \
  --val-metrics "$RUN/evaluate_val/metrics.json"

# 在适合公平测速的时段另行运行；默认batch=1，640，同步计时，包含预处理/forward/NMS。
$PY -W ignore::FutureWarning -u "$EXP/benchmark.py" --weights "$RUN/weights/best.pt"

$PY "$EXP/package.py" --result-dir "$RUN/evaluate_test" --output "$RUN/yolov13n_core.zip"
du -h "$RUN/yolov13n_core.zip"
```

默认官方权重缓存：母目录 `weights/pretrained/yolov13n.pt` 及 `.source.json`；
训练权重：`$RUN/weights/{best,last}.pt`；评测：`$RUN/evaluate_{val,test}/`；
复杂度/测速：`$RUN/benchmark/`；精简包：`$RUN/yolov13n_core.zip`（严格小于 12,000,000 bytes，超限保留完整数据并报错）。
正式 run 和结果包目前尚未生成；本机有限验证的实际路径、大小及环境记录见 `validation.json`。
