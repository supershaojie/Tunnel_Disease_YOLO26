# b19 + SCE-Fusion

固定基线：`4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6`。分支：`exp/sce-fusion-b19`。
正式训练状态：**NOT_STARTED**。本目录提供实现、有限验证和后续独立评估入口，不包含权重、数据或训练结果。

## 固定结构

原始第 0–22 层保持不变。`[16,19,22] → SCEFusion(23) → Index(24,25,26) → Detect(27)`；
末层仍为原生 Detect，O2M/O2O、detach、分配器、损失与 stride `[8,16,32]` 保持原生行为。

- 仅支持 n 尺度，nc=1；三路实际通道为 64/128/256。
- 三个独立 1×1 投影到 64 通道；P3 平均池化、P5 双线性插值到实际 P4 网格。
- 每个来源独立经过两个轴向 context block，共六个；局部 DW 3×3，两个方向分别用均值/极值描述、
  1×1 降维和 DW 7×1 / 1×7，随后 64→128→64 通道混合及残差。
- 三个独立 router，输入为本来源原始 Z 与另两个来源的 V；192→16→16→24。
  每组连续 8 个通道，共 8 组，每组仅对“来源 j、来源 k、零候选”做 FP32 softmax。
  末层清零，初始三候选各 1/3，两个非零来源不再次归一化。
- 消息恢复到各原尺度，独立 DW 3×3 和 1×1 输出投影后残差相加；三个 lambda 初值均为 0.1。
  resize、统计、softmax、消息加权及最终残差使用局部 FP32，输出转回输入 dtype。
- 支持矩形输入；不满足通道、batch/device/dtype 或两个空间轴 4:2:1 比例时明确报错。

Unfused 新增参数的静态分解：投影 29,056，六个 context 209,280，router 10,872，refine 2,112，
out 29,568，lambda 3，总计 **280,891**。`verify.py` 另行实测参数量，不使用静态规格冒充运行结果。

| 模型/模块             |   Unfused | Native fused |
| --------------------- | --------: | -----------: |
| b19 nc=1              | 2,504,190 |    2,375,031 |
| SCE 新模块 / 整网净增 |   280,891 |      278,523 |
| SCE 整网              | 2,785,081 |    2,653,554 |

此表为本机有限验证的测量值。没有替换或删除任何原生参数。原生端到端 Detect fuse 同时删除 O2M，
参数下降不能全部归因于 BN 折叠。未测量公平推理速度或完整 FLOPs；通用计数工具可能漏掉池化、插值、
统计和逐元素交互。

## 初始化与训练入口

`train.py` 必须读取完整 b19 `args.yaml`，拒绝缺字段的局部配方。报告记录原文件 SHA、全部原配置、
解析值、逐字段变化以及 Trainer 的实际生效值。只改模型/权重/数据路径、输出位置和明确指定的设备。
原始 `yolo26n.pt` SHA256 必须是
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`。

真实 `setup_model → SCETrainer.get_model` 路径先调用原生 DetectionTrainer 构建 nc=1 参考模型并加载原始权重，
保留原生类别适配和正常 RNG 消耗，再隔离构建新图。逐键复制 `model.0.*`–`model.22.*`，
将完整前缀 `model.23.` 映射到 `model.27.`，检查全部状态的形状、值与独立存储。
参考模型释放，不作为教师或包装模型保留。已训练 SCE checkpoint 直接使用新图状态，不再次重映射。

原始预训练保留 606 个状态张量 / 2,416,120 个参数元素；原生类别适配 102 个状态张量 / 88,070 个参数元素。
原生合计 708 个状态张量 / 2,504,190 个参数元素全部继承，含 Detect 的 240 个状态张量 / 241,566 个参数元素。
SCE 新增 211 个状态张量 / 280,891 个参数元素。buffers 不计入参数元素。意外遗漏和有意移除均为 0。

实验 Trainer 在调用原生优化器构造器时使用临时名称视图，保留 b19 将 `model.23.*cv3*` 按 3×LR 分组的原规则，
实际图、参数和 checkpoint 仍保持 Detect 位于 27。SCE 所有新参数恰好入组一次，按原生 MuSGD 规则处理。
OOM 重试请求在修改 batch 之前报错；原生 Trainer 和其他实验不受影响。不会自动降 batch 或等待 GPU 空闲。

入口切换到本 worktree，先建立/核验本地同名原始权重供原生 AMP 检查使用；不覆盖 SHA 不匹配的文件。
`YOLO_AUTOINSTALL=false`。原生 AMP 检查照常执行，失败则不启动固定 AMP 配方。
正式输出目录存在时拒绝覆盖、自动改名或续训。dry-run / verify 只使用临时训练目录。

| 入口          | 必填参数                                    | 可选参数                                                       |
| ------------- | ------------------------------------------- | -------------------------------------------------------------- |
| `train.py`    | `--data --weights --baseline-args`          | `--project --name --device --dry-run --report`                 |
| `verify.py`   | `--data --weights --baseline-args --report` | `--device`                                                     |
| `validate.py` | `--weights --data --project --name`         | `--device --split {val,test} --diagnostic-samples {0,1,2,3,4}` |

三个入口均支持 `--help`。report 是明确的 JSON 文件路径，自动建立父目录。默认正式输出名为
`sce_fusion_b19_e200_i640_b32_s42`，project 为本 worktree 的 `runs/detect/sce_fusion`。
独立评估写入指定目录的 `evaluation.json`。

后续服务器正式训练/评价分别使用独立 tmux `sce-b19` / `sce-eval`；已有会话先查看状态，不能覆盖或 kill。
允许 GPU0 并发，不排队等待空卡。日志经 tee 时必须保留 Python 的真实退出码。

## 有限验证及边界

`verify.py` 检查矩形/错误输入、六块独立参数、每组路由数学、轴向脉冲和广播、lambda=0 受控直通、
640 图连接/缓存、原生 detach、全状态继承、独立参考 RNG 和逐参数原生优化器分组。
在独立副本上用非空合成框和真实 E2ELoss/MuSGD 完成每种精度三个有效任务更新，每种精度最多 24 个微批；
记录 GradScaler、unscale 后有限性/非零梯度、跳步、参数变化和零初始化 router 上游后续梯度。
CPU FP32 与可用 CUDA AMP 分别报告。合成数据不代表数据集精度，smoke 模型不可用于正式初始化。

生命周期使用已改变的 lambda 和非均匀 router，检查原生 EMA/save、实际评估加载入口、新进程受限加载、
默认 eval/predict、native fuse 和重复 fuse。保存使用原生 FP16 EMA 序列化，比较参考做相同量化，
融合比较同一 O2O 原始输出，报告绝对/相对误差与容差。所有人工设值、BN 更新和 smoke 均在副本中进行。
另复制两张非空 val 样本到临时目录，以 160/B2 检查 FP32 原生验证器及指标/曲线 schema；不修改原数据缓存，
不将这个有限入口检查作为数据集精度结果。

本机环境为 Windows、Python 3.11.15、torch 2.7.1+cu118、torchvision 0.22.1+cu118、RTX 2060；
Albumentations 2.0.8 已安装，区别于文档的服务器环境。本机结果不得当作服务器 RTX 4090 / B32/640 的实测。
保留原生确定性警告；不修改 AMP 检查结果或全局确定性设置来绕过算子警告。

报告用 PASS/FAIL/UNVERIFIED 区分结果。开发不连接服务器、不进行完整 val/test、不启动 200 轮训练。
数据核对 YAML SHA、单类和 split 图像/框数量，并记录本地标签摘要；没有服务器图像字节比对的结论。
默认 train/eval/predict 是支持范围；tuple 节点 23 的 `profile`、`visualize`、`embed` 可选工具未承诺支持。
未开展 ONNX/TensorRT 导出、参数扫描、消融训练或多种子实验。

## 后续独立评价

`validate.py` 从学习后的 SCE checkpoint 恢复完整结构，调用原生验证器；FP32、640、batch=32、conf=0.001、
iou=0.7、max_det=300、rect=True、augment=False、workers=8、half=False、quantize=None、end2end=True、native fuse。
检查实际模型/输入 dtype。原生 CPU 验证器将 workers 改为 0，实际配置会写入报告。

输出 Precision、Recall、F1、AP50、按实际 IoU 数组定位的 AP75 和 mAP50–95，保留 PR/置信度曲线。
conf=0.25/0.50 的 P/R/F1 标为“曲线估计”，不伪造精确 TP/FP/FN。val 用于选择方案，test 仅作固定最终评估。
诊断默认关闭，最多四个指定 split 样本；只保留各候选均值/分位数、组间分布、lambda 和消息/残差范数统计。

设计参考 [Gold-YOLO](https://arxiv.org/abs/2309.11331) 的多尺度集中处理/分发和
[Coordinate Attention](https://arxiv.org/abs/2103.02907) 的两轴编码；实现遵循本实验固定公式。
没有整体引入 IIA_Fusion/HFFE 或第三方模块包。“来源互补”“背景抑制”“定位改善”仍是待验证假设。
原尺度直通不等于已证明无损融合；零候选也不保证训练后的 BN 路径严格为零，只有 lambda=0 的受控测试是恒等。
