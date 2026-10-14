# 原始 YOLOv5n / B19 对比实验

本目录自包含地实现 `comparison-b19-v1`。开发基点为
`4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6`；模型是官方 **v7.0 原始 YOLOv5n P5 检测模型**，
完整后端 SHA 为 `915bbf294bb74c859f0b41f1c23bc395014ea679`。没有调用新 Ultralytics 的 YOLOv5u。
服务器正式训练状态为 **NOT_STARTED**，本机所有训练、评测、测速产物均为 **SMOKE**。

## 入口与参数

| 入口           | 作用                                                          |
| -------------- | ------------------------------------------------------------- |
| `prepare.py`   | 检查源码锁，实际下载或校验复用官方 COCO 权重                  |
| `preflight.py` | 检查配置、三划分文件字节、类别、导入路径和环境                |
| `train.py`     | 新训练及显式恢复；原生检测损失、优化器、预热、EMA、AutoAnchor |
| `evaluate.py`  | 新进程重载训练权重，val/test 统一指标、完整预测和论文图       |
| `benchmark.py` | 独立 batch1 测速，训练/推理参数量和 640 输入 FLOPs            |
| `package.py`   | 显式结果目录，严格检查单模型核心包字节数及成员哈希            |
| `aggregate.py` | 只读标准输出并核验协议，汇总多模型，严格检查 100MB 总上限     |
| `validate.py`  | 小型合成数据有限验证；真实检测损失更新与跨进程评测            |

所有入口支持 `--help`。训练支持 `--config --data --weights --baseline-args --project --name --device --dry-run
--resume` 和重复 `--set key=value`。评测另有 `--split val/test --result-dir --threshold-from`。
`package.py --result-dir` 必填；`--max-bytes` 可显式覆盖保存配置里的打包限制。

优先级为 **锁定官方默认 → 公共预设 → 配置文件 → 显式 CLI/--set**。未知键、重复 YAML 键、矛盾 CLI
设置、非法概率和不支持的优化器会报错。`dry-run` 不访问服务器数据，不下载模型，不创建正式输出目录；
参考文件缺失时只为展示计划使用历史增强核对清单，并输出 UNVERIFIED。正式 b19 训练必须提供真实 B19 args。

```bash
python experiments/comparison/yolov5n/train.py --help
python experiments/comparison/yolov5n/train.py \
  --config experiments/comparison/yolov5n/configs/default.yaml --dry-run \
  --set train.epochs=150 --set train.batch=16 --set train.optimizer=AdamW \
  --set train.lr0=0.001 --set augment.overrides.mixup=0.05 \
  --set augment.overrides.cutmix=0.0 --set augment.overrides.close_mosaic=5
```

默认配置为 epochs200 / imgsz640 / batch32 / seed42 / workers8 / GPU0 / AMP 请求开启。
自动名称来自最终 epochs/imgsz/batch/optimizer/seed；显式 `--name` 优先。目录已存在即拒绝新训练，
恢复须显式 `--resume path/to/weights/last.pt`，并保留相同 `--project/--name`。
恢复检查本实验身份、数据内容、增强/评测协议及优化配置，记录允许的 epochs/device/workers/路径差异，
恢复 model/optimizer/EMA/scheduler/scaler/早停状态；保留原始 COCO 初始化、原始请求快照与迁移清单。

## 官方配方与模型身份

`official_defaults.json` 从实际锁定的 `train.py::parse_opt`、`hyp.scratch-low.yaml`、
`utils/torch_utils.py::smart_optimizer` 解析生成。默认 **SGD、lr0=0.01、lrf=0.01、linear schedule**，
nominal batch64；也支持 Adam/AdamW、momentum（Adam beta1）、beta2、weight_decay、warmup、cos_lr 和原生损失字段。
v7.0 没有 optimizer=auto 模式，显式请求会报错。B19 MuSGD、lrf=0.003 和 YOLO26 损失没有被移植。
每个参数组实际 LR/decay/betas/momentum、损失/衰减缩放、动态累积与有效 batch 均单独保存。

原生 C3/SPPF、三层 anchor Detect、box/objectness/class 损失及正负分配保留。默认 AutoAnchor 真正运行；
`--set train.autoanchor=false` 是显式变体。实际锚框写入 effective_config。nc=1 从官方 COCO 权重原生迁移，
验证中继承了 343/349 状态项，六个不同 shape 的输出卷积 weight/bias 明确未迁移；训练前未 fuse。
checkpoint 保存 FP32 模型/EMA，避免最佳 val 模型保存时因半精度舍入改变参数；正式推理 dtype 单独实测。

`backend.lock.json` 保存 Git 源 SHA、原始 Git blob 哈希、私有副本哈希和补丁清单。私有补丁将准备、缓存、
日志、恢复、统一验证与最佳选择交给实验入口；删除自动 batch、隐式多 GPU、远程日志、动态权重下载、
结束时额外 FP16 验证/优化器剥离、NMS 超时截断和原数据 JPEG 写回。AMP 保留原生比较逻辑，fallback 使用已校验同族官方缓存。
仅私有可信 checkpoint 加载点使用带哈希验证的 `weights_only=False`，没有全局替换 `torch.load`。
源码许可证分别保留：官方 v7.0 为 GPL-3.0，冻结 B19 模块为 AGPL-3.0。

## 增强与数据协议

`../shared/b19/` 是独立命名空间，逐文件复用 YOLOv8n 已提交版本
`482e555248c0ba52b4bea2b474498cab61fd6f37` 的 B19 检测变换、数据审计、框匹配、AP 和预测交换实现。
全部共享文件随本分支提交，运行时不依赖另一分支或另一工作区。`shared.lock.json` 记录来源和逐文件哈希。
顺序为 Mosaic → CopyPaste → RandomPerspective → MixUp → CutMix → 可选冻结 Albumentations → HSV → 垂直/水平翻转。
MixUp 的 Beta 分布、CutMix 的候选区域/IoA/框裁剪、HSV 公式、空框处理保持冻结源码语义。
到达 `epochs-close_mosaic` 时关闭 Mosaic/CopyPaste/MixUp/CutMix，保留 HSV/仿射/翻转；短训练不存在负 epoch 触发。
v5 的重复增强不执行，样本边界只输出 RGB CHW uint8 和一次归一化 class/xywh；collate 仅加 batch_index，
像素除以255仍由原生训练循环完成。worker 使用 PyTorch worker seed，无逐 batch 随机重置。

`augment.profile=b19` 从真实 baseline args 取值；`custom` 从冻结 B19 默认配置起步；`none` 关闭所有随机操作；
`native` 为官方 v5 增强变体，不能偷偷接受其不支持的 CutMix/BGR/close_mosaic。
`augment.overrides` 可编辑任一增强概率/强度/关闭 epoch。非零操作不会静默丢弃。

B19 源码还含 **可选** Albumentations。历史服务器是否实际安装/启用、具体版本仍为 **UNVERIFIED**。
当前可编辑预设显式选择 `augment.albumentations=disabled`，避免环境隐式引入随机操作；
确认历史行为后可显式选 `frozen_default`，运行冻结默认 Blur/MedianBlur/ToGray/CLAHE 并要求 albumentations2.0.8。
两种政策有不同协议哈希，不能将其历史等价性描述为已核实。

冻结源码与共享提供器具有独立哈希，各后端适配器另有哈希。增强核心 SHA256 为
`0ba1f5198cb24dd404212a25ffceeed14aa01b5468a79beabbf7e1828f299b6b`，与 YOLOv8n 使用同一源码。
公共协议使用 YOLOv8n 提交cd9e69720b5c0d6b3fa297ec588431099ded9e58的规范化字段；正式启动前应以双方已推送版本再次核对 `protocol_sha256`。
公共哈希使用规范化的数据内容/划分、实际增强配置、评测几何及实现版本，不包含模型名、优化器、绝对目录；
完整请求另有 run config 哈希。图像/标签原文件只读，所有扫描索引和 native 标签/图像缓存放在本实验结果/缓存目录。

正式 preflight 检查 YAML、类别 crack=0、三划分排序清单、有效标签与图像字节 SHA256。
路径迁移后 YAML 哈希不一致会如实记录，数据身份由文件内容、类别和划分清单确定。不存在的数据不会替换为示例数据。

## 评测、选择和论文输出

每轮仅使用 val，复用同一次前向的检测输出和 loss logits 计算冻结 B19 greedy IoU matching、AP 和曲线。
`weights/best.pt` 按最高 **unified val mAP50–95** 保存；`native_best.pt` 保留原生加权 fitness 选择。
原生早停仍按原生 fitness，选择原因、最佳 epoch 和完成轮数单独记录。

独立评测必须重载本实验训练 checkpoint，不接受原始 COCO 权重冒充已训练 best。默认 FP32、低 conf0.001、
NMS IoU0.7、max_det300、rect=True、fuse=True、无 TTA。原始 anchor 头的得分是 objectness×class probability，
没有重复 sigmoid，也没有 end2end 假设。实际 dtype、批次形状、resize/padding、坐标还原和 fuse/NMS 状态均写入 metrics.json。

先评测 val 生成 `val_threshold.json`，再评测 test：test 严格校验阈值来自相同权重/数据/评测协议的 val。
输出 AP50/AP75/mAP50–95、conf0.25/0.50 下精确 TP/FP/FN/P/R/F1、val 选择阈值的精确统计，
并将 test 曲线峰值独立标为 `legacy_curve_peak`。AP 使用完整低阈值预测；展示阈值不影响 AP。
历史四组模型 P/R 的阈值和组合权重所属服务器提交仍为 UNVERIFIED，不能把 test 峰值冒充 val 选阈值。

标准 NPZ 与 YOLOv8n 共用同一保存函数：`boxes/scores/classes/image_offsets/image_ids`，
分别为原图像素 float32 xyxy/score、int32 class、int64 offsets 和图像 ID，覆盖所有评测图像和空图像。
同时保存 GT、排序图像索引、fraction 单位 summary.csv、可重绘曲线/混淆矩阵数值与 SVG。
固定对照图复用 YOLOv8n 的固定42哈希排序规则：前4个非空GT、前2个空GT，与模型预测无关；
`eval.fixed_images` 可指定同一已有清单，保存阈值、图片索引、实际成功/漏检/误检计数及少量裁剪。

参数量同时记录 nc=1 unfused/fused 整数和 M 值。本机测得 **1,765,270 / 1,760,518**；640×640 batch1
THOP2.0.20 测得 **2,064,793,600 MACs / 4,129,587,200 FLOPs**（1MAC=2FLOPs），并列明 SiLU/Concat/解码等未覆盖操作。
这些是本模型结构测量，合成样本指标和短测速不代表论文精度或 RTX4090 性能。

## 环境、运行保护及打包

实验入口使用 Python3.11+ / PyTorch2.6+ 的现代运行环境；本机 Python3.11.15 / Torch2.7.1+cu118 / RTX2060 已验证。
服务器参考 Python3.12 / Torch2.8 / RTX4090 仍 UNVERIFIED。`core.lock.txt` 和 `optional.lock.txt` 分别固定依赖；
Torch/Torchvision/CUDA 使用宿主已配置版本，实际记录，不自动安装或升级它们。
如需补依赖，只在本模型独立 `python -m venv --system-site-packages .venv` 中安装固定依赖；不改 B19 共享环境。

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r experiments/comparison/yolov5n/core.lock.txt
python -m pip install 'ultralytics-thop==2.0.20'
python experiments/comparison/yolov5n/prepare.py
python experiments/comparison/yolov5n/preflight.py
bash experiments/comparison/yolov5n/launch_tmux.sh cmp-y5n train \
  --config experiments/comparison/yolov5n/configs/default.yaml
```

tmux 已存在则显示状态并拒绝覆盖，不 kill；每模型独立 Python 进程。不等待 GPU 空闲、不轮询排队、不静默
降低 batch/imgsz、关闭 AMP、切 CPU 或更换优化器。Python 退出码与 FAILED 状态真实保存，训练日志直接 tee 到 run/train.log。
测速另外运行，CUDA 每段计时同步且排除磁盘/绘图；发现其他 GPU 计算进程标 CONTENDED，不能作为主论文测速。

核心包默认严格小于12,000,000 bytes，保留所有数值、完整 float32 test 预测、配置/来源/环境、SVG 和固定对照图。
排除权重、数据原图、缓存、环境、TensorBoard、重复图格式和重复全量清单；完整原文件保留。
超限会报成员大小，不删低分框、不改 max_det、不改 float16、不交付伪合格包。八模型总包须通过 aggregate.py 的
严格100,000,000字节检查，不能从单包大小推断总包合格。

```bash
python experiments/comparison/yolov5n/package.py --result-dir runs/comparison/yolov5n/YOUR_EXPLICIT_RUN
python experiments/comparison/yolov5n/aggregate.py --help
```

有限验证结论见 `VALIDATION.md` 与 `validation_report.json`。所有本机验证输出放在被忽略的 `.dev/`。
服务器源码拉取、独立 worktree、训练、val/test、测速、打包和下载命令随最终交付给出真实推送 SHA。
