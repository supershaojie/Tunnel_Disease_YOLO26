# YOLOv8n / comparison-b19-v1

本目录仅准备官方 YOLOv8n 检测对比实验。服务器正式训练状态 **NOT_STARTED**。
母仓库及其 B19、DTR、SCE、Ours 源码、权重和历史结果没有修改。本实验从 B19
`4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6` 建立，独立分支 `codex/compare-yolov8n-b19`。

## 后端与来源

- 官方仓库：<https://github.com/ultralytics/ultralytics>。
- 标签 v8.3.0 的 tag object：`a46b49a27a97d44c642882a405c8b57979e6d368`。
- 解引用源码 commit：`6e43d1e1e5db72afbf686dee6745669bcb124b0a`，包版本 8.3.0。
- 私有源码：`vendor/ultralytics_source/`，包含 LICENSE、原生模型/训练/推理代码与配置。
  `backend.lock.json` 保存上游文件 SHA256、当前文件 SHA256 和每项私有补丁。
- 权重：<https://github.com/ultralytics/assets/releases/tag/v8.3.0> 中的 `yolov8n.pt`，
  asset id 195719301，6,549,796 bytes。官方没有公布该 asset 的 digest。
  prepare 计算并记录的 SHA256 仅是下载完整性记录，不冒充官方校验值。
- 运行时把私有源码置于导入路径首位，报告版本、完整源码 SHA 和 `ultralytics.__file__`。
  与 YOLOv5n 使用独立进程。禁止在已导入其他 Ultralytics 的 Python 会话内调用。

私有补丁仅涉及可信旧 checkpoint 的明确 `weights_only=False` 加载点、AMP 探针本地权重路径、
NumPy AP 积分 API 兼容、缓存路径、只读 JPEG 验证和增强关闭后的加载进程刷新。删除了原包全局 `torch.load` 替换、
外部集成注册、导入时遥测和训练时 pip 更新探测。没有修改 C2f、SPPF、Detect、DFL/TAL、
box/cls/dfl 损失、EMA、原生优化器选择或调度器。AMP 保留原生 FP32/AMP 相等性检查，
不下载其他模型，不伪造通过状态。

权重发布文件保留早期 YOLOv8 的 depth/width nano 配置。按 v8.3.0 原生 Trainer 重建 nc1 时，
该 tag 的 Detect 分类支路使用 DWConv；没有额外改写检测头。有限验证继承 313/391 个状态键，
其余 78 个重建键及源/目标维度逐项保存，包含分类维度与该版本原生分类支路的差异。
模型仍是 C2f/SPPF 的 v8 nano 检测结构；没有改成 YOLO11/26。

## 配置与参数

所有入口支持 `--help`。先编辑 `configs/default.yaml`，或用重复 `--set key=value` 覆盖。
优先级：锁定官方默认 → 公共预设 → 配置文件 → 显式 CLI / `--set`。YAML 使用安全解析，
重复键、矛盾重复覆盖、未知键和不合法概率会报错。`--dry-run` 不导入模型、不下载权重、
不创建正式输出目录。开发机缺失服务器文件时只展示明确标记的归档增强预览，正式训练必须读取真实 args。

```bash
python experiments/comparison/yolov8n/train.py --help
python experiments/comparison/yolov8n/train.py --config experiments/comparison/yolov8n/configs/default.yaml --dry-run \
  --set train.epochs=150 --set train.batch=16 --set train.optimizer=AdamW --set train.lr0=0.001 \
  --set augment.overrides.mixup=0.05 --set augment.overrides.cutmix=0.0 --set augment.overrides.close_mosaic=5
```

可改 optimizer、lr0/lrf、momentum、weight_decay、cos_lr、warmup_epochs/warmup_momentum/warmup_bias_lr、
patience、nbs 及锁定后端支持的其他训练键。显式 Adam 系优化器还支持外层 `train.betas=[0.9,0.999]`，
在原生优化器创建后应用到全部参数组，不把额外键传入旧 cfg。
`official_default` 在外层解析为 **auto**。锁定源码 `BaseTrainer.build_optimizer` 使用：

```text
iterations = ceil(train_images / max(batch, nbs)) * epochs
iterations > 10000: SGD, lr=0.01, momentum=0.9
otherwise: AdamW, lr=round(0.002*5/(4+nc), 6), beta1=0.9, beta2=0.999
auto 同时设置 warmup_bias_lr=0；请求的 lr0/momentum 可能被覆盖。
```

历史 8414 张 train、200 epoch、batch32、nbs64 的预算算式为 26,400 iterations，按旧源码会选择 SGD。
这只是源码推导；正式环境启动后以 `effective_config.json`、逐字段差异和每轮参数组记录为准。
显式 AdamW + LR 会走 AdamW；没有在内部改回 auto/MuSGD。
旧 v8 不按类别数缩放 box/cls/dfl（对应行在上游就是注释），权重衰减按 batch/累积/nbs 原生缩放并实录。
自动命名使用最终 epochs/imgsz/batch/seed；`--name` 优先。请为变更后的独立实验显式选择不同名称。

## B19 冻结增强与统一指标

`../shared/b19/` 从 B19 锚点抽取实际操作类、Instances、坐标操作、AP 和匹配函数；
`source.lock.json` 记录来源和提取边界。`../shared/provider.py` 保留原有顺序：
Mosaic → CopyPaste → 仿射/透视 → MixUp → CutMix → HSV → 上下/左右翻转。
MixUp 的 Beta(32,32)、CutMix 的 Beta(1,1)、候选贴片、IoA 标签筛选、Mosaic buffer 选图、
框裁剪和空框处理来自锚点函数，没有用 v8 同名函数代替。
B19 配置从真实 `--baseline-args` 读取；关闭日程在 `epochs-close_mosaic` 的零基 epoch 触发，
关闭 mosaic/copy_paste/mixup/cutmix，保留 HSV、仿射和翻转。关闭操作同时重启加载进程，
正常训练及关闭边界之后的断点恢复都使用关闭后的变换。

专用 `ComparisonDataset` 在原生 Trainer 的生产路径执行一次共享增强，显式转换 Instances，
然后使用 v8 原生 Format/collate/归一化/损失。B19/custom 下没有旧版原生重复增强及 Albumentations 额外随机操作。
这是显式移植策略；历史服务器是否安装并执行过 Albumentations 尚未核实，不能宣称所有历史随机采样逐次相同。
crack 数据只有检测框，非零 CopyPaste 请求会明确拒绝（没有分割多边形可复制）。
`none` 真正关闭随机增强；`native` 使用锁定 v8 原生增强，并拒绝其不支持的非零 CutMix；
`custom` 使用冻结提供器的可编辑配置。所有变体保存独立协议。
native 的协议散列锁定原生操作源码，并实录可选 Albumentations 的版本、启用状态和变换配置。
原生 Dataset 构建变换后发布实际增强值；native 矩形训练关闭 Mosaic/MixUp 的行为也进入协议。
训练输入尺寸、batch、rect、cache、epochs、seed 和关闭日程随增强身份保存。

当前未发现已有可靠 YOLOv5 对比分支。本分支提供自包含共享实现，另一模型应逐文件复用同一版本，
并比对 `augment_core_sha256` / `metrics_sha256` / `protocol_sha256`。另一模型尚未接入，跨模型一致性状态 **UNVERIFIED**。
协议只包含公共数据内容、增强、评测、输入尺寸和实现哈希，不包含模型/优化器/绝对安装路径。
完整 run config 与后端适配器另外散列。
匹配身份包含共享匹配函数和评测汇总；适配器身份包含训练接口及独立推理/坐标还原代码。
每个 best/last/epoch 权重冻结其训练增强配方及完整关闭日程，晚期零概率状态不替代早期配方。
独立评测和测速沿用该权重的增强身份；当前配置只提供实际评测/NMS/输入设置及调用哈希。
恢复时旧 best/epoch 仍保留各自的身份，避免按当前 profile 或 Albumentations 环境重新标记已学权重。
恢复入口用实际起始 epoch 合并所选权重的训练历史；更改 batch/输入尺寸等策略时，新权重记录全部阶段，
区别于全程使用新配置的训练。未改变策略的恢复沿用原身份，不重复增加阶段。

每轮只验证 val，复用该轮原生预测计算 B19 统一 mAP50–95，用它选择 `weights/best.pt` 和早停。
不使用 test 选权重，不做重复验证前向。保留可恢复的 last.pt；删除原生训练结束时再验证并剥除 optimizer 的路径，
拒绝 `train.save_hybrid=true`，避免原生自动标注功能把真值框加入验证预测。
训练 summary 保存已完成 epoch、最佳 epoch 和停止原因。
独立 evaluate 重载本实验权重，原生 DFL 解码与 NMS 使用 class score，无额外 objectness 和 end2end。
保存低阈值完整原图像素预测、GT、AP50/AP75/mAP50–95、固定 conf0.25/0.50 的真实 TP/FP/FN/P/R/F1，
以及单独命名的 legacy_curve_peak 插值诊断。先跑 val，再把 `--val-metrics` 传给 test，
即可冻结 val 阈值并得到真实 test 计数。没有 val 文件时主摘要 P/R/F1 留空，不冒充 val 阈值。

## 数据、恢复与输出保护

数据 YAML、图像、标签只读。正式入口审计 train/val/test 原划分、crack=0、有效内容、重复标签、
图像/框数和排序清单。适配后的 YAML hash 与历史 hash 分开报告；数据身份按内容和划分判断。
标签 cache（包括 cache=false 时的扫描 cache）及图像 npy 写入可改的 `cache_dir`，不写原数据目录。
模型/权重缓存与原 B19 环境互相独立。缺数据不会下载示例数据。

新训练目录已存在时报错。恢复必须用 `--resume /absolute/run/weights/last.pt`，核验源码、模型、
checkpoint 哈希、数据和原配方，保存恢复请求。旧后端会忽略的配方变更在恢复前拒绝；
batch/device 等原生允许变更会在有效配置中记录。没有自动最新运行查找、自动降 batch、切 CPU 或等待 GPU 空闲。
模型初始化、参数来源、官方默认、请求/有效差异、实际优化器组、累积与有效 batch、环境、数据协议、
逐轮损失/参数组、权重来源和训练/评测/打包状态各自保存。

## 服务器入口

使用本机已提交的分支，在服务器另建 worktree。无需安装另一个 Ultralytics 包；
依赖不足时在本模型独立 venv 安装锁定依赖，不能升级/卸载 B19 环境。
核心/可选依赖清单见 `dependencies.lock.json`；正式环境的实际版本另外记录，不把开发环境当服务器验证。

服务器缺少旧后端依赖时可在独立环境补充：

```bash
python -m venv --system-site-packages .venv-y8n
source .venv-y8n/bin/activate
python -m pip install --no-deps -r experiments/comparison/yolov8n/dependencies-supplemental.lock.txt
```

保留已有 Torch/Torchvision/CUDA；无需 pip 安装 Ultralytics。THOP 是独立可选项；
未安装时 FLOPs 标 NA。正式服务器的其他库是否与开发锁一致仍须 preflight 实录。

```bash
X=experiments/comparison/yolov8n
python "$X/prepare.py"
python "$X/preflight.py" --output "$X/cache/preflight.json"
python "$X/launch.py" --session cmp-y8n --mode train
tmux attach -t cmp-y8n
```

launch 使用当前解释器、独立日志及 tmux。已有会话明确报错，选择新 `--session` 即可。
tee 保留 Python 退出码；会话创建不是训练完成。正式命令和实际提交 SHA 在最终交付消息中提供。
每次 tee 日志独立保存在 cache/logs，只有实际启动的运行在生命周期中关联该日志；
拒绝已有目录时不会覆盖旧日志。打包从生命周期历史读取日志来源，保留真正失败的运行日志。
训练完成后分别执行 evaluate、benchmark、package；每项使用独立 `--result-dir`。

测速为 batch1、可改精度/输入/预热/次数、原生 fuse、CUDA 同步，分别记录前处理/forward/NMS/总延迟。
THOP 在统一 640×640 batch1 下实测，1 MAC=2 FLOPs，记录版本和未覆盖的功能性算子，缺依赖时为 NA。
并发 GPU 任务标 CONTENDED，不能用于主论文测速；不会停止其他任务或轮询等空闲。
参数量同时保存 nc1 unfused 和 fused 精确整数，不引用 COCO80 网上规格。

package 默认严格小于 12,000,000 bytes，保留完整 float32 预测/GT、核心数值/来源、SVG 论文图和固定 GT 选图，
不放权重/数据原图/环境/缓存/重复 PNG。保留服务器原始结果，不用删低置信度预测节省空间。
生成成员/大小/校验清单，ZIP 实际校验，不合格就报错。`--results DIR...` 做协议检查的跨模型摘要汇总，
总包另按严格小于 100,000,000 bytes 检查。
打包包含全部已有恢复请求/有效配置/差异/环境/初始化记录。测速完成后在训练目录记录其实际位置，
自定义测速目录也可打包；其权重及实际评测协议不匹配时单列为 unmatched，不作为本次评测的测速。

## 有限验证

```bash
python experiments/comparison/yolov8n/verify.py --output .development/SMOKE_verification --pretrained-cache /path/to/prepared/cache
```

只在新临时目录创建带 SMOKE 标记的合成数据：CPU FP32 两次真实检测损失优化器更新、
可用 CUDA 的原生 AMP 少量更新、强制增强与关闭日程、冻结 B19 AP/匹配、独立正式 evaluate 重载、
完整 NPZ 导出、少量测速和严格大小打包。它不代表精度或服务器通过。
验证入口明确设置 SMOKE 身份并随运行保存；重载与测速沿用该身份，不依据临时目录名称猜测。
实际验证结论见 `validation.json`，未执行项明确标 UNVERIFIED。
历史四组权重/独立 test 参数及组合源码仍须服务器只读核对；本次不接触这些权重。
