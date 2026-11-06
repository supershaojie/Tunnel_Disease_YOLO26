# RT-DETR-R18：隧道裂缝 B19 对比实验

论文模型登记为 **RT-DETR-R18（LCBR 项目 C2）**。后端来自
<https://github.com/supershaojie/concrete-crack-rtdetr> 的
`exp-rtdetr-r18-lite-paperalign`，固定提交为
`67c3078e54a657fd96d65fee657a75fbb1dae0d6`。
模型文件保留原名 `ultralytics/cfg/models/rt-detr/rtdetr-resnet18-lite.yaml`。
`vendor/ultralytics_source/` 保存该提交的 Python 包和许可证；来源与实际文件摘要见 `backend.lock.json`。
实际构建检查固定 C2 图结构，禁止 LIF-Down、CBR、DTR-C2PSA、SCE-Fusion。

## 初始化与训练

只使用 `rtdetr_r18_lite_imagenet_backbone_init.pt`，SHA256 为
`fe8501bbcc1d1d5b366bdb10fd47d0fbb91edff1f9558f39e31683a550bcad8e`。
这是 `nc=80, epoch=-1` 的完整 checkpoint：骨干采用 ImageNet 初始化，Neck/Encoder/Decoder
保存原始随机初值。原生模型构造加载全部同名同形状态，类别相关层按 `nc=1` 适配；
`initialization.json` 记录实际加载数量与类别层变化。
已有 YOLO 对比使用各自 COCO 检测预训练，两者初始化条件不同。
权重不进入 Git。

`prepare.py --source-weights FILE` 可将原文件复制到公共缓存；缓存已存在则校验后复用。
默认缓存为
`/root/autodl-tmp/projects/Tunnel_Disease_YOLO26/weights/pretrained/rtdetr_r18_lite_imagenet_backbone_init.pt`。
缺文件或摘要不同会明确报错。

默认配置为 `configs/default.yaml`：200 轮上限、早停 patience=50、640、batch=32、workers=8、GPU0、
seed=42、AMP、deterministic；优化器沿用旧 C2 的 AdamW 配方：lr0=0.0005、lrf=0.01、
momentum=0.937、weight_decay=0.0001、cos_lr=true、5 轮 warmup、nbs=64。
保留原生梯度累积、参数组、RT-DETR 损失、查询、去噪、匹配与 EMA。
默认训练 `rect=false`。显式覆盖可使用 `--set train.batch=32` 等叶子字段；解析后配置与实际优化器参数组均保存。
本入口从原始初始化启动，不提供 resume；重新实验使用新的 `--name`。

数据使用母项目的
`datasets/Tunnel_Crack_AugFirst_Diverse5x_RandomSplit_7_2_1_seed42/data.yaml`，
保持 train/val/test=8414/2404/1202 的现成划分。
增强读取母项目 B19 真实 `args.yaml`，直接复用 `shared/provider.py` 与 `shared/b19/`，
最后 10 轮沿用关闭混合增强的日程。
公共数据适配采用保持纵横比的图像加载和 resize，替代 RT-DETR 原有提前拉伸及 `stretch=True` 增强入口，
公共增强只执行一次。

每个进程只导入本实验私有后端。
私有修改删除了原生 CUDA OOM 自动缩小 batch 并重启的分支；真实 OOM 保留原始异常，显式 batch、尺寸和 AMP 不自动改变。
CUDA AMP 检查在实际 RT-DETR 上执行有限 autocast forward，替代下载无关 YOLO 权重的检查路径。
其他后端改动与来源摘要以 `backend.lock.json` 为准。

## 独立评测与产物

训练每轮只使用 val，并以公共 val mAP50–95 选择 best。独立评测由新进程加载 best，默认
FP32、imgsz640、batch32、conf0.001、max_det300、无 TTA、原生融合。
沿用公共 LetterBox/rect 预处理，归一化 xywh 先乘实际输入张量宽高，再撤销 resize/padding。
保留原生查询概率，无额外 sigmoid、objectness 或 NMS；公共配置中的 IoU 字段不作为有效 NMS 参数记录。

`evaluate.py --split val` 冻结验证阈值；随后
`evaluate.py --split test --val-metrics VAL_METRICS_JSON` 对同一 checkpoint 独立 test。
主表 `summary.csv` 使用 `legacy_curve_peak`：P/R/F1 来自同一平滑 F1 最大点。
`val_frozen` 结果在指标 JSON 中独立命名。AP 始终根据收集到的完整低置信度预测计算。
完整预测、GT 与图像索引分别保存到 `predictions.npz`、`ground_truth.npz`，曲线数值保存为 `curves.npz`。

`complexity.py` 在实际 nc=1、640 输入的原生融合推理模型上重新计算参数量与 THOP 计算量，明确标注未覆盖的函数算子。
原生 `fuse()` 保留 ResNet `ConvNormLayer` 中的 BatchNorm；结果如实记录剩余模块和 `is_fused()` 状态。
`benchmark.py` 单独记录同步推理时间及 GPU 上其他计算进程；并发训练时间不能用于正式 FPS。
训练允许 GPU0 上多个实验并行，不设独占锁或等待队列。

默认 run 为
`/root/autodl-tmp/projects/Tunnel_Disease_YOLO26/runs/comparison/rtdetr_r18/compare_rtdetr_r18_b19_e200_i640_b32_adamw_s42/`。
其下主要目录和文件为：

- `weights/best.pt`、`weights/last.pt`：完整权重留服务器。
- `run_config.json`、`effective_config.json`、`results.csv`、`train_status.json`、`training_summary.json`：配置和训练完成状态。
- `evaluate_val/`、`evaluate_test/`：独立评测指标、曲线、完整预测与少量预览。
- `complexity/`、`benchmark/`：复杂度与单独测速结果。
- `rtdetr_r18_core.zip`：`package.py` 复用现有白名单打包并输出实际路径、字节数和摘要。

精简包保留指标、配置、训练结果和状态、曲线数值、完整 test 预测与索引、源码及初始化来源、少量预览；
默认排除权重、数据集、环境、vendor 和每轮 checkpoint。
大小限制只触发明确报错，不删除完整预测以满足阈值。

## 运行环境与验证范围

服务器使用独立 `.venv-r18`，通过 `venv --system-site-packages` 复用宿主
Python3.12.3、Torch2.8.0+cu128、Torchvision0.23.0+cu128；`requirements.txt` 只补充运行依赖（包含 SciPy）。
不向母环境安装另一份 ultralytics，不降级 Torch。
训练在交互式 `cmp-r18` tmux 内直接执行 Python；保持 TTY，原生进度条每轮原地刷新，轮结束换行。
日志可用 `tmux pipe-pane` 保存，不在训练命令外接 `tee`。

开发验证限于实际 YAML/初始化加载、少量真实检测损失更新、保存后新进程重载和公共评测形状的推理与坐标还原。
开发结果以交付时记录为准；本次开发不启动服务器 200 轮训练或全量 test，服务器正式指标由后续命令产生。
实际有限开发验证见 `validation.json`；原生 deterministic 设置采用 warn-only，CUDA grid_sample 反向仍可能非确定。
固定最终提交的完整服务器命令在推送后于仓库外生成的 `SERVER_COMMANDS.md` 交付。
