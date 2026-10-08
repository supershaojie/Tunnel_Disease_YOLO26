# b19 + SWR-Fusion

语义引导小波重建融合模块。代码起点为
`4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6`，独立分支为
`exp/swr-fusion-b19`。正式 200 轮训练：**NOT_STARTED**。

## 结构与继承

实验 YAML 将第 14/15 层改为无状态 Identity，第 16 层完整替换为 SWRFusion，
直接读取 `[4,13]`，即浅层 L 与语义 S。640 输入时分别为
128×80×80、128×40×40，输出为 64×80×80。隐藏宽度固定 64，
两个独立 SWRGatedBlock 内部分支为 16/16/32 通道、3/5/7 深度卷积。
Detect 输入保持 `[16,19,22]`，原生 O2M/O2O 及 detach、损失、分配器均未修改。

固定顺序为 Haar `[L0,H1,H2,H3]`、每通道内部的逆变换四相位，以及融合
`[Lp,Sup,R]`。Phi 和三个控制器末层零初始化，其余层正常初始化。
初始 R≈Lp 只描述子带路径，**不表示整网与 b19 等价**。
非法两倍尺寸关系明确报错；不进行补救插值。

`SWRTrainer.get_model` 复用锁定源码的 DetectionTrainer 构造 nc=1 原生参考模型，
从核验过的原始 yolo26n.pt 加载，再在 CPU RNG 隔离范围内构造 SWR。
保留状态逐项复制，核对图连接、模块类型、键、形状、数值和独立存储。
同时审计 CPU、Python、NumPy 和已初始化的 CUDA RNG。正式入口实际走
`setup_model → get_model`，学习后的 checkpoint 加载不会重做原始权重初始化。

| 状态类别                 | tensor 数 | 参数 tensor 数 | 参数元素数 | buffer tensor 数 |
| ------------------------ | --------: | -------------: | ---------: | ---------------: |
| 原始预训练成功迁移且保留 |       552 |            273 |  2,381,816 |              279 |
| 原生 nc80→nc1 适配后保留 |       102 |             66 |     88,070 |               36 |
| 有意移除的原 model.16.\* |        54 |             27 |     34,304 |               27 |
| 新 SWR 初始化状态        |       115 |             64 |     80,008 |               51 |
| 意外遗漏或不一致         |         0 |              0 |          0 |                0 |

654 项保留状态逐项完全一致，构建后的训练 RNG 与独立原生参考路径相同。
完整键清单、buffer 元素数和逐参数梯度由 dry-run / verify 报告提供。

| 参数量       |       b19 |       SWR |    净差 |
| ------------ | --------: | --------: | ------: |
| unfused      | 2,504,190 | 2,549,894 | +45,704 |
| 原生 fuse 后 | 2,375,031 | 2,420,159 | +45,128 |

SWR 的 80,008 参数分解：两个投影各 8,320，context_mix 8,320，
context_dw 1,728，low_delta 4,096，三个门控共 6,984，
三路 fuse 12,416，两个提取块共 29,824。
原生 end-to-end Detect 的 fuse 同时移除 O2M 头，参数统计不限于 BN 折叠。

## 入口与约束

- `train.py`：`--data --weights --project --name --device --baseline-args --dry-run`。
  data、weights、baseline-args 必填，正式名称默认
  `swr_fusion_b19_e200_i640_b32_s42`。支持直接脚本或模块方式运行。
- `b19_recipe.yaml`：来自用户保存的 b19 完整 train_run/args.yaml，
  全字段交叉核对输入的 baseline-args；只允许模型、原始权重、相同数据、
  输出路径和明确设备表示等运行字段变化，不从新版默认值拼装训练配方。
- dry-run 使用临时输出，仅构建、校验环境/数据/继承，不训练、不占用正式目录。
  正式入口拒绝已存在目录、resume、配方漂移和已知服务器环境差异。
  复制或复用核验过的本地 `yolo26n.pt` 供原生 AMP 检查使用，
  `YOLO_AUTOINSTALL=false`。
- 正式训练保留 batch32、imgsz640、AMP、MuSGD。每轮开始使用锁定训练器现有的
  耗尽重试预算，使 OOM 在原生异常分支直接抛出；不创建缩小 batch 的训练流水线。
  原生 AMP 检查失败时，在首个训练 batch 前退出。
- `verify.py`：`--weights --data --baseline-args --device --output`。
  data 与 output 必填；默认原始权重为工作目录的 yolo26n.pt，
  默认配方为随附完整快照。每种精度最多 12 个微批，合计最多 24 个，
  各达到 3 次真实更新即停止；使用 B2、96×128 合成非空标签，
  不把 smoke 设置写回正式配方。
- `validate.py`：`--weights --data --split {val,test} --device --project --name`
  和 `--diagnostic-samples {0,1,2,3,4}`。weights、data、name 必填。
  固定 FP32、640、batch32、conf0.001、IoU0.7、max_det300、rect=True、
  augment=False、quantize=None、原生 end2end/fuse；同时接受 b19 与 SWR 的
  单类 crack checkpoint，输出实际模型/输入 dtype 和源码 SHA。
  AP75 按 IoU 数组的 0.75 定位。原生 P/R/F1 是各自最佳平滑 F1 工作点；
  conf0.25/0.50 的 P/R/F1 明确为曲线插值估计，不冒充精确 TP/FP/FN。
  阈值在 val 确定，再固定用于 test。

数据 YAML 和权重的 SHA256 固定于 train.py；核对 train/val/test 图像数
8414/2404/1202，val/test 标注框数 2985/1477。计数检查只读文件元数据和标签文本。
不重划分数据，不运行其他实验，不安装、升级或卸载环境依赖。

## 本地有限验证

本地环境为 Windows、Python 3.11.15、Torch 2.7.1+cu118、torchvision 0.22.1+cu118、
RTX 2060，安装了 Albumentations。这不等于服务器的 Python 3.12.3、
Torch 2.8.0+cu128、torchvision 0.23.0+cu128、RTX 4090、无 Albumentations 环境。

- FP32 Haar 随机矩形、子块常量、单相位/多通道脉冲、能量、可微性、
  初始重建及分别非零的低频/三个子带检查通过。随机往返最大绝对误差
  2.39e-7，初始 R/Lp 最大绝对误差 1.20e-7。
- B1 640×640 和 320×512 的真实整网接线通过，单模块 6×10 / 3×5 及非法尺寸检查通过。
- CPU FP32：3 个微批、3 次有效 MuSGD 更新。CUDA AMP：10 个微批、
  3 次有效更新、7 次 GradScaler 跳步。所有 64 个新参数 tensor 入原生分组恰好一次；
  全部获得非零任务梯度和有效更新，并与零当前梯度的 decay/momentum 对照区分。
- 已学习状态的 deepcopy、EMA、FP32 保存重载、新进程加载、原生半精度存储保存、
  学习后 Trainer 加载、原生 fuse 均通过；低频、三个门控、两个块仍执行且不重置。
  两组学习后模型均转 CPU FP32 做同设备生命周期数值比较。
  原始 O2O fuse 比较最大绝对误差 3.15e-5，最大稳定相对误差 9.97e-6
  （相对误差分母下限 1e-6，验收 atol/rtol 均为 2e-4）。
- 公共 YOLO.predict 自动融合路径在独立非零状态副本上通过。
  原生 DetectionValidator 的独立 FP32 自动融合路径与训练 EMA 未融合路径也通过单批合成检查；
  各分支执行、实际输入/模型 FP32，参数及剩余 BN buffers 保持不变，门控和低频状态未重置。
  dry-run、已有输出拒绝、配方漂移拒绝、原生 OOM 分支保持 batch32、AP75 定位检查通过。
  NumPy RNG 使用完整二进制状态指纹；改变打印格式不改变指纹，修改被省略显示的状态元素会改变指纹。
- 默认没有诊断开销或持久化激活；显式 diagnostics() 仅在 eval 下提供 gate 均值/分位数、
  低频更新范数比和三路特征范数/有限性。这些统计不证明裂缝增强、背景抑制或 AP 提升。

未验证：服务器精确环境、正式 batch32/640 显存与 AMP 稳定性、200 轮训练、
完整 val/test、速度、ONNX/TensorRT 导出。未做公平延迟基准。
原生 profiler 对 Haar 加减、增益逐元素乘法、重排等算子覆盖不全；
不将未计入的运算记为零成本，也不由参数量推断速度。

## 设计来源

显式 Haar 分解依据用户 RHDWT 结构说明；
本实验没有导入整个模块包或其他实验分支。
[FreqFusion](https://arxiv.org/abs/2408.12879) 及其
[作者代码](https://github.com/Linwei-Chen/FreqFusion) 提供频率信息分工的参考，
未照搬其自适应滤波与偏移重采样。
[MogaNet（ICLR 2024）](https://github.com/Westlake-AI/MogaNet) 提供多感受野与门控聚合思想；
SWRGatedBlock 是本文固定规格的自定义块，不是论文完整机制复刻。
历史 MPDF 已使用 Haar；本实验不把小波本身称为新贡献，不承诺超过 b19 或 MPDF。
