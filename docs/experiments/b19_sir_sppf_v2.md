# YOLO26n / b19 / SIR-SPPF v2

本轮按用户完整数学定义实施一个确定版本，属于 SIR 的版本迭代。仅验证“逐尺度独立修正减少同一修正量重复传播”的结构假设，不声称已证明有效或一定提升指标。

## 实验身份

- 起点：已验证且包含保存重载修复的 `4c3648b45481e59cc5dc24235f6bdcf5664e96a5`。
- 分支：`exp-yolo26n-b19-sir-sppf-v2`。
- 本地：`E:/PycharmProjects/Tunnel_Disease_YOLO26/.worktrees/exp-yolo26n-b19-sir-sppf-v2`。
- 服务器：`/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_SIRSPPF_v2`。
- 运行：`runs/detect/yolo26n_b19_d1_sir_sppf_v2/`；tmux：`y26_sir_v2`。
- 历史 b19、SIR v1、DCR v1/v2 的代码、权重与结果保持原样；没有使用 RT-DETR 项目的文件。

## 实际读取的参考与沿用内容

本次直接访问了 `D:/7.21yolo26改/YOLO26缝合/` 中以下三个文件，逐个读取并计算摘要。没有解压 ZIP 或扫描整个模块包。

| 相对路径                                                            | SHA256                                                             |
| ------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `ultralytics/nn/newsAddmodules/MSAttention_ICCV2025.py`             | `24de21a29bde9daa559ece29a9ff694d6da5544575840388e6831a4e37c1f1a8` |
| `ultralytics/cfg/models/add26/yolo26_MultiScaleAttention.yaml`      | `26d9ee52d30987098f8e5562f21eb52ad86622d4fe5f07b4e3269f2c35773469` |
| `ultralytics/cfg/models/add26/yolo26_C3k2_MultiScaleAttention.yaml` | `ac64455c2e02c94a0211287c06095e92368cecaef3f6076da00b829d2ccc4c7c` |

包内实现是并行池化/卷积分支与融合，接入位置包含 head 或多个 C3k2，均未移植到本轮。`att*x+(1-att)*x` 门控相消的问题不作为创新点。本轮未重新访问论文网页，不把 v1 文档中的论文阅读记录冒充为本次直接阅读。

实际读取仓库文件：

- `ultralytics/nn/modules/sir_sppf.py`、`block.py` 中原生 SPPF、模块导出及 `tasks.py` 的模型解析与检查点加载。
- `ultralytics/cfg/models/26/yolo26n-sir-sppf-v1.yaml`、原 b19 `yolo26.yaml`。
- `tools/experiments/run_b19_sir_sppf.py`、`finish_b19_sir_sppf.py`、`server_b19_sir_sppf_v1.sh`。
- `tests/test_sir_sppf.py`、`docs/experiments/b19_sir_sppf_v1.md`、`b19_sir_sppf_reload_fix.md`。
- `tools/experiments/b19_reference.json`、`b19_launcher_expanded.txt`。
- 当前 `DetectionValidator`、`BaseValidator`、`ConfusionMatrix.process_batch`，核对 FP32 接口及混淆矩阵实际阈值。

沿用 v1 构造函数、router、参数命名、RNG 隔离、原生 Trainer 重建、完整 b19 配方比较、权重加载审计、MuSGD/EMA 审计、固定 batch OOM 退出以及已修复的保存重载上下文。没有复制训练循环或使用运行时全局 monkey patch。

自主改动：新增独立 forward 与单类 YAML；给共享入口增加显式 model/trainer/entrypoint/source/check 参数；新增 v2 的三入口、独立修正检查、FP32 val/test 报告、对应诊断与结果选择索引。去掉共享入口写死的 v1 架构断言与调用点，改由各版本显式提供。历史 `SPPF_SIR.forward` 的字节与语义不变。

## 结构与初始化

仅 backbone 第 9 层从 `SPPF` 改为：

```yaml
[-1, 1, SPPF_SIR_V2, [1024, 5, 3, True]]
```

YAML 显式声明 `nc: 1`，文件名解析为 nano。第 4 层仍为 C3k2，第 10 层仍为 C2PSA；Detect 输入、深度/宽度、Neck 与损失均保持 b19 行为。只在 `base_modules` 注册，不加入 `repeat_modules`，池化次数仍为 3。

```text
z0 = cv1(x); zi = MaxPool(z(i-1))
di = zi - z(i-1)
L1,L2,L3 = chunk(router(concat(z0,d1,d2,d3)), 3, channel)
ri = 0.5*tanh(Li)*di
corrected = [z0,z1+r1,z2+r2,z3+r3]
y = cv2(concat(corrected)); output = y+x if self.add else y
```

后一级池化只读取原始特征；增量与 router 输入不含已修正特征。Router 仍为 1×1 Conv → SiLU → 3×3 DWConv → SiLU → 1×1 Conv，隐藏通道 16，末层 weight/bias 为零；构造继承 v1 的 `fork_rng(devices=[])`。

单类原模型参数 2,504,190；v2 未融合 2,519,086；新增 14,896，与 v1 相同。完整模型 640×640 与 640×960 零修正输出要求逐元素相等。v2 原生 Trainer 另外比较 v1/v2 全部参数与 buffers，包括 router。

## 配方和运行约束

从 baseline 根目录 `yolo26n.pt` 开始，固定 SHA256：
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`。
不从任何实验 best/last 权重训练。原始 args/launcher 优先从服务器指定位置读取，缺失时只恢复仓库中有来源的 b19 快照并明确标注；所有配置逐字段比较，未知新默认值停止。

保持 Python 3.12.3、PyTorch 2.8.0+cu128、Ultralytics 8.4.98、GPU 0 和 b19 原环境；不升级依赖。
训练仍是 epochs=200、patience=60、batch=32、imgsz=640、workers=8、seed=42、deterministic=True、AMP、MuSGD、完整 b19 增强、余弦学习率和损失配置。不降 batch、不续训、不运行消融矩阵。

`train` 先执行必要预检；指纹覆盖 v2 入口、共享入口、全部模型源码/YAML、初始权重、配方、环境及数据清单。预检在子进程中进行 3 次真实 batch=32、640、增强 AMP/MuSGD 更新，正式训练重新初始化，不继承检查模型、BN、RNG 或优化器。

首步 router 前层梯度为零符合末层零初始化；少量更新后应非零。保存时同一 FP16 EMA 快照形成独立 FP32 参考；新进程统一 CPU 1 线程、FP32 和后端状态。714 项参数/buffer 精确比较与输出比较分开，未融合输出仍为零容差；融合 one2one/解码输出仍使用原先 `1e-4/1e-4`，没有放宽。

包装器对全部 stage 使用同一 v2 专属锁，原子认领正式输出。每次调用有独立 `*.attempt.*` 日志与状态，当前状态文件保留 Python/tee 退出码；旧失败日志保留。正式完成同时需要成功训练退出状态和匹配 best.pt 的 `completed.json`。预检通过不等于训练完成。

## 评估、诊断和打包

```bash
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_SIRSPPF_v2
bash "$WORK/tools/experiments/server_b19_sir_sppf_v2.sh" train
```

部署须使用交付回复中固定到实际推送提交的 refspec/worktree 命令。在交互 tmux bash 内运行，发送训练命令前设 `remain-on-exit on`，训练退出后保留窗口。没有授权且可用的服务器连接时，只完成本地实现与推送。

```bash
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_SIRSPPF_v2
bash "$WORK/tools/experiments/server_b19_sir_sppf_v2.sh" test
bash "$WORK/tools/experiments/server_b19_sir_sppf_v2.sh" diagnose
bash "$WORK/tools/experiments/server_b19_sir_sppf_v2.sh" package
```

`test` 对验证集选出的 best.pt 各运行一次独立 FP32 val 和 test。训练内验证沿用 b19；独立验证不使用训练末次指标替代。固定 imgsz=640、batch=32、workers=8、device=0、conf=0.001、iou=0.7、max_det=300、rect=True、augment=False、quantize=None。不得用 test 选择 epoch、阈值或结构。

val 必须为 2404 张/2985 目标，test 为 1202 张/1477 目标；保存完整精度 P/R/AP50/AP75/mAP50-95 和 0.50–0.95 全部 AP。JSON 以 0–1 为单位，不四舍五入。保存实际配置、数量、权重/data/代码摘要、提交、PR 曲线与混淆矩阵。
当前原生混淆矩阵使用 conf=0.25、匹配 IoU=0.45：传入默认检测 conf=0.001 会映射为 0.25；这与 AP 的 conf/iou 设置不同，代码与行为测试均核对。

相同证据复用有效报告，不同或失败的报告保留在独立目录；`evaluation.json` 指向当前 val/test。诊断固定排序前 16 张 val 图并记录标识及字节摘要。修正范数按 `ri=0.5*tanh(Li)*di` 单独计算，绝不累计；系数是相对于原始前一级特征的增量系数，不解释成已修正相邻特征之差。

打包只整理已完成的匹配评估与诊断，验证索引摘要、核心图表和原生保存证据。包含 best.pt、args/results、核心曲线、当前 val/test、诊断、日志/状态、环境、源码清单、source.patch 和提交源码 source.tar。排除数据、模块库、环境、.git、last/epoch 权重和旧评估大图；最多保留 8 张可选 batch 图。原文件不删除。归档逐文件 SHA256/gzip CRC 验证，输出实际路径、字节大小和 SHA256。

## 实验依据与结果边界

以下为用户提供的已完成历史结果，单位百分数，不是本次重跑或 v2 结果：

| 模型/集合           | P      | R      | AP50   | AP75   | mAP50-95 |
| ------------------- | ------ | ------ | ------ | ------ | -------- |
| b19 val             | 86.328 | 78.861 | 86.080 | 52.481 | 50.265   |
| SIR v1 val          | 88.122 | 79.532 | 87.655 | 53.876 | 50.805   |
| b19 test            | 87.459 | 78.379 | 87.006 | 53.675 | 50.777   |
| SIR v1 test         | 85.751 | 80.027 | 87.648 | 51.185 | 49.972   |
| SIR v1 关闭分支 val | 84.205 | 74.472 | 83.691 | 46.773 | 46.676   |

关闭分支导致下降说明当前已训练 v1 依赖分支，不能当作相对 baseline 的独立增益，也不能确定 test 下降原因。

本地 24 项定向及回归测试均通过（初次回归 22 项通过，修复后对打包、Bash 路径与退出码完成针对性复测），包括 v1 兼容、v2 公式/初始化/完整图、真实预训练权重接入、2 张真实裂缝图的 3 次 640 AMP 更新、MuSGD/EMA、新进程保存重载/fuse/预测、篡改状态拒绝、原生 OOM、评估复用/损坏记录、轻量归档和 Python/tee 退出码。Windows 归档成员名已统一为 POSIX 分隔符，归档 SHA256/gzip CRC 验证通过。独立 Ruff 0.11.13、固定 Prettier 3.6.2、Bash 语法检查通过，训练环境依赖未升级。

本地环境为 Windows、Python 3.11.15、PyTorch 2.7.1+cu118、RTX 2060 6 GB。`runs/sir_v2_development/` 保存检查证据，其中 `formal_preflight=False`。新进程 714 项状态精确相等，未融合 raw 最大误差 0；融合最大绝对差 `9.1552734375e-5`，保留原容差。第三步全部 6 个 Router 参数张量有非零梯度。

正式规格 preflight 在本地实际调用并返回 2：Python、PyTorch、GPU 与原环境不符，显存余量约 4.99 GiB，小于原检查要求的 8 GiB。`runs/detect/yolo26n_b19_d1_sir_sppf_v2_preflight/missing.json` 记录真实原因，未生成任何 `passed.json` 或正式训练目录。未发现 SSH 配置或活动 SSH 连接，正式训练未启动。

服务器 batch=32 正式规格预检、200 epoch 训练、真实 FP32 val/test、训练后诊断与结果包尚需服务器执行。本地没有把缩小 batch 的检查记为服务器通过，也没有生成或捏造 v2 正式指标。
