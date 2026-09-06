# YOLO26n / b19 / D1 SIR-SPPF v1

## 实验身份与边界

- 分支：`exp-yolo26n-b19-sir-sppf-v1`，从已归档 b19 提交 `4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6` 建立。
- 远端：`https://github.com/supershaojie/Tunnel_Disease_YOLO26.git`。
- 本地工作树：`E:/PycharmProjects/Tunnel_Disease_YOLO26/.worktrees/exp-yolo26n-b19-sir-sppf-v1`。
- 服务器工作树：`/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_SIRSPPF_v1`。
- 正式运行名：`yolo26n_b19_d1_sir_sppf_v1`；tmux：`y26_sir_v1`。
- 仅新增本轮实验；原 b19 YAML、SPPF、训练框架、数据划分和 DCR v1/v2 工作树保持原样。未重跑 baseline，未安排内部消融或多种子。

## 实际阅读的来源

任务来源为用户明确授权执行的 `E:/ditieyolo26跑结果/8.12离线在线/b19  SIR-SPPF v1/Codex_YOLO26n_b19_SIRSPPF_v1.md`。
实际读取 `D:/7.21yolo26改/YOLO26缝合/` 下的以下三个源文件，而非执行模块包或整体覆盖项目：

| 相对路径                                                            | SHA-256                                                            |
| ------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `ultralytics/nn/newsAddmodules/MSAttention_ICCV2025.py`             | `24de21a29bde9daa559ece29a9ff694d6da5544575840388e6831a4e37c1f1a8` |
| `ultralytics/cfg/models/add26/yolo26_MultiScaleAttention.yaml`      | `26d9ee52d30987098f8e5562f21eb52ad86622d4fe5f07b4e3269f2c35773469` |
| `ultralytics/cfg/models/add26/yolo26_C3k2_MultiScaleAttention.yaml` | `ac64455c2e02c94a0211287c06095e92368cecaef3f6076da00b829d2ccc4c7c` |

PY 标注 arXiv/ICCV 2025，两个 YAML 标注“来自:Ai缝合怪 改进”。该包适配不能视为论文作者官方 YOLO 实现。
已阅读 [M2SFormer 原文 §3.2](https://arxiv.org/html/2506.20922v1)；保留用户给出的
[ICCV 条目](https://openaccess.thecvf.com/content/ICCV2025/html/Nam_M2SFormer_Multi-Spectral_and_Multi-Scale_Attention_with_Edge-Aware_Difficulty_Guidance_for_ICCV_2025_paper.html)，后者本次网页访问失败。

原文的多尺度部分使用降采样、膨胀卷积和可学习的前景/背景权重；整个网络还包含多谱 DCT 与难度引导解码器。
模块包使用并行 AvgPool/卷积分支、上采样与相加，没有移植整个原文网络。
`Gate_Function` 中 `att*x + (1-att)*x` 在数学上抵消成 `x`，后续卷积/ReLU 仍有效，但这一注意力选择无效。
包内一个 YAML 在 head 末端增加注意力，另一个替换多个 C3k2；本次均不沿用这些接入位置。

SIR 只借鉴多尺度空间选择思想，按用户给出的固定数学定义独立实现。它联合观察串行最大池化的三个增量，
以 `0.5*tanh(L_i)` 修正增量，再累计到原始金字塔。没有引入 DCT、额外监督、大卷积分支或原包依赖链。
修复包内抵消门控不作为创新；池化增量不称为真实裂缝边缘或已验证噪声，效果等待真实训练与评估。

**后续本项目模块改动约定：先读取对应模块包的 PY/forward 和模型 YAML，再记录来源、差异与改动；不可访问时明确标注。**

运行代码参考已读取的 `exp-yolo26n-b19-dcrstrip-v2` 工作树提交 `f99a90d` 中
`run_b19_dcrstrip.py`、`run_b19_dcrstrip_v2.py`、`server_b19_dcrstrip_v2.sh` 和 `finish_b19_dcrstrip_v2.py`。
复用了原生 Trainer 重建、配方比较、权重审计、输出认领、子进程预检与打包校验逻辑；去掉 DCR 的层号、分支参数、标量回调和历史依赖。
SIR 的入口完全在自己的分支内，不依赖邻接工作树。

## 架构与公式

从原 b19 `ultralytics/cfg/models/26/yolo26.yaml` 复制模型 YAML，唯一图结构替换为：

```yaml
- [-1, 1, SPPF_SIR, [1024, 5, 3, True]] # 9
```

`nc: 80` 留在基础配置，由原生 Trainer 根据数据适配为 1。nano scale 由文件名正确解析。
第 4 层仍为 C3k2，第 10 层 C2PSA 与 Detect 输入 `[16,19,22]`、深度/通道、`end2end`、`reg_max` 全部不变。
`SPPF_SIR` 继承 SPPF，保留 `cv1`（含 `act=False`）、`cv2`、`m`、`n`、`add` 和主路权重名称。
注册于 `base_modules`，未加入 `repeat_modules`。

先算原始 `Z_i=MaxPool(Z_(i-1))` 和 `D_i=Z_i-Z_(i-1)`，然后计算：

```text
L = Conv1x1(16→3c)(SiLU(DWConv3x3(SiLU(Conv1x1(4c→16)(concat(Z0,D1,D2,D3))))))
correction_i = 0.5 * sum(tanh(L_j) * D_j, j=1..i)
Zhat_i = Z_i + correction_i
Y = cv2(concat(Z0,Zhat1,Zhat2,Zhat3)) + X  # 本轮启用原 shortcut
```

三个 router 卷积均有 bias，没有 BN。最后卷积权重和 bias 为零，新分支构造位于 CPU `fork_rng(devices=[])` 内。
无 detach、原始特征 in-place 修改或全零跳过逻辑。每级增量系数数学上在 `(0.5,1.5)`，浮点可饱和到端点。

| 检查                | 实测                                                  |
| ------------------- | ----------------------------------------------------- |
| 单类原模型参数      | 2,504,190                                             |
| 新 router 参数      | 14,896                                                |
| 单类 SIR 未融合参数 | 2,519,086                                             |
| 640 输入第 9 层     | 输入、输出均为 `[B,256,20,20]`，隐藏通道 128          |
| 全模型输出          | 零路由时，640×640 和 640×960 的完整原始输出逐元素相同 |

## 配方、预训练和数据证据

实际读取本地原归档：
`E:/ditieyolo26跑结果/8.12离线在线/b19 200e/b19_yolo26n_e200_train_val_test_20260823_224153/train_run/args.yaml`。
完整配方快照保存在 `tools/experiments/b19_reference.json`，服务器优先读取原 run 的 args。
原文件缺失时恢复这一有来源的快照，并明确写入 `args_source`；不会猜配置或要求重跑 b19。
历史启动文本来自本次用户附件第 9 节，保存在 `b19_launcher_expanded.txt`，仅作核对证据。

原始初始化文件为 baseline 根目录的 `yolo26n.pt`，固定 SHA-256：
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`，本地已独立核实。
原生 AMP 检查需要 cwd 下同名文件，runner 只用这份已核验原始文件填充 SIR 工作树缓存，不另下载权重。

继承 epochs=200、patience=60、batch=32、imgsz=640、workers=8、seed=42、MuSGD、AMP、cos_lr 和完整在线增强。
`get_cfg` 生效配置只允许模型、等价初始化表示、输出目录与有记录的路径迁移差异。
原始 args/launcher、副本生效配置、逐字段差异、初始权重摘要、源码/环境/数据清单指纹都随运行保存。

本地核对实际 train/val/test 为 8414/2404/1202 张，目标数量分别为 10243/2985/1477。
保持既有增强后按文件随机划分；未改变 split 或重新增强。
清单指纹包括稳定图片路径/大小/mtime 与标签内容摘要，另保存既有 metadata 文件摘要；不是声称重新逐字节校验所有大图像。

共同初始化在 `DetectionTrainer.get_model` 内比较：原模型 708 个参数/buffer 项逐张量相同，606 个兼容预训练项均已加载。
共同不匹配来自 COCO 80 类向 crack 1 类适配，详见 `local_weight_audit.json`。新增项只能出现在 `model.9.router.*`。
正式 `_setup_train` 后、第一 batch 前再次核对共同张量、router 末层零值、可训练状态、EMA 和最终 MuSGD 分组；不取 batch、不更新 BN、不推进正式 RNG。

固定 b19 的训练循环没有可覆写 OOM handler。SIR 专用 Trainer 在该循环请求增加 `_oom_retries` 时原样抛回活动异常，
位置早于 batch/args 修改；正常零值重置沿用原行为。没有复制训练循环或改全局 Trainer，已在真实原生 catch 边界注入 OOM 验证。

## 已执行验证与尚未完成事项

本地环境：Windows，Python 3.11.15、PyTorch 2.7.1+cu118、Ultralytics 8.4.98、RTX 2060 6 GB。
11 项定向测试通过：形状/累计公式/非负增量、零初始化与 RNG、完整图、真实权重 Trainer、真实裂缝样本 AMP 梯度、
FP16 EMA 保存后新进程 FP32 加载/fuse/预测、原生最终设置/OOM/审计 RNG、历史快照与原生 Model.val 单次调用契约，以及干净子进程中的原 CLI 线程初始化。
融合时原生 YOLO26 删除 one-to-many 分支，因此比较保留的原始 one-to-one 张量和解码输出（FP32 atol/rtol=1e-4）；
保存前后未融合原始输出要求逐元素相同。

真实数据梯度检查仅用 2 张真实样本、640 输入、3 次可丢弃更新，未跑训练 epoch。
第一步 router 前两层梯度为零、最后层非零，随后前两层梯度非零；没有用 weight decay 的参数变化代替梯度证据。
最终优化器审计在独立 CPU setup 中通过，6 个 router 参数张量全部进入 MuSGD；这是本地检查，不冒充服务器 batch=32 结果。

报告在工作树 `runs/sir_development/` 与 `runs/detect/yolo26n_b19_d1_sir_sppf_v1_preflight/`。
正式规格 preflight 已在本地实际调用，因 Python/PyTorch/GPU 不符且显存余量不足停止，记录 `missing.json`，未写 SIR `passed.json`。
当前未发现可用 SSH 配置或活动 SSH 连接，**正式训练尚未启动**，没有真实 SIR val/test 指标或训练后的诊断。

服务器 preflight 使用完整原生 `_setup_train`，同 batch=32/增强/MuSGD，执行 3 次真实 AMP 前后向更新。
这 3 步采用 unit-scale backward 观察有限梯度，避免把 GradScaler 初始溢出探测误判为断路；正式训练仍完整保留原生 GradScaler。
首轮 warmup lr/momentum 与原生前三步对应。预检在短生命周期子进程中退出，正式 train 重新初始化所有模型、BN、EMA、优化器和 loader。
同源码/配置/环境/数据/权重指纹可复用 SIR 自己的通过记录。并行 GPU 任务只记录，不终止；显存预留至少 8 GiB 才尝试真实预检。

## 服务器操作接口

首次部署请使用交付回复中固定到实际提交的连续 fetch/worktree/tmux 命令。已有路径必须核对工作树、分支、HEAD 和干净状态；不 reset 或自动覆盖。
包装器从自身位置解析工作树、设置 PYTHONPATH，从而导入本轮代码；默认 Python 是原记录的 `/root/miniconda3/bin/python`，可用 `B19_PYTHON` 指向已确认原环境。

```bash
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_SIRSPPF_v1
bash "$WORK/tools/experiments/server_b19_sir_sppf_v1.sh" preflight
bash "$WORK/tools/experiments/server_b19_sir_sppf_v1.sh" train
```

`train` 自动完成缺失的有效预检，再启动唯一正式运行。以上两个命令无需重复执行预检；日常只用 train 即可。
运行锁只针对 SIR；正式 `exist_ok=False`，输出路径原子认领，不自动产生 2/3 后缀。
Python 与 tee 退出码分别记录，失败均产生非零状态；完整入口日志为 `${NAME}_train.console.log`，正式训练日志为 run 内 `train.log`。
实际训练成功应看到 epoch/batch/loss 日志，创建 tmux 或预检通过均不等于训练已经启动。

训练结束后：

```bash
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_SIRSPPF_v1
bash "$WORK/tools/experiments/server_b19_sir_sppf_v1.sh" test
bash "$WORK/tools/experiments/server_b19_sir_sppf_v1.sh" diagnose
bash "$WORK/tools/experiments/server_b19_sir_sppf_v1.sh" package
```

后处理要求 `${NAME}_train.exit_status` 为 0、完成记录匹配实际 run 和 best.pt 摘要。
test 只调用一次 `YOLO.val(split=test)`，FP32、batch=32、imgsz=640、conf=0.001、iou=0.7、max_det=300、rect=True、augment=False，保留 end2end。
同次调用保存精确 P/R/mAP/AP75、速度、样本数、JSON（即使检测为空）、曲线、混淆矩阵和生效参数；成功记录相同可复用，冲突不覆盖。
diagnose 使用稳定排序前 16 张真实 val 图，报告三个尺度 tanh/系数统计、`abs(tanh)>=0.99` 饱和比例、累计修正范数比例和整体模块改变量。
这是同一已训练模块的特征诊断，不是重新训练的消融。

package 只整理已有结果，要求 test/diagnose 已完成。包含 best.pt、核心训练/评估/诊断日志、初始化/优化器/来源、配置、
所需自定义源码、Git 补丁和 source.tar，排除数据集、模块包、环境、.git、last.pt/epoch\*.pt 和预检权重。
150 MiB 以上列出大文件并精简可选 batch 图，核心证据仍超限时停止。归档逐文件校验 SHA-256 并验证 gzip CRC，生成旁路 `.sha256`。
自定义 best.pt 需本分支代码与对应提交，不保证任意原生 Ultralytics 可单文件加载。

入口先导入 Ultralytics，再导入 torch，复用原 CLI 的 OMP 初始化；环境报告记录 OMP 和实际 torch 线程数。test 从 on_val_end 捕获实际输出目录和原生回写的精度参数，不假设 Model/Validator 持有不存在的 backend 属性。

另已通过模拟打包契约检查：归档文件摘要和 gzip CRC 全部一致，last/epoch 权重被排除，原始字节源码补丁通过 git apply --check 验证；该模拟归档只验证程序行为，不是训练结果。草稿 PR #3 的 GitHub API 未返回任何 workflows/check-runs/自动审阅，因此没有将外部自动检查记为通过。
