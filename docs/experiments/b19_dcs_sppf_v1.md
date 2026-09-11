# DCS-SPPF v1 实施与审计报告

本次完成独立 DCS-SPPF v1 实现、分层验证和服务器工作流。正式训练状态为 **NOT STARTED**，没有启动 AutoDL 训练，没有 DCS 的正式 val/test 精度结论。严格保留附加指令规定的结构及 b19 参数；不创建 PR。

## 基线与源码来源

- Branch：`codex/exp-yolo26n-b19-dcs-sppf-v1`。
- Base SHA：`4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6`。
- 原始 b19 归档：`E:/ditieyolo26跑结果/8.12离线在线/b19 200e/b19_yolo26n_e200_train_val_test_20260823_224153`。
- 归档 `environment/git_state.txt` 直接记录上述 SHA；SIR/NDP/SICR/CCA 等近期实验历史也可回溯到该提交。
- 当前分支通过 `git worktree add` 直接从该 SHA 创建，没有承接任何创新分支的模型代码或参数。
- 原始 `args.yaml` SHA-256：`b08b915756bf85c91d3356586a651a37156d71867b84b6e8493d75c9568642b9`。
- 原始 `results.csv` SHA-256：`44795068d0ebc687c29ff7b7b8cbf337469546b2d4dd515cb0616c270bdf9507`。
- 初始化 `yolo26n.pt` SHA-256：`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`。只接受与该原始文件逐字节一致的权重。
- b19 比较用 best.pt SHA-256：`d0b2ca5a5d30de9ed002c64c9238b182ddeccce644c055ae5f2dba566878bd5e`；它只用于训练完成后的比较，不用于初始化。

原始 b19 使用 native DetectionTrainer、MuSGD、200 epochs，best.pt 由验证集选择，再独立执行 Test。归档包括 train_run、test_run、config、environment 和日志；没有原始 shell 文件。沿用成熟 b19 工作流中的展开 launcher 记录，并逐项验证它能重建原 args，明确区分历史展开记录和原始 shell。

复用并审计了 SICR `d6e9009` 中的 `b19_common.py`：原参数/数据清单/初始权重哈希校验、逐 tensor 对齐、源码指纹和 launcher 审计；删除其未使用的推理属性辅助代码，替换实验身份。打包沿用 CCA v2 `05c79c6` 的规范证据选择与逐文件哈希验证方式，以及后续 SICR 工作流中的目录预剪枝、best/last 双权重修复。未引入任何其它创新模块。

## 既往 SPPF 审计

以下为同条件 FP32 复核或注明的归档数值，单位为百分数；SIR v1 Test 为既有记录，不冒充本次重跑。

| 模型    | Val mAP50-95 | Val AP75 | Val Recall | Test mAP50-95 | Test AP75 | Test Recall |
| ------- | -----------: | -------: | ---------: | ------------: | --------: | ----------: |
| b19     |      50.2648 |  52.4806 |    78.8610 |       50.7770 |   53.6748 |     78.3792 |
| SIR v1  |      50.8045 |  53.8757 |    79.5320 |       49.9717 |   51.1855 |     80.0271 |
| SIR v2  |      50.4030 |  52.3672 |    78.0718 |       50.6947 |   53.1789 |     77.9959 |
| NDP v1  |      49.9188 |  51.8050 |    78.1851 |       49.7944 |   52.1134 |     76.2356 |
| SICR v1 | 无正式结果包 |       无 |         无 |            无 |        无 |          无 |

SIR v1 的累计 router 修正带来明显表示扰动；第一张固定样本的模块变化 norm ratio 为 0.24172，第一尺度 tanh 饱和比例约 23.49%。Val 增益没有转化为 Test 定位增益：Test AP75 比 b19 低 2.4893 个百分点。SIR v2 独立尺度修正的同一样本 ratio 降到 0.16559，但 Test mAP50-95/AP75/Recall 仍均略低于 b19。这些观测支持避免将 router 激活或 Val 改善视为 Test 成功，不能单凭它们证明因果机制。

NDP 的标准化 softmax 分布聚合不是前景置信度；已读取 unfold、valid-mask、均值/方差、tanh logits 与投影残差实现。固定样本的 5/9 窗口归一化熵约为 0.827/0.873，聚合较分散，同时投影残差没有 DCS 的单标量系数约束。Test 三个目标指标下降。已有 dc_half 推理干预使 Val mAP50-95 再下降约 1.674 个百分点，不等同于重训 v2 的结果。

SICR 的代码包含三个 theta、方向性 refine 和阶段内修正，与本次 DCS 完全不同；本地指定目录只有说明文件，没有正式结果归档，因此不能编造其 AP75/Recall。已审计其零初始化预检、共享 RNG、checkpoint 重载和阶段回执修复；本次不做两个未对齐随机整网的 raw Detect 等价比较。

原始数值、成员路径及源码哈希见 [source audit](evidence/dcs_sppf_v1_source_audit.json)。

## 模块包审计与结构

实际读取 `D:/7.21yolo26改/YOLO26缝合.zip`，无需使用备用路径。阅读内容包括原 SPP/SPPF pooling、HLKConv 的大核/空洞深度卷积、StripConv/StripNet 的方向卷积及乘性 attention、newsAddmodules 导出与 tasks parser 注册。只借鉴组织及注册方式，未复制包内现成模块或 YAML，也未引入 timm/mmcv 等依赖。

YOLO26n 顶层仍为 24 层：

| 位置  | b19                                       | DCS v1                            |
| ----- | ----------------------------------------- | --------------------------------- |
| 0–8   | 原 backbone                               | 完全相同                          |
| 9     | SPPF，256→128→256，含 shortcut            | DCS_SPPF，同 cv1/cv2 和原生主路径 |
| 10    | C2PSA                                     | 完全相同                          |
| 11–22 | 原 neck                                   | 完全相同                          |
| 23    | Detect，输入 [16,19,22]，stride [8,16,32] | 完全相同，保留 one2one/one2many   |

DCS 从 SPPF 继承原生构造与权重路径，前向中原 cv1→三次 MaxPool5→concat→cv2→shortcut 运算顺序完整保留；Z0 仅计算一次，M5/M9/M13 直接复用。新增部分固定为：

```text
C5  = M5  - AvgPool2d(5,1,2)(Z0)
C9  = M9  - AvgPool2d(9,1,4)(Z0)
C13 = M13 - AvgPool2d(13,1,6)(Z0)
Ri  = DWConv3×3(dilation=1/2/3, padding=1/2/3) → BN → SiLU
R   = Conv1×1(concat(R5,R9,R13)) → BN
Y   = Y_native + 0.10 × tanh(theta) × R
```

三个 AvgPool 保留指定 PyTorch 构造的默认 `count_include_pad=True`。theta 为形状 `[]` 的单一标量，初值严格为 0。没有 abs、比值归一化、sigmoid、router、多 alpha、StripConv、deformable 或 attention。新增 Conv 初始化位于 CPU `fork_rng` 中，后续 backbone/C2PSA/neck/head 的随机序列不受影响。

## 训练配置逐项继承

读取、哈希验证并比较 b19 args 的全部 **112 字段**。仅模型/预训练文件的等价位置、数据文件的等价位置及实验输出身份允许差异。数据清单实际校验 train/val/test 的 8414/2404/1202 张图像及标签内容，Val/Test 目标数为 2985/1477。

固定 RUN：`yolo26n_b19_dcs_sppf_v1`；数据集：`Tunnel_Crack_AugFirst_Diverse5x_RandomSplit_7_2_1_seed42`。

| 项目                                      | 固定值                              |
| ----------------------------------------- | ----------------------------------- |
| epochs/imgsz/batch/workers/device         | 200 / 640 / 32 / 8 / 0              |
| seed/deterministic/amp/cache              | 42 / True / True / False            |
| optimizer/lr0/lrf                         | MuSGD / 0.01 / 0.003                |
| momentum/weight_decay                     | 0.937 / 0.0005                      |
| warmup_epochs/momentum/bias_lr            | 3.0 / 0.8 / 0.1                     |
| patience/cos_lr/close_mosaic              | 60 / True / 10                      |
| hsv_h/hsv_s/hsv_v                         | 0.024 / 0.84 / 0.535                |
| degrees/translate/scale/shear/perspective | 11.0 / 0.17 / 0.735 / 3.5 / 0.00055 |
| flipud/fliplr/bgr                         | 0.0 / 0.5 / 0.0                     |
| mosaic/mixup/cutmix/copy_paste            | 1.0 / 0.135 / 0.03 / 0.0            |

其余字段同归档 `b19_archived_args.yaml`，未凭记忆补写。保留原 patience=60，因此“200e”是训练预算，仍可能原生早停。禁止自动 OOM 降 batch；在原生重试请求边界重新抛出异常，不能悄悄改变 batch32。

## 分层验证结果

本机：Windows，Python 3.11.15，PyTorch 2.7.1+cu118，RTX 2060。原 b19 服务器记录为 Python 3.12.3 / PyTorch 2.8.0+cu128 / RTX 4090，入口会独立验证该运行环境。

| 检查                                          | 实际结果                                                                |
| --------------------------------------------- | ----------------------------------------------------------------------- |
| import/parser/YOLO 与 DetectionModel build    | PASS                                                                    |
| CPU FP32 模块 zero-init                       | max_abs=0，mean_abs=0，atol=rtol=0                                      |
| CUDA FP32 模块 zero-init                      | max_abs=0，mean_abs=0，atol=rtol=0                                      |
| CUDA autocast 模块 zero-init                  | max_abs=0，mean_abs=0，atol=rtol=0                                      |
| 小尺寸边界/不同通道/shortcut/独立公式         | PASS                                                                    |
| 全部共享 tensor                               | 708/708 exact match，100%                                               |
| 预训练兼容 tensor                             | 606/606 全部继承，100% 兼容覆盖                                         |
| 原始 checkpoint 总 tensor 覆盖                | 606/708，约 85.5932%                                                    |
| shape mismatch                                | 102 个，均为原生 nc80→nc1 Detect 适配；共享模型间为 0                   |
| missing shared/unexpected                     | 0 / 0                                                                   |
| 原 cv1/cv2、C2PSA/neck/Detect 对齐            | PASS，逐 tensor 明细留存                                                |
| 可复现新增权重初始化                          | PASS，重复构建全部 state 精确相同                                       |
| 非零 theta 保存/重载                          | 733 个 state tensor 和固定输出精确一致                                  |
| 新进程重载                                    | 完整输出树精确一致，max_abs=mean_abs=0                                  |
| FP32 原生 detection loss + MuSGD              | B2/640 合成检查，8 次更新观察后全部 13 个新增参数 tensor 均有梯度并更新 |
| AMP 原生 detection loss + MuSGD               | B2/640，10 次尝试后全部 13 个新增参数 tensor 均有梯度并更新             |
| batch32/imgsz640                              | meta 整网形状 [32,3,640,640]→boxes [32,4,8400]、scores [32,1,8400]      |
| 原生 trainer 模型重建及 optimizer 成员/组签名 | PASS，无训练循环                                                        |
| 固定验证图诊断                                | 16 张，FP32 fused，JSON/CSV 全字段成功                                  |
| 专项与相关原生测试                            | 30 passed                                                               |
| Ruff/syntax/bash -n                           | PASS                                                                    |
| reference docs                                | 已用仓库生成器更新新增 API 页面及单条导航                               |
| package dry-run                               | best/last、瞬态目录排除、逐文件 SHA-256、gzip CRC 均 PASS               |

零 theta 的首步支路任务梯度为零，这是链式法则的必然结果；模块检查显式断言首步 theta 非零梯度、支路零梯度，随后验证支路更新。没有手工给 theta 设置非零训练起点，也没有删除失败断言或放宽数值容差。仅固定公式/非零 checkpoint 测试在它们独立的模型副本上设置 theta=0.37。

AMP 完整模型测试的前 5 次默认 GradScaler 尝试发生**缩放梯度溢出**，原生机制将 scale 从 65536 降到 2048，并跳过这 5 次 optimizer 更新；审计确认跳步时所有参数不变。后续接受的梯度、损失、输出及参数均有限，且所有新增参数都实际更新。不能将该结果表述成“从未出现任何缩放梯度 Inf”。模块 autocast 的独立直接 backward 则无需 loss scaling，数值有限。

B32/640 的本地结果是形式检查，不冒充 RTX 4090 上的正式 B32 训练通过。服务器独立预检还会使用原数据加载器、B32、AMP、原 MuSGD 和原 warmup，最多观察 64 个 batch，严格要求 theta、refine、fuse 的全部新增参数有有限梯度并发生更新；不满足则失败，不能启动正式训练。预检与正式训练分属独立进程，训练重新从原始预训练文件和 seed42 初始化。

完整数值见 [local validation](evidence/dcs_sppf_v1_local_validation.json)。诊断样例见 [JSON](evidence/dcs_sppf_v1_diagnostic_smoke.json) / [CSV](evidence/dcs_sppf_v1_diagnostic_smoke.csv)，均来自未训练的非零 theta 测试副本，不能当作训练结果。

## 参数量与计算量

按同一 `get_flops(...,640)` / THOP 口径：

| 模型            |                参数量 |                  GFLOPs |
| --------------- | --------------------: | ----------------------: |
| 原生 b19 nc1    |             2,504,190 |               5.7717760 |
| DCS-SPPF v1 nc1 |             2,607,231 |               5.8555392 |
| 增量            | 103,041（约 4.1147%） | 0.0837632（约 1.4513%） |

新参数由三组 DW+BN、384→256 的 1×1+BN 和一个 theta 构成。GFLOPs 是工具估算值，未宣称覆盖每个池化/逐元素操作，也不能替代实际延迟测量。

## 服务器入口与产物

```bash
bash tools/experiments/server_b19_dcs_sppf_v1.sh train
bash tools/experiments/server_b19_dcs_sppf_v1.sh test
bash tools/experiments/server_b19_dcs_sppf_v1.sh diagnose
bash tools/experiments/server_b19_dcs_sppf_v1.sh package
```

train 自动完成 runtime audit → 独立 preflight → 当前进程、当前源码/参数/权重/数据对应的 PASS → native 200e 训练。使用唯一 attempt 目录，旧 PASS 不能复用；原实验目录存在则拒绝覆盖。无需额外设置 CUDA_VISIBLE_DEVICES、TF32 或 CUBLAS。`init_seeds` 执行的是本仓库原生 deterministic 初始化。

test 使用验证集选出的 best.pt，对 DCS val/test 与原 b19 test 做相同 FP32 设置评估，输出 AP75、Recall 和 mAP50-95 差值；不重新训练 b19，不按 Test 重新选模型。diagnose 固定前 16 张排序验证图，保存图像/标签哈希、theta、alpha、C5/C9/C13 与 R5/R9/R13 的 mean/std/max、fused_R、alpha_times_R、Y_native 和 residual_ratio。

package 要求完成训练及匹配的比较/诊断证据。包含 best/last.pt、args.yaml、results.csv、results.png、val/test 曲线及预测、baseline comparison、diagnose、provenance、Git SHA、source.patch、完整 source.tar（含 YAML、模块、注册、实验脚本、文档）和逐文件 SHA-256 清单/包摘要。目录遍历前排除 `.attempt.*` 和符号链接，不要求 `console.log` 存在；不追随 dangling symlink。Windows 无创建真实 symlink 的权限，断链谓词使用仿真覆盖，其余打包与校验路径实际执行。正式 Linux 端仍应实际执行完整 package 阶段。

提交后的完整源码打包复核发现并修复了 Windows 默认 GBK 解码 Git 输出的问题：文本 Git 查询显式按 UTF-8 解码，`source.patch` 直接按原始字节写入，并新增逐字节断言。修复后重新验证打包，保留中文内容、二进制 patch 和末尾换行。

## 修改文件及审计结论

主要文件：`ultralytics/nn/modules/dcs_sppf.py`、模块 `__init__.py`、`nn/tasks.py`、`yolo26n-dcs-sppf-v1.yaml`；四个指定 run/verify/finish/server 脚本；复用的 `b19_common.py`、归档 args/reference/test_reference/dataset_manifest/launcher；专项测试；实施/证据/API 文档、单条 mkdocs 导航；`.gitignore` 仅新增本地 artifacts 排除。

差异审计检查了结构、初始化、参数来源、零初始化延迟梯度、checkpoint/fused 诊断、固定 recipe、并发阶段锁、失败退出和打包证据。核心 `block.py`、Conv、C2PSA、Detect、neck 和 engine trainer/loss 均无修改。

Deleted: 基线中不删除原生代码。本次是在不允许删除/改动原生网络的前提下新增独立实验；复用 SPPF/Conv/native trainer 和既有证据/打包工具，移除了迁入工具中未使用的推理属性辅助函数及其它实验专属诊断。没有为了消除测试失败而给模型增加跳过支路的条件。

已知风险：尚无 DCS 正式精度证据；本地环境不同于服务器；±0.1 限制的是残差系数，不能保证 residual_ratio ≤0.1；池化零填充的边界语义按指定默认保留；AMP 初始 scaler 跳步已如实记录；真实 Linux 断链及完整 B32 预检待服务器执行。没有叠加其它创新，也没有根据既往 Test 指标自动更改 v1 设计。

论文机制：原生 SPPF 通过重复最大池化传播强响应；DCS 在完整保留主路径的同时，用同尺度 MaxPool–AvgPool 显式表示相对局部背景的对比显著性，经轻量空洞深度卷积精炼，再由零初始化受限残差补充原特征。强化细裂缝可分性是待正式实验验证的机制假设，不是本次已证实的精度结论。
