# b19 SGK-P3 v1 实施与验证

本实验从原生 b19 `4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6` 独立建立，分支为
`codex/exp-yolo26n-b19-sgk-p3-v1`。SGK 是项目工作名，尚未经过正式训练验证，不能据此宣称准确率或召回提升。
服务器步骤见 [操作文档](b19_sgk_p3_v1_server.md)。最终交付目录另有代入真实提交 SHA 的操作文档。

## 结构及固定公式

唯一图结构差异是第 16 层从原生 `[-1, 2, C3k2, [256, True]]` 替换为
`[[15, 13], 2, C3k2_SGK_P3, [256, True]]`。继承原生类，保留 `cv1/cv2/m` 权重路径，新增参数仅位于
`model.16.sgk.*`。该层同时进入 P3 Detect 和第 17 层 PAN，可能影响所有下游检测尺度。

```text
F = native_C3k2(X15); a,b = split(F,32+32)
G = nearest(SiLU(DW5(Ps(X13))), size=b.shape[-2:])
Z = SiLU(Pb(b)+G)
K = softmax(Wk(Z).reshape(B,4,9,H,W), dim=2)
T = sum_j K[:,:,j] * replicate_padded_P3_value_at_offset_j
Y = cat(a,b+Po(T-b))
```

Ps=128→16 1×1，DW5=16 通道深度 5×5，Pb=32→16 1×1，Wk=16→36 1×1（唯一 bias），Po=32→32 1×1。
全部为裸 Conv2d，无新增 BN；仅 Wk.bias 和 Po.weight 置零，其余正常初始化。
新分支构造使用 `torch.random.fork_rng(devices=[])`，不消耗后续原生层的 CPU 初始化随机序列。

邻域顺序为从左上到右下的行优先九点，四组各八个值通道；九次切片，不构造正式 batch 的 unfold。
softmax、加权累加和 `T-b` 在显式禁用 autocast 的 FP32 区域执行，再转回 b.dtype 送 Po。
P4 仅用于核预测。`forward_split` 同样执行 SGK。其他层、head 的 detach、loss、分配器、MuSGD、EMA 和后处理均沿用原生代码。
解析器只接受 nano、单类、第16层 `[15,13]`，不隐式推广其他 scale。

## 实测与边界

| 项目                          | 本地结果                                                                                       |
| ----------------------------- | ---------------------------------------------------------------------------------------------- |
| 未融合原生参数                | 2,504,190                                                                                      |
| 未融合 SGK 参数               | 2,508,786                                                                                      |
| 新增参数                      | 4,596（六个参数张量）                                                                          |
| fuse / one-to-many 裁剪后参数 | 2,379,627，不能与训练预算混比                                                                  |
| THOP 640 GFLOPs 估计          | 5.8066432；未覆盖 functional softmax/切片聚合等操作，不是完整算量                              |
| 原生共有 state_dict           | 708 项含 BN buffers，逐键及形状相同                                                            |
| 原始预训练匹配                | 606 项；102 项原生 nc=80 → nc=1 head 形状差异逐键记录                                          |
| 零投影整网 FP32 eval          | CPU 640×640、640×960 逐元素相等                                                                |
| 训练态相等及 RNG              | 通过，同样输入的独立原生/候选模型                                                              |
| 独立索引参考                  | 不对称网格、各组/位置不同核、九个 one-hot 偏移、replicate 边界均通过                           |
| MuSGD 学习路径                | 首步仅 Po 非零任务梯度，第二步六个张量均有非零梯度及真实更新                                   |
| 预检 step 观察器              | 与原生 optimizer_step 的模型/EMA 结果逐元素一致；零任务梯度副本排除单独 weight decay 的更新    |
| 保存重载及 fuse               | 独立进程 FP16 checkpoint → FP32；全部保留 one-to-one 原始/解码输出在 1e-4 内，原生对照同时通过 |
| 真实 Validator / diagnose     | 固定16张真实 val 图片，原生变换与真实检测 loss，功能探针通过                                   |

本地环境是 RTX 2060 6 GiB、Python 3.11.15、PyTorch 2.7.1+cu118、Ultralytics 8.4.98。
原 b19 环境为 RTX 4090、Python 3.12.3、PyTorch 2.8.0+cu128。没有升级或重装环境。
本地 CUDA 测试为 batch=1 的 640×640、640×960，均完成 FP32/AMP 检测 loss 前后向有限性检查。
对应显存峰值记录在 `artifacts/sgk_local/cuda.json`（PyTorch max_memory_allocated，不是 nvidia-smi 总占用）。
本轮测得约 270/173 MiB（640方图 FP32/AMP）、407/247 MiB（640×960 FP32/AMP）；最终以 JSON 原始字节数为准。

这些开发测试不等于服务器 batch32/640 的正式预检。没有执行 SSH、服务器训练、全量 SGK val/test 或正式结果打包。
16 张图的功能探针是原始预训练加人工非零 Po，仅用于检验路径，不能用其 AP 或图片作为实验结论。

## 来源证据

实际读取了本地原生 b19 的 `train_run/args.yaml` 与 `results.csv`，哈希分别匹配
`b08b915756bf85c91d3356586a651a37156d71867b84b6e8493d75c9568642b9` 和
`44795068d0ebc687c29ff7b7b8cbf337469546b2d4dd515cb0616c270bdf9507`。
原始 yolo26n.pt 匹配 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`。
数据 zip 实际计数为 train=8414、val=2404/2985 实例、test=1202/1477 实例。
原始服务器 launcher 文件当前不可访问；本地只核验已存档命令文本能逐字段重建真实 args。
正式入口强制要求用户指定原始 launcher，缺失或不一致时报错，不使用本地文本替代。

实际模块参考包为 `D:/7.21yolo26改/YOLO26缝合.zip`，内容 SHA256 为
`a0c197f38e1510291a9a762fc7dc0a8c13694742fff9e3132c50a3ee32ddadcf`。
仅阅读 `CGhalfConv_2025ESWA.py` 的部分通道思想及 `MRFAConv_2025ICCV.py` 的低维投影；没有引入包内实现或依赖。
机制出处：[LSNet: See Large, Focus Small, CVPR 2025](https://arxiv.org/abs/2503.23135)；
[Involution, CVPR 2021](https://arxiv.org/abs/2103.06255)。参考包文件名会议标签不作为论文出处证据。

## 运行基础设施

从 `1b4b98ede9499eae225010893c12c57ac5c3cd5f` 移植 b19 配方审计、原生更新时序、进程隔离、重载融合、评估和归档代码，
没有移植 PKC 模型、YAML 或依赖。已删除原 PKC 的 BN gamma replay、BN 更新断言和专属诊断，替换为六个 SGK 参数和动态核诊断。
正式来源检查不允许从常数恢复缺失 args；检查原始 args/results/权重哈希、全部配置字段、实际数据计数和图像/标签内容指纹。

`train` 每次启动独立 preflight 子进程，最多观察128个不同训练 batch；独立 FP32 探针复用首个输入 batch。
观察器沿用原生 warmup、scheduler、accumulate、AMP unscale、裁剪、MuSGD、EMA 时序；post-step hook 确认真实 optimizer.step。
每张量记录梯度/裁剪/LR/scale/跳步/实际更新，另用同参数与历史 momentum 的零任务梯度副本排除仅衰减导致的移动。
通过后重新设 seed、用原始权重创建新 trainer，不复用预检状态。正式 preflight 还运行全量原生 Validator。

每次 shell stage 有独立 attempt、PID、命令、HEAD、日志、退出码及 `.current_attempt` 指针；Python preflight 子进程另有独立 invocation 状态。
锁、已有 run 拒绝覆盖、原生 OOM 重试在改变 batch 前抛出原错误；不降 batch、不关闭 AMP、不静默 resume。
`test` 的 val/test 由两个独立进程完成，固定同一 best.pt。此版本源码以 `quantize=None` 表示 FP32，是弃用参数 `half=False` 的原生等价方式。
记录实际 TF32/cuDNN/backend 设置；不额外加 NMS。P/R 是原生最佳 F1 操作点；AP75 取第5号 IoU 索引，mAP50–95 不是 IoU=.95 单点 AP。
R@P≥.85 的阈值先由 val 确定并写出冻结文件，再启动 test；不可达就如实记录。
若服务器原 b19 best.pt 存在，`test` 自动通过同一入口补充独立 baseline 对比，无需重训 baseline。

`diagnose` 固定按路径排序的前16张 val 图，保存图像/标签哈希；记录分组归一化核熵、中心值、空间变化、最大概率、近均匀/单点饱和比例、残差比和引导/控制器任务梯度。
核权重不是裂缝概率；同权重残差开关只是功能诊断，不是重训消融。
`package` 要求当前训练/test/diagnose attempt 成功，核对提交、best、配置、数据和各报告哈希，拒绝缺项。
归档只包含当前 run 与复现源码、best 和来源证据；不含原始数据、环境、整个 Git、其他实验 runs 或全部 epoch 权重。归档后逐项校验及完整 gzip CRC/SHA256。

## 本地复现

```powershell
$env:PYTHONUTF8='1'
D:/miniconda3/envs/yolo26/python.exe tools/experiments/verify_b19_sgk_p3.py `
  --pretrained ../exp-yolo26n-b19-pkc-sppf-v1/yolo26n.pt `
  --baseline-args 'E:/ditieyolo26跑结果/8.12离线在线/b19 200e/b19_yolo26n_e200_train_val_test_20260823_224153/train_run/args.yaml' `
  --dataset-zip '../../artifacts/datasets/Tunnel_Crack_AugFirst_Diverse5x_RandomSplit_7_2_1_seed42.zip' `
  --reference-zip 'D:/7.21yolo26改/YOLO26缝合.zip'
```

该入口输出 `artifacts/sgk_local/`，测试、完整逐键匹配、CUDA原始字节数、真实图片清单和诊断均保留。
最终小型报告快照位于 `docs/experiments/sgk_p3_v1_evidence/`，不提交本地图片、checkpoint 或大日志。

Deleted: 原生层16的单输入 C3k2 配置在候选 YAML 中被替换；移植运行器时删除 PKC 专属 BN/replay/诊断及缺失原始来源的回退。
Reused: 原生 C3k2、解析器缩放、检测 loss/head、MuSGD/EMA/Validator，以及已有独立实验运行基础设施。
新增模块和验收文件是独立功能所必需，无法仅通过删除或搬移原生算子实现固定 SGK 公式。
