# QCA-C2PSA / b19

正式训练状态：**NOT_STARTED**。本目录交付独立结构实验、有限工程检查与服务器入口；工程通过、theta 更新或注意力变化都不证明检测涨点。

## 固定范围

- 基线：`4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6`，分支 `exp/qca-c2psa-b19`。
- 只替换 YOLO26 第 10 层 C2PSA 内的注意力聚合；其余骨干、Neck、SPPF、Detect、detach、损失及训练流程沿用 b19。
- 原生 YAML 仍构造原生模型。QCA 使用独立 YAML，经原生 parser、Trainer 重建、EMA、保存加载和融合流程运行。
- 保留原生参数路径与 BN buffer，只增加 `model.10.m.<i>.attn.theta`，每个实际头一个标量。

## 数学与精度

QCA 使用有效邻域的 3×3 平均查询 Qc（`count_include_pad=False`），共享 K/V；
`lambda = 0.25*tanh(theta)`，`Aeff = A + lambda*(A-Ac)`。
Aeff 是有符号聚合权重，行和约为 1，但可以为负；没有额外 softmax 或 clamp。
负 lambda 表示向局部上下文参照平滑，Qc 不预先等同于背景，lambda 的范围也不限制特征变化幅度。

原生 b19 主路严格保留 `(q * scale).transpose(-2, -1) @ k`、原生 softmax 和 V 聚合。
仅修正局部关闭 autocast，执行 FP32 空间平均、`(Qc32.T @ K32)*scale`、
`softmax(native_logits.float())`、参照 softmax 与差分 V 聚合。
FP32 的 `lambda*D` 转回主消息 dtype 后相加，再按原顺序加原始 V 的 PE 并投影。
theta=0 仍计算全部修正分支，允许从零学习；没有跳过、detach 或第二个零门。
AMP 原生 logits 已经低精度舍入，因此这是一种明确的混合精度策略，不声称与统一低精度直接公式逐位一致。

该实验受 [Differential Transformer](https://arxiv.org/abs/2410.05258) 与
[Linear Differential Vision Transformer](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5820ad65b1c27411417ae8b59433e580-Abstract-Conference.html)
启发，但不复制它们的完整架构，也不继承线性复杂度或其任务上的性能结论。

## 配方与输入来源

完整训练配置源于本地历史包：
`E:/ditieyolo26跑结果/8.12离线在线/b19 200e/b19_yolo26n_e200_train_val_test_20260823_224153/train_run/args.yaml`。
源文件 SHA256：`b08b915756bf85c91d3356586a651a37156d71867b84b6e8493d75c9568642b9`。
同包 `environment/git_state.txt` 记录上述 b19 锚点。
`b19_train.yaml` 保留算法字段，运行时只绑定模型、数据、原始权重与独立输出位置；不继承 smoke 配置。

- 200e / patience60 / 640 / batch32 / MuSGD / seed42 / AMP / workers8。
- lr0=.01，lrf=.003，momentum=.937，weight_decay=.0005，cos_lr=True。
- warmup_epochs=3，warmup_momentum=.8，warmup_bias_lr=.1，nbs=64；正常阶段累积 2。
- 8414 张训练图片对应每 epoch 263 batches，warmup 789 iterations；尾部 10e 原生关闭 mosaic、copy_paste、mixup、cutmix。
- O2M/O2O 权重从 .8/.2 逐 epoch 变至 .1/.9；assigner 与损失沿用固定源码。
- val 选择 best 的原生 fitness 为 box mAP50–95；训练期间 val 使用 EMA、batch64、rect=True、FP16 autocast，conf=None 对应 .001。

本地公共初始化 `E:/PycharmProjects/Tunnel_Disease_YOLO26/yolo26n.pt` 的 SHA256 已核实为：
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`。
正式入口必须使用该原始文件，全新构造、`resume=False`；不能使用任何 best、smoke、EMA 或 optimizer 状态。
nc80→nc1 的原生分类头适配应与 native 相同；详细共同键、形状及加载数值检查见验证报告。

本地数据 YAML：
`E:/PycharmProjects/Tunnel_Disease_YOLO26/datasets/Tunnel_Crack_AugFirst_Diverse5x_RandomSplit_7_2_1_seed42/data.yaml`。
SHA256：`1f18760508e9dbf2332cd7102ee9c15e11e08f3d15fc4202c8ee0ed1bb785b12`。
实查 train/val/test 图片和标签数量分别为 8414/2404/1202，nc=1，类别 crack。
这是文件级随机划分，存在同来源原图跨集，不能宣称来源完全隔离。没有重分数据或使用 RT-DETR 数据。
服务器同名路径只是历史记录，本轮没有连接服务器核验。

历史 b19 为 Python3.12.3 / torch2.8.0+cu128 / RTX4090；本地工程检查为
Python3.11.15 / torch2.7.1+cu118 / RTX2060 6GiB。
历史 pip_freeze 和日志均无 Albumentations，本地装有 2.0.8。
正式训练前须在单独的服务器环境核对该差异，避免引入额外默认增强；本轮不修改共享环境。

历史 test 记录缺少完整独立 args/启动命令；仅核实 1202 图/1477 实例、38 batches、augment=False、save_json=True。
精确历史 test mAP50–95 `50.7769939%` 来自任务文档，只作为研究参照；日志显示四舍五入的 .508。
最终 test 协议需先恢复核实，不能用 test 选择结构、超参数或 checkpoint；报告同时关注 Recall、Precision、AP75。

## 开销

实际 nc=1、nano：1 个 QCAAttention（`model.10.m.0.attn`），2 个头；
模型参数从 2,504,190 增至 2,504,192。Trainer/THOP 摘要约 5.8 GFLOPs，
该统计漏计 functional matmul/softmax 等，不能用它判定 QCA 零开销。

QCA 保留 N×N attention，并额外执行 QcK 矩阵乘法、参照 softmax、FP32 参考 softmax、差分 V 聚合及局部池化。
新增主矩阵乘法约为 `B*h*N*N*(dk+dv)` MACs，即按乘加各计一次约两倍 FLOPs。
这尚未包含池化、softmax、tanh、转换、逐元素计算及显存成本。
640 nano 的第 10 层空间为 20×20、2 头、dk32/dv64，新增主矩阵乘法为
30.72M MACs / 61.44M FLOPs。单个 FP32 attention 张量为 1,280,000 字节；实际峰值包含多张张量和反向状态。
参数很少并不意味着零计算或零显存，也不恢复已经丢失的裂缝像素。

## 有限验证结果

完整事实与误差见 `check_results.json`，检查代码为 `check_qca.py`。已完成后停止追加检查。

| 检查               | 结果                                                                                                              |
| ------------------ | ----------------------------------------------------------------------------------------------------------------- |
| 独立 FP32 直接公式 | 5×7、1×7、7×1、1×1；1/2/4 头；有效邻域平均和行和通过，含正负 theta                                                |
| 构造与初始化       | native/QCA RNG 消耗相同；nc1 的 708 个共同参数/BN buffer 同步且一致                                               |
| 公共权重继承       | 606 个形状匹配张量逐项数值一致；102 个原生 nc80→nc1 分类塔形状缺口；意外丢失原生键 0                              |
| theta=0            | Attention、2-repeat C2PSA 前向与共同梯度最大绝对误差 0；整网 top-k 前 O2M/O2O 在 FP32 与 CUDA FP16 的最大误差均 0 |
| theta 梯度         | 正常非均匀输入下有限且非零；没有跳过零状态修正                                                                    |
| 640 单张           | CUDA 无梯度，输出 `[1,300,6]`，有限；仅做一次                                                                     |
| 非零 theta=.4      | 相同权重下相对 theta=0 原始输出最大差 .048676；deepcopy/EMA 保留 theta                                            |
| 融合               | 全部原有 BN 融合，保留 QCA；raw O2O 最大误差 3.48e-5（atol/rtol=1e-4）                                            |
| 原生保存重载       | 原生 save 将参数转 half，theta=.39990234375；与相同 half→float 状态的输出误差 0，默认 predict 通过                |
| 实际 Trainer       | 6 张非空合成训练图、2 张合成 val 图；FP16、MuSGD 实际 step=3，loss/梯度/参数有限                                  |

theta 在原生普通 weight 参数组恰好注册一次，lr=.01、weight_decay=.0005、use_muon=False（MuSGD 内原生一维 SGD 路径）。
三步后 theta 为约 `[-0.00048117, -0.00013244]`。原生模型与检测塔继续训练，没有自设参数学习率。
该 smoke 仅覆盖工程集成：batch2、nbs2、imgsz160、1 epoch、workers0、warmup0；
GradScaler 初始 scale128 仅用于有限检查。正式配置仍 batch32/nbs64/640/200e，并使用原生 scaler。
本进程用已经实际执行的 QCA FP16 检查代替下载式 stock AMP probe，并在 plots=False 时跳过字体下载与外部日志集成；
这些替代不进入正式入口。本地可选 Albumentations 的差异保留在 smoke 中，因此这不是正式数据协议验证。

保存重载前后的 FP32 logits 差异包含原生 half 保存量化（本次 scores 最大 .001657）；
不能把它与 QCA 零初始化误差混为一谈。fuse 原生移除 O2M 塔，因此比较仍用于默认推理的 raw O2O。
没有验证 BF16、正式 B32 显存或服务器依赖；本次未采用 BF16，不以 FP16 结果替代其检查。

初次重载受 b19 下载器删除路径单引号影响而失败；只将临时目录改到不含单引号的位置，
重跑 serialization/trainer 两组后通过，未修改下载器或放宽已有误差阈值。
Windows 重跑检查时，先将本进程的 `TEMP` 与 `TMP` 设为已存在、可写且不含单引号的目录。
`check_results.json` 合并首次通过的 formula/initial 和随后通过的 serialization/trainer 结果，并记录该过程。

## 交接

`check_qca.py` 是有限工程验证脚本，`train_qca.py` 是正式入口。
服务器须从现有 origin 正常 fetch，在独立 worktree 检出最终交付的完整 SHA，核对 HEAD 和 `ultralytics.__file__` 后运行。
固定 SHA 和可复制 Bash 命令在最终交付答复中给出，避免提交记录自身 SHA 的循环提交。
本轮不运行正式 200e，不连接服务器，不上传数据、权重或密钥。
