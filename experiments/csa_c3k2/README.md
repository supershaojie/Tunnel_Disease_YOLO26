# b19 + CSA-C3k2

正式训练状态：**NOT_STARTED**。本目录只实现固定首版和有限验证，未进行完整 val/test、200 轮训练、性能扫描或服务器连接。构建通过不代表优于 b19。

基线：`4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6`；分支：`exp/csa-c3k2-b19`。

## 结构和精度

`yolo26n-csa.yaml` 从锁定的原生 YAML 复制，`scale=n`、`nc=1`，仅第 4 层换为 `CSAC3k2(64,128,64,32,2,7,4)`。其余节点及 Detect `[16,19,22]` 保持原样，内部宽度不经 parser 缩放，本类不属于 repeat_modules。

`stem:64→128` 分成 A/B 两条 64 通道特征；B 经两个独立 CSAUnit 得 B1/B2；`merge:256→128` 接收 `[A,B,B1,B2]`。每个块 `64→32` 后计算局部 3×3、横向 1×7、纵向 7×1 三分支，各有 BN；空间选择器按四个连续 8 通道组在三分支间 softmax，`32→64` 投影后残差相加。

横/纵分支各自预测四组独立的六个步进。FP32 tanh 后累计为 `[n3,n2,n1,0,p1,p2,p3]`，横向仅 dy、纵向仅 dx，偏移顺序是组/点/[dy,dx]。权重卷积 groups=1；偏移组和选择组为 4。中心固定，两侧独立，累计偏移不再次截断。使用 torchvision 原生 `deform_conv2d(mask=None)`，零填充、双线性边界；不是完整 DCNv2。

偏移预测末层和选择末层为零初始化；其余参数正常初始化。初始选择为 1/3。外部卷积、偏移预测器保留原生 autocast；只有坐标、采样核调用及选择累加用 FP32，采样结果转回分支 dtype 后再 BN。支持 FP32 参数的 AMP 训练和半精度 checkpoint 推理。分支融合只折叠 kernel/BN，仍运行曲线采样，并保留融合 bias。

## 入口

从本 worktree 根目录运行，三个入口均支持 `--help`：

```text
python experiments/csa_c3k2/train.py --data DATA --weights WEIGHTS --baseline-args BASELINE_ARGS
    [--project PROJECT] [--name NAME] [--device DEVICE] [--dry-run] [--report REPORT]
python experiments/csa_c3k2/verify.py --data DATA --weights WEIGHTS --baseline-args BASELINE_ARGS
    --report REPORT [--device DEVICE]
python experiments/csa_c3k2/validate.py --weights WEIGHTS --data DATA --project PROJECT --name NAME
    [--device DEVICE] [--split {val,test}] [--diagnostic-samples {0,1,2,3,4}]
```

`--weights` 在 train/verify 中必须是原始 yolo26n.pt，SHA256 为 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`；validate 中是已学习的 CSA checkpoint。入口检查从本 worktree 导入，并设置 `YOLO_AUTOINSTALL=false`，不会安装依赖。原生 AMP 检查使用 worktree 内已核验的同名权重副本，不覆盖错误同名文件。

必须提供 b19 **完整** args.yaml。所有非运行字段的规范 JSON SHA256 固定为 `81c506eb50bcb29a50d37c080ba35b212eb4faff42dff65ca5e535f02dd0ee7f`，拒绝缺失字段或机制差异。原配置、解析配置、实际 Trainer 配置和逐字段差异写入 JSON；CPU 下原生 workers=0 的变化单独记录。模型/权重/数据路径、project/name/save_dir、显式 device 是允许的运行差异。

默认输出 `runs/csa_c3k2/csa_c3k2_b19_e200_i640_b32_s42`；已存在即报错，resume=False。dry-run/verify 使用临时目录，报告必须在正式输出目录外。正式入口始终保持 B32/640/AMP/MuSGD，原生 AMP 不通过时终止。OOM 策略由实验 Trainer 覆写，在 batch 减半前报错；原生 Trainer 的重试行为保留。后续服务器训练/评估分别使用独立 tmux `csa-b19` / `csa-eval`，允许 GPU0 并发，不等待空闲、不终止其他实验。

## 初始化和验证

生产调用链是继承的 `BaseTrainer.setup_model` → `CSATrainer.get_model`：从核验的原始权重按 `DetectionTrainer.get_model` 建 nc=1 参考模型（含类别头适配），记录 CPU/Python/NumPy/CUDA RNG，在隔离 RNG 中建创新模型，仅逐键复制第 4 层以外的状态。完整状态严格加载后核对键、形状、连接、stride、值和存储独立性；创新图与元数据由自身持有，不注册参考模型。迁移报告分别列出原预训练保留、原生类别适配缺口、主动删除、新增初始化、意外遗漏及参数元素数。

独立 RNG 对照在真实的两个 setup_model 调用入口使用相同快照；这避免将原生 Events 单例首次导入消耗 Python RNG 混入构建对比。创新构建本身始终在生产 get_model 中隔离，验证中不补拷贝权重。

verify 包含矩形/边界零偏移等价、四组非零偏移的坐标斜坡与脉冲标量 oracle、非均匀选择路由、640 和 384×640 图接线、参数逐项审计，以及原生 O2M/O2O 检测损失和 MuSGD 的 CPU FP32/CUDA AMP 各 3 次有效更新。smoke 使用 B2/64×96 合成非空框，最多每种精度 24 微批，记录 unscale 后每个新参数的梯度、实际变化和 GradScaler 跳步。独立副本保证正式初始化、BN、权重文件和 RNG 不被污染。

生命周期测试使用学习后的模型，再显式加入非零偏移、非均匀选择和 BN 偏置，检查 EMA、公开 save、受限新进程重载、FP32 eval、半精度前向/融合、predict、native fuse、重复 fuse 和融合后再保存重载。模块通过原生安全类发现机制登记；受限加载仅补充原生融合类的 `forward_fuse` 绑定方法，继续拒绝其他实例属性。

本机有限验证：Python 3.11.15、torch 2.7.1+cu118、torchvision 0.22.1+cu118、RTX 2060。上述核心检查已通过；CUDA 反向实际报告 `compute_grad_input` 非确定性警告，保留 deterministic=True/warn_only=True，不声称逐位可复现。目标服务器 Python 3.12.3 / torch 2.8.0 / torchvision 0.23.0 / RTX 4090 以及 B32/640 均为 **UNVERIFIED**。

| 对象           | unfused 参数 | native fused 参数 |
| -------------- | -----------: | ----------------: |
| 原第 4 层 C3k2 |       26,080 |            25,840 |
| CSA-C3k2       |      105,624 |           104,984 |
| b19 整网 nc=1  |    2,504,190 |         2,375,031 |
| CSA 整网 nc=1  |    2,583,734 |         2,454,175 |
| 净增加         |       79,544 |            79,144 |

这是本机实际构建/融合计数，buffers 不计参数。若评价输入已融合，无法从该 checkpoint 取得的 unfused 计数写为 null/UNVERIFIED，仍报告实际 fused 计数；不会重建随机模型充当未融合状态。整网融合下降还包含原生 O2M 移除。THOP 不完整覆盖函数式变形采样、坐标、softmax 和加权聚合，其摘要不是完整 FLOPs，不以参数量推断速度。验证发现原生 profiling/warmup 的未初始化测试输入会让采样算子崩溃，已在输入创建处改为零张量；同时修复原生权重路径处理删除内部单引号的问题。

## 后续评估

validate 恢复 checkpoint 自身结构，不触发原始权重迁移。口径为 FP32（显式 half=False，锁定源码归一化为 quantize=None）、640、B32、conf=0.001、iou=0.7、max_det=300、rect=True、augment=False、workers=8、native end2end=True/fuse=True，记录实际模型和输入 dtype。输出 P/R/F1/AP50/AP75/mAP50–95，AP75 从实际 IoU 数组定位；保留 PR/置信度曲线。conf=0.25/0.50 的结果标为曲线估计，不伪造 TP/FP/FN。test 仅用于固定方案最终评价。

诊断默认关闭，最多四个真实评价样本，在预热后启用可移除 hooks，仅保存两个块/两个方向的步进与累计偏移分位数、tanh 饱和率、组/分支均值、残差/输入范数比。不保存全量激活，不更改正常 forward 返回类型，采样行为不能直接解释为裂缝跟随或拓扑改善。

## 来源

条带表达参考 [Strip R-CNN](https://arxiv.org/abs/2501.03775)，连续累计坐标参考 [Dynamic Snake Convolution，ICCV 2023](https://arxiv.org/abs/2307.08388)，空间响应选择参考 [LSKNet 原论文](https://arxiv.org/abs/2303.09030)和[作者版本说明](https://github.com/zcablii/LSKNet)。接口依据 [torchvision deform_conv2d 文档](https://docs.pytorch.org/vision/0.23/generated/torchvision.ops.deform_conv2d.html)，实际安装版本用 oracle 检验。没有导入模块包、mmcv/timm、拓扑损失、额外头或辅助创新；本组合的有效性和相对历史 DCRStrip/PCS 的差异仍需实验验证。
