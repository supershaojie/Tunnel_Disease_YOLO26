# b19 NDP-SPPF v1 实施与验证

本分支只实施归一化分布池化 SPPF（Normalized Distribution Pooling SPPF）。它是待检验的固定候选结构，尚无正式训练指标，不承诺提升 mAP、定位或召回。

## 版本、原始证据与复用范围

- 分支：`codex/exp-yolo26n-b19-ndp-sppf-v1`。
- 基线锚点：`4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6`；实验直接从该提交建立。
- 交付版本：运行 `git rev-parse HEAD`，再与 `git ls-remote origin refs/heads/codex/exp-yolo26n-b19-ndp-sppf-v1` 核对。包含文档自身的提交不在本文写死自身 SHA。
- 推送仓库：`https://github.com/supershaojie/Tunnel_Disease_YOLO26.git`，仅上述分支；不创建 PR、不合并、不 force push。
- 本机 worktree：`E:/PycharmProjects/Tunnel_Disease_YOLO26/.worktrees/exp-yolo26n-b19-ndp-sppf-v1`。
- 真实 b19 档案：`E:/ditieyolo26跑结果/8.12离线在线/b19 200e/b19_yolo26n_e200_train_val_test_20260823_224153`。
- 实读 `train_run/args.yaml`、训练 console、数据配置与 Git 环境记录。args SHA256 为 `b08b915756bf85c91d3356586a651a37156d71867b84b6e8493d75c9568642b9`。具体原始文件哈希见 `tools/experiments/ndp_sources.json`。
- 原始 shell 脚本未收录于 b19 结果包；复用的 `b19_launcher_expanded.txt` 来自历史 b19 实施文档第 9 节。必须与原始 args 逐项一致，不能将其称为从服务器取回的 shell 原件。
- 读取并复用 BDI 分支 `662122d` 的 b19 通用审计、阶段进程、预检、评估与打包实现，包含已修复 PKC/RPCA 预检思想；没有引入 BDI、PKC、SIR 等模型代码和训练后权重。

Deleted: 替换实验 YAML 第 9 层的原生 SPPF 条目；从复用基础设施中删除 BDI 的 P2 边、滤波测试、band-pass 统计和重复诊断梯度读取。
新增代码用于独立实验公式、注册、数值与生命周期证据；基线没有实验基础设施，不能仅通过删除/迁移实现新实验。通用配置审计集中于 `b19_common.py`，沿用原生 Trainer、MuSGD、EMA、Validator、后处理与融合。

## 参考来源与机制差异

实际模块包：`D:/7.21yolo26改/YOLO26缝合.zip`，SHA256：
`a0c197f38e1510291a9a762fc7dc0a8c13694742fff9e3132c50a3ee32ddadcf`，与指定版本一致。

阅读 `YOLO26缝合/ultralytics/nn/newsAddmodules/` 中：

- `PATConv_AAAI2026.py::SRM`：按通道跨全图计算 mean/std，再经过卷积与 Hardsigmoid 乘法 gate；代码声明 BN，但该 forward 中 BN 被注释。NDP 仅借鉴统计视角，改用逐有效局部窗口总体方差，不采用全局 gate、BN 或 PATConv。
- `RCSSC_2025TGRS.py::ChannelPool/CALayer/spatial_attn_layer/RCSSC`：分别按通道或空间轴组合 mean/max 并用于乘法注意力。NDP 不搬入乘法注意力、coarse sigmoid 或该卷积网络。

上述文件名中的会议年份不作为出版证明。文件逐一 SHA256 已保存于 `ndp_sources.json`。

[SoftPool 原文](https://arxiv.org/html/2101.00440v3) 第 3.1 节采用激活值指数归一化后的加权池化；[LIP 原文](https://arxiv.org/html/1908.04156v3) 使用可学习子网络生成局部重要性。NDP 不直接复现这两篇论文：采用固定局部标准化、有界得分和中心化残差，stride=1，作为原 SPPF 的并行补充。加权池化本身已有研究，名称不证明学术创新充分。

实际阅读 SIR v1、SIR v2 和 PKC 源码并记录哈希：SIR v1 累积相邻最大池化增量的路由修正；SIR v2 每尺度独立加入 `0.5*tanh(logit)*increment` 后进入 cv2；PKC 是串行的低维空间卷积金字塔和零末端投影。NDP 从同一 U 直接取 5/9/13 原始窗口值，不计算池化增量路由或空间卷积金字塔。此审计只覆盖这些真实版本，不宣称穷尽全部既有研究。

## 唯一架构改动

文件：`ultralytics/nn/modules/ndp_sppf.py`，类 `SPPF_NDP(SPPF)`；在 `modules/__init__.py` 导出并在 `nn/tasks.py` 的 base_modules 注册。实验 YAML 为 `ultralytics/cfg/models/26/yolo26n-ndp-sppf-v1.yaml`。

保持原签名 `(c1,c2,k=5,n=3,shortcut=False)`，继承原 `cv1/cv2/m/n/add` 及所有共有参数路径。nano 单类实测第 9 层输入/输出 256×20×20，cv1 输出 128×20×20，act=False，add=True。只计算一次 cv1，保留未池化 Z、三次 MaxPool5、cv2 和外层 X 残差。

新路径：`ndp_in` 是 128→16 无偏置 1×1；三个窗口直接读同一 U；`ndp_out` 是 48→256 无偏置 1×1，只有它的 weight 初始化为零。没有新 BN、激活、可学习温度、gate 或 loss；r=16 不参与 width scaling。新初始化放在 CPU `fork_rng(devices=[])` 中，后续原生层和调用方随机序列保持一致。

有效窗口公式：`mu=mean(u)`，`d=u-mu`，`sigma=sqrt(mean(d²)+1e-4)`，`w=softmax(2*tanh(d/sigma))`，`E=sum(w*d)`。
统计在关闭 autocast 的 FP32 区域进行，输出恢复 U 的 dtype，Pin/Pout 沿用原生 AMP。unfold 的有效性 mask 排除零 padding；无效 d 安全置零、logits 为负无穷。窗口可大于 H/W，支持矩形和 1×1。无 mask 缓存、无输入相关缓存、无 detach 截断训练梯度。

逐尺度调用避免无必要的同时显式构造；autograd 仍保留三尺度反向中间量。没有分块/重计算优化。B32、r16、20×20、k13 的单个 FP32 patches 为 132.03125 MiB，不代表整体峰值。

C2PSA、全部 C3k2、neck、Detect 层号 23、输入 `[16,19,22]`、stride8/16/32、one-to-many/one-to-one 和 detach、损失/分配器、原生后处理完全保留。原生 fuse 会去除 Detect 的 one-to-many 推理冗余，因此融合对照比较 one-to-one 原始量及解码结果，而不要求被原生删除的字典仍存在。

## 理论解释边界

常量窗口 E≈0，整个有效窗口加常数时 E 不变；这不是整网光照不变。有效 logits 在 [-2,2]，权重比不超过 exp(4)。输入值未截断，不能宣称完全抗异常值。聚合对窗口内部排列不变，不是显式形状、连通性或方向建模。

25 个值中，1 个 1 与 24 个 0、8 个 1 与 17 个 0 的中心响应分别约 0.275157、0.583010。它们仅展示相同 max 下的分布区分，不代表裂缝判别或支持数量的单调性。

极端有限输入也可能使 FP32 中心化平方溢出；测试报告记录方差有限性与独立 FP64 参考误差，不能只因输出恰好有限就称其正确。没有使用 nan_to_num 隐藏问题。

## 验证范围与证据

- 参数实测：原生单类未融合 2,504,190；NDP 2,518,526；差 14,336，等于 128×16+48×256。
- 共有状态 708 项全部匹配，预训练匹配 606 项；NDP 总状态 710 项。逐键 shape/equal/pretrained 清单在 `artifacts/local_lifecycle_final/weights.json`，不是硬编码的成功结论。
- `tests/test_ndp_sppf_v1.py`：独立 FP64 循环与向量公式/梯度、边角/矩形/窗口大于输入、常量/平移/权重和与比值、排列、小例、极端数值诊断、实际算子 dtype、cv1 一次调用、完整网络结构/RNG/零初始化，以及失败预检返回码、拒绝不完整打包、原生匹配、CUDA FP32/AMP 初始化对照。
- `verify_b19_ndp_sppf_v1.py --local`：两张真实验证图上的开发检测任务（B2/96，固定开发学习率）、MuSGD 中的新参数、Pout 先更新后 Pin 更新、非零 EMA、FP16 原生保存后跨进程恢复 FP32、fuse、AutoBackend、one-to-one 与真实 Validator。还检查 CUDA B1/640 和矩形 640×960 的 FP32/AMP 检测损失。此开发验证不是正式训练、不是服务器预检。
- 本机 Python/PyTorch/GPU 与 b19 环境不同：RTX 2060 6GB、torch 2.7.1+cu118；b19 为 RTX4090、Python3.12.3、torch2.8.0+cu128、Ultralytics8.4.98。入口记录并拒绝未审计的运行环境差异。
- 固定 16 张 val 图片上的 diagnose 和评估 raw matching 导出已做本地执行验证；使用开发更新后的检查权重，仅验证接口与统计，不能作为本实验性能结论。
- 正式 4090 B32/640 原生增强、AMP、warmup/累积/裁剪、最多 128 batch 预检，完整 val/test、正式诊断和最终结果包尚待服务器执行。未创建正式训练完成/预检通过凭证或最终 tar.gz。

FP32 整网零初始化 atol=1e-6、rtol=1e-5；CUDA AMP atol/rtol=1e-3；融合 one-to-one atol/rtol=1e-4。逐输出实测误差在结构/重载报告，既不强求浮点逐位相等，也不对出错路径放宽检查。

新增点卷积算术预算 0.0114688 GFLOPs（MAC=2）仅包括两层投影。本地 PyTorch profiler 能计入的整网算子为原生 5.74197708 / NDP 5.75708748 GFLOPs；CPU 4线程、B1/640、未融合 FP32（3次预热+10次计时）的完整 forward 实测为 108.116 / 118.670 ms。这不是4090性能；unfold、统计、softmax 与访存并不都被 FLOPs profiler 完整计数，不能由此宣称完整成本或低延迟。真实 B32/640 显存与运行耗时由服务器 `checks.json` 输出，当前未测。

## 完整配置与训练过程

运行前读真实服务器 b19 args，并验证历史内容/哈希、启动命令展开、数据 YAML/类别/划分和 8414/2404/1202 图片、val/test 2985/1477 目标。哈希全部图像与标签；不通过移动图片或修改标签凑数字。

完整配置写入 `resolved.json` 的 `config` 和 `evidence.field_comparison`，所有差异保存在 `config_differences`。只允许 model/project/name 以及原始预训练/数据的等价路径表示改变；重建 save_dir，cfg 已展开。服务器预训练固定原始 `yolo26n.pt`，SHA256 为 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`，拒绝训练后 best/last 初始化。

保留 200 epoch、patience60、batch32、640、MuSGD、lr0=.01/lrf=.003、momentum=.937、wd=.0005、seed42、deterministic、workers8、device0、amp、cacheFalse、warmup3、nbs64。在线增强继续保留 hsv=.024/.84/.535、degrees11、translate.17、scale.735、shear3.5、perspective.00055、mosaic1、mixup.135、cutmix.03 等真实参数，全部字段由机器核对，不以这里的短列表代替。

不复制训练循环或调整优化器。在独立预检子进程观察原生 `_do_train`/`optimizer_step`，记录每批实际 LR、累积、AMP 跳步、裁剪前后新参数梯度、实际更新。独立优化器副本使用零当前梯度重放，排除仅衰减/旧动量造成的变化。Pout 更新后，两个新增张量均须有有限非零任务梯度、可表示更新和当前任务造成的差异；不要求首批上游非零，也不要求每个元素非零。

预检停止条件还检查 EMA 经原生 FP16 快照仍保留非零分支和 one-to-one 改变量；上限128 batch。通过后另启全新解释器，从 seed42 和原始 checkpoint 的 epoch0 重新构建正式模型/优化器/EMA。原生 OOM 恢复请求在其首个 batch 变更前重新抛出原异常；不改动共享 Trainer。失败记录保留且不会启动正式训练。

## 交付接口与结果口径

固定入口 `tools/experiments/server_b19_ndp_sppf_v1.sh`，实际 runner/收尾/核验分别为 `run_b19_ndp_sppf_v1.py`、`finish_b19_ndp_sppf_v1.py`、`verify_b19_ndp_sppf_v1.py`，身份文件 `ndp_experiment.py`。

`train` 自动且仅先执行一次独立预检；也提供 `preflight/test/diagnose/package`。一实验一共享 flock 覆盖完整进程树及代码部署；每阶段独立 attempt、完整 SHA/实际命令/PID/console/process_status/exit_status。Python、tee、shell 非零状态穿透，INT/TERM 转发至独立进程组；保留旧 attempt 和 run，不自动 resume/覆盖。

`test` 从训练验证规则选出的同一 best.pt 执行全量 FP32 val/test，独立进程和目录。640/batch32、conf=.001、iou=.7、max_det300、rectTrue、augmentFalse；本固定版本 `quantize=None` 等价 half=False（旧 half 参数已更名）。保留原生 end-to-end 后处理。保存真实 split/图像/目标数量、权重/代码/数据哈希、融合状态、完整 args、P/R/mAP50/AP75/mAP50-95、逐 IoU AP、曲线、混淆矩阵、预测 JSON 和 `matching_stats.npz`。

原生 P/R 对应最佳 F1 操作点。额外固定规则只在 val 的完整同分阈值边界中，选择 Precision≥0.85 时 Recall 最大的阈值，然后冻结到 test；无可行阈值明确 unavailable，不从 test 选择。记录实际达到的 Precision/Recall/FPPI，不称为精确相同 Precision。若原 b19 best.pt 存在，独立同口径补评且不重训；不把历史四舍五入数值当涨点证据。

`diagnose` 固定字典序前16张 val，保存文件列表和哈希；同权重 eval，只通过 Pout 输出 hook 令 Delta=0，完整 native SPPF 仍执行。两组均640、conf=.25、iou=.7、max_det300、相同 letterbox 与原生后处理。每尺度保存有效计数 m、权重熵、H/log(m)（m=1明确为0）、最大权重、E范数/有限性/跨尺度相关、Delta/Y0，NPZ 保留逐位置统计。输出 IoU.5/.75 的 TP/FP/FN、补回/丢失目标、定位 IoU 变化、配对可视化和预测/GT。聚合系数不命名前景置信度，不把边界低熵称为聚焦，不把16张诊断称为全量评估或重新训练消融。

`package` 检查同一次正式训练来源的成功预检与 train/test/diagnose attempt、权重/源码/数据哈希及必要产物。打包 best/last、args/results、评估与诊断、全部 provenance/console/进程状态、依赖版本、原始哈希、整个已提交源码 archive（含注册/YAML/文档，不含数据集）。恢复时将 `source.tar` 解压到独立目录，以该目录优先 import，再读取权重。

产物仅在成功后发布至本 worktree 的 `artifacts/experiments/yolo26n_b19_ndp_sppf_v1_<12位实际提交>.tar.gz`，伴随外部 `.sha256`、`.manifest.json` 和大小/哈希 JSON。验证归档逐成员 manifest、gzip 可读和 `sha256sum -c` 后才输出实际路径/大小/SHA256；已有包拒绝覆盖。

服务器复制命令见 [操作文档](b19_ndp_sppf_v1_server.md)。
