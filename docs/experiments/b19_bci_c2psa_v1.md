# b19 + BCI-C2PSA v1

本实验是双分支通道交互 C2PSA（Branch-Conditioned Channel Interaction）的独立候选，尚无正式训练指标，不能声称涨点。
分支 `codex/exp-yolo26n-b19-bci-c2psa-v1` 从原生 b19 `4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6` 创建。
按照本轮专项要求 commit/push，不创建 PR，不自动 SSH。

## 唯一结构变化

`ultralytics/cfg/models/26/yolo26n-bci-c2psa-v1.yaml` 直接源自原生 `yolo26.yaml`，改成单类并且只将第 10 层类型改成 `C2PSA_BCI`。
SPPF、Neck、Detect、全部层号/from、原生空间注意力、FFN、detach、双检测分支、损失和后处理保持原样。
实际模型为 24 层、stride 8/16/32、reg_max=1、end2end=True；第 10 层 256 通道，半分支 128，PSA 堆栈一次，原生注意力 2 heads、key_dim=32、head_dim=64。

新增参数全部在 `model.10.bci.*`，未引入 `.base` 包装。构造原生 `cv1/cv2/m` 后，使用隔离 CPU RNG 创建新分支。
框架 `initialize_weights` 对裸 Conv2d 不重置权重，预训练也没有这些新增键，输出投影在正式 trainer 初始化边界仍为零。
其他通道、重复次数或扩展率明确报错。

固定公式与源码对应：

```text
a,b = native_cv1(X).split(128,128)
b0 = native_m(b)
Q = dwq(pq(a)); K = dwk(pk(b0)); V = pv(b0)  # B,32,H,W
Qc = Q.flatten(2) - mean(Q.flatten(2), -1)
Kc = K.flatten(2) - mean(K.flatten(2), -1)
Qn = L2_normalize(Qc, dim=-1, eps=1e-6)
Kn = L2_normalize(Kc, dim=-1, eps=1e-6)
A = softmax(4.0 * Qn @ Kn.transpose(-1,-2), dim=-1)  # B,32,32
D = reshape(A @ V.flatten(2) - V.flatten(2), B,32,H,W)
Y = native_cv2(concat(a, b0 + po(D)))
```

六个卷积都无 bias、BN、激活；pq/pk/pv 为 128→32 的 1×1，dwq/dwk 为 32 组 3×3，po 为 32→128 的 1×1。
只有 po 置零。投影遵循原生 AMP，中心化、归一化、两次矩阵乘法、softmax 和减法在显式禁用 autocast 的区域执行 FP32，D 转回 b0 dtype 后进入 po。
没有排序、跨 batch 缓存、额外温度、gate 或新损失。`BCI.interaction` 同时服务正常前向、独立数值测试和诊断，正常前向不保存注意力矩阵。

## 实际来源核验

原始权重 `E:/PycharmProjects/Tunnel_Disease_YOLO26/yolo26n.pt` 的 SHA256 与要求一致：
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`。

实际 b19 归档目录：
`E:/ditieyolo26跑结果/8.12离线在线/b19 200e/b19_yolo26n_e200_train_val_test_20260823_224153/train_run/`。
其中 args.yaml 的 112 个字段逐项匹配，args/results SHA256 分别为：

```text
b08b915756bf85c91d3356586a651a37156d71867b84b6e8493d75c9568642b9
44795068d0ebc687c29ff7b7b8cbf337469546b2d4dd515cb0616c270bdf9507
```

本地数据实测 train 8414 张/10243 实例，val 2404 张/2985 实例，test 1202 张/1477 实例。
本地保存的历史展开启动记录能够逐字段复现 args；服务器 `/root/autodl-tmp/experiment_backups/records/b19_launcher_expanded.txt` 未远程读取，部署时必须重新读取核验，缺失即报错，不以摘要代替。
正式阶段还核验所有图片内容、标签内容、原始权重、完整配置、环境和源码身份。

模块包实际路径为 `D:/7.21yolo26改/YOLO26缝合.zip`，SHA256 为
`a0c197f38e1510291a9a762fc7dc0a8c13694742fff9e3132c50a3ee32ddadcf`，与附件内容匹配。
只读取 `HMHA_2025CVPR.py`；没有导入其代码或依赖。
机制参考 [Restormer, CVPR 2022](https://arxiv.org/html/2111.09881v2) 的通道注意力及 [HINT 作者仓库, ICCV 2025](https://github.com/joshyZhou/HINT) 的通道交互思想。
模块包文件名中的 CVPR 标签不是 HINT 的会议依据，恢复任务效果不构成本检测实验的收益证据。

## 已完成的本地验证

环境：Windows、Python 3.11.15、PyTorch 2.7.1+cu118、RTX 2060 6GB；没有升级或重装已有训练环境。
数据副本只用于开发检查，不改主工作树和其他实验。

| 检查                                     | 实测结果                                                                      |
| ---------------------------------------- | ----------------------------------------------------------------------------- |
| 未融合原生/候选/新增参数                 | 2,504,190 / 2,521,150 / 16,960                                                |
| 共有状态（含 BN buffers）                | 708 项全部一致                                                                |
| 原始预训练匹配                           | 606 项；不匹配键及 nc=80→1 原因完整保存                                       |
| 同 seed 原生构造随机序列                 | 一致                                                                          |
| 零投影模块 train/eval                    | 逐元素一致                                                                    |
| 零投影整网 FP32 eval                     | 640×640、640×960 全部原始输出逐元素一致                                       |
| 独立公式参考                             | 不对称 3 通道×5 空间 float64 标量参考通过                                     |
| 恒定/近恒定/低精度/矩形                  | 有限，softmax 行和为 1；独立完整投影公式通过                                  |
| 旁路条件                                 | 非恒定空间扰动改变 A 和残差，原始 a 未被改写                                  |
| 真实检测 loss / MuSGD                    | 2 张真实标注、96 输入；首步仅 po 有任务梯度，第二步六个张量都有任务梯度及更新 |
| 参数组                                   | 六个权重分别且仅一次进入原生 MuSGD 的 muon 组                                 |
| EMA / FP16 checkpoint / 新进程 FP32 重载 | 通过，状态逐键一致                                                            |
| 融合                                     | 候选与原生对照，全部保留的 one-to-one 原始/解码输出通过 atol=rtol=1e-4        |
| 真实 Validator / AutoBackend             | 2 张真实图、FP32 通过，hook 确认 BCI 与 one-to-one 执行                       |
| CUDA 检测前后向                          | batch=1，640×640 和 640×960，FP32/AMP 均有限                                  |
| AMP 运算拦截                             | 两次 BMM 与 softmax 实测 FP32                                                 |
| PyTorch profiler GFLOPs                  | 原生 5.74197708；候选 5.757235704                                             |
| 单元测试                                 | 7 passed；新增代码 Ruff 检查通过                                              |

FLOPs 是实际 1×3×640×640、未融合 eval 的 profiler 可计数算子统计，不包含所有标量操作；不能与框架 fuse 后显示的估计值混用。
GPU batch=1 峰值显存和实际 TF32 设置见 [本地验证 JSON](evidence/bci_v1_local.json)。本机 CUDA matmul TF32=False，cuDNN TF32=True。
完整匹配键、逐字段基线证据分别见 [weights](evidence/bci_v1_weights.json) 和 [b19 sources](evidence/bci_v1_b19_sources.json)。

第一次新进程重载比较因父子进程 CPU 线程数不同而失败，现保存并恢复线程数后精确比较通过，没有放宽容差。
首次失败日志保留于 worktree 的 `artifacts/local_check_01/`。最终开发检查为 `artifacts/local_check_03/`。
文档生成器已运行；其 Windows 输出产生的无关导航路径/缩进变化被移除，仅保留新模块引用。

本地测试不是正式 warmup/accumulate 预检，也不是召回率实验。2 张图的 Validator 指标不用于判断收益。
AutoDL batch=32、640 的真实预检、正式训练、完整 val/test、训练后 diagnose/package 均待服务器执行；未生成任何正式成功标记。

## 服务器流程与证据

[服务器操作文档](b19_bci_c2psa_v1_server.md)；交付目录另提供替换了真实完整提交 SHA 的可复制版本。
`train` 自动启动独立 preflight 子进程，成功后重新 seed 并从原始权重重建正式 trainer；不继承预检模型、BN、EMA、优化器或迭代器。

preflight 使用原生训练循环，最多 128 个 batch，保留原始 warmup、LR、accumulate、GradScaler、unscale、clip=10、MuSGD、EMA 时序。
实际 optimizer hooks 记录成功 step，不使用 step 返回值。逐张量保存裁剪前后梯度、LR、scale/跳步、参数范数、更新量、首个有效更新时间。
独立优化器副本以零当前梯度重放相同步骤，确认更新并非仅由 weight decay/旧动量导致；不修改真实优化器。
满足六个张量有效任务梯度与真实更新且至少两个成功 step 后提前结束；否则报错保留现场。
同一真实预处理 batch 还用于独立副本的 FP32 检测 loss 前后向。正式训练始终保持 batch32 和 AMP，不接受 OOM 自动降 batch。

`test` 对固定 best.pt 在独立进程执行 val/test，640、batch32、conf=.001、IoU=.7、max_det=300、rect=True、augment=False、workers=8、device=0。
本版本以 `quantize=None` 表示 FP32（对应 half=False），同时检查实际参数 dtype，记录 TF32，不增加额外 NMS。
保存原始 P/R、AP50、AP75（真实 IoU 索引 5）、mAP50–95、曲线、混淆矩阵、预测、配置、速度与内存。
AP95 若作为简称仅指 mAP50–95，不能当成 IoU=.95 的 AP。
R@P≥.85 使用原生 IoU=.5 匹配和同置信度整体边界，只从 val 选择阈值并冻结给 test；无法达到目标则明确为空，并保留 R–FPPI 曲线。
默认 P/R 是框架最优 F1 操作点，不与冻结阈值结果混用。
服务器可访问 b19 best.pt 时，自动通过同入口补齐基线独立评估，不重训；否则记录 unavailable。

`diagnose` 使用排序后前 16 张验证图，固定清单与哈希；记录 A 行和误差、有限性、熵、对角/非对角质量、最大概率、相关分布、残差相对范数和权重/预检梯度。
同 checkpoint 的残差开关对比只表示功能诊断，不是重训消融。32×32 图是全局通道关系，不是裂缝定位；均匀 A 或近零残差如实报告。

所有阶段独立 attempt、日志、PID、HEAD、命令和退出状态；实验锁防止并发重复启动，不覆盖已有正式 run，不自动 resume。
`package` 核验必要阶段成功、完整评估图数和实例数、预检校验、配置/代码/数据/权重身份及文件哈希。
只打包 best.pt、必要训练/评估/诊断/预检证据、源码归档、环境和 manifest，不包含 epoch 权重、原数据、其他实验、完整 .git 或模块 ZIP。
输出 `artifacts/experiments/yolo26n_b19_bci_c2psa_v1_<12位SHA>.tar.gz` 和 `.tar.gz.sha256`，逐成员校验、gzip 完整性校验后打印绝对路径/大小/SHA256。

## 复用和变更范围

复用 native C2PSA、解析重复/宽度处理、DetectionTrainer/MuSGD/EMA/Validator，以及已有 MSI 分支的 b19 来源核验、进程/结果保护和打包基础设施。
参考 PKC 修复提交 `1b4b98ede9499eae225010893c12c57ac5c3cd5f` 的有界更新证据思想；没有移植 PKC/MSI 模型。
移植时删除旧分支的 FFN/identity/bias 选择器和诊断、32-batch 窗口及仅输出投影变化判定，改为 BCI 的六个参数和 128-batch 证据。
原生训练、优化器、EMA、损失源码均未修改。

Deleted: 运行基础设施移植中的旧实验专属验证/诊断与过短预检窗口；主模型唯一替换为第 10 层 C2PSA 类型。
本次是独立候选功能，必须新增模块、专属测试和可复现入口；删除或迁移原生模型无法实现该固定公式。
