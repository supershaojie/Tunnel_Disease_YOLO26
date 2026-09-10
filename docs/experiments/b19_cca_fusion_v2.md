# CCA-Fusion v2：Reliability-Constrained CCA

## 实验身份与审计

从正式 CCA v1 `42072916046b7598bf90640fcde907a62265406d` 创建独立 worktree：
`.worktrees/exp-yolo26n-b19-cca-fusion-v2`，分支 `codex/exp-yolo26n-b19-cca-fusion-v2`。
原主目录与 v1 worktree 的未跟踪产物保持原样。本次不创建 PR，不连接 AutoDL，不启动正式200轮训练。
服务器 WORK 固定 `/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_CCA_Fusion_v2`，
RUN 固定 `yolo26n_b19_cca_fusion_v2`。

完整阅读 v1 模块/YAML/注册/parser、run/finish/verify/server/deploy、两份v1说明、
b19_reference.json、b19_dataset_manifest.json、原来源证据，以及指定v2说明。
从正式v1分析包实际读取保存的112项训练args、val/test/b19比较指标和16图诊断。
[v1实测审计](evidence/cca_v2_source_audit.json)记录所读路径、精确数值及配置差异。

## 唯一假设与固定公式

v1 val AP50提高0.0069194，但AP75下降0.0218966；test Recall下降0.0216102，
mAP50-95下降0.0145960，AP75下降0.0437836。训练已达200轮平台。
原诊断的高熵分布和不小的R/U扰动支持一次定向检验：可靠性约束能否减少不确定correspondence对裂缝定位的破坏。
这些是机制动机，尚不证明因果，也不证明v2涨点。

v1：`a=softmax(mask(4*cos(q,k)))`，`D=sum(a*(V_neighbor-V_parent))`，`Y=cat(U+Wo(D),L)`。

v2固定为：

```text
s_i = 4*cos(q,k_i) + 1[i=4]*1.0
a = softmax(mask(s))
H = -sum(a*log(clamp_min(a,1e-6)))
H_norm = clamp(H/log(K_valid),0,1), K_valid<=1 时定义为0
C = clamp(1-H_norm,0,1)
G = 0.25 + 0.75*stopgrad(C)
D = sum(a*(V_neighbor-V_parent))
R_raw = Wo(D)
R = 0.50*G*R_raw
Y = cat(U+R,L)
```

CENTER_PRIOR=1.0，RELIABILITY_FLOOR=0.25，RESIDUAL_SCALE=0.50，rank=16，similarity_scale=4.0。
中心4按行优先排列；固定不可学习先验优先尊重nearest parent，明显匹配的非中心邻居仍可胜出。
K_valid按实际粗尺度几何计算，标准图角/边/内部为4/6/9，1×1时为1；不从权重非零数量推断支持集。
对分母先clamp到2再取log，并显式where处理K<=1，不会因where未选分支除零。
G形状为B×1×H_fine×W_fine，范围[0.25,1]，有效gate范围[0.125,0.5]。
C gate detach阻断直接通过熵放大gate的梯度，weights→delta→Wo路径完整保留。
0.25 floor保留弱学习路径，0.5 scale约束最大直接修正；不扫任何参数组合。

四个bias=False的1×1投影完全复用v1；Wq/Wk/Wv默认Conv2d初始化，Wo全零；CPU fork_rng(devices=[])
不影响原生后续层RNG。Q/K normalize eps=1e-6、FP32 softmax、invalid mask及显式父差值保持。
熵和缩放也在autocast(False)内；输出残差转回原分支dtype。无输入尺寸缓存、无新可训练参数、无动态创建层。
没有5×5、多头、多处CCA、额外卷积/门控模块或N×N矩阵；只验证一个局部残差机制，不叠加创新。

## 图结构、参数与初始化公平性

v2 YAML与v1仅第12层类名不同。输入[11,6,10]及[256,128,256]通道，输出384通道。
layer11 nearest、layer13 C3k2、P4→P3 concat、PAN、Detect[16,19,22]、strides[8,16,32]，
nc=1/end2end=True/reg_max=1均保留。
v1文件与类保留，旧checkpoint可加载。只把v1的得分和残差应用提取为两个可覆写方法，
其原始数学操作和state_dict不变；v2继承公共初始化/局部对应/shape/concat逻辑。

v1与v2 CCA都是14,336个learnable parameters，v2相对v1新增0个。
原生模型2,504,190，v1/v2总参数均2,518,526。
Wo=0使初始R精确为0；结构检查覆盖640×640与640×960，加载检查逐tensor比较708项公共状态，
原始COCO权重应加载606项，新增4项只在model.12。one-to-many与one-to-one必须严格相等。
正式训练必须重新从原始yolo26n.pt加载，SHA256
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`；绝不从v1 best.pt warm start。
否则预训练条件不同，无法判断机制贡献。

## b19全量配方与公共生命周期

复用v1 resolve_recipe/manifest/launcher/structural_checks/AuditedTrainer/observed_optimizer_step/
preflight/EMA/reload/fuse/Validator以及完成阶段。独立v2入口只传实验身份、类型和机制统计，
没有复制第二套训练超参数或删掉失败检查。112项以原始b19 args的固定SHA256及reference逐项核验；
v1正式保存args也已全量核对，差异只有model/project/name/save_dir及原权重路径表达。
服务器继续逐字段核验实际数据manifest、导入目录、源码摘要、初始权重、最终优化器和EMA。
只有路径/身份表达可以变，数据内容及预训练SHA仍需完全一致。

epochs=200、patience=60、imgsz=640、batch=32、workers=8、device=0、seed=42、deterministic=True、
amp=True、cache=False、MuSGD、lr0=0.01、lrf=0.003、momentum=0.937、weight_decay=0.0005、
warmup=3.0/0.8/0.1、cos_lr=True、close_mosaic=10及全部在线增强都来自原记录。
不改全局默认配置、loss、MuSGD、trainer/clip/GradScaler/EMA实现。
服务器train强制独立子进程真实batch32/640预检，FP32与AMP、最多128批观察；
Wo及Wq/Wk/Wv必须全部出现有限非零任务梯度和排除weight decay伪更新后的有效更新。
记录首次gradient/update step、参数norm前后、group/lr/decay、scaler、clip和EMA。
失败就停止，不取消detach、不改LR、不降batch、不用旧预检模型正式训练。

## 模块包实际参考

ZIP路径 `D:/7.21yolo26改/YOLO26缝合.zip`，SHA256
`a0c197f38e1510291a9a762fc7dc0a8c13694742fff9e3132c50a3ee32ddadcf`。
本次再次直接完整读取下列两文件，见[读取与哈希证据](evidence/cca_v2_reference_sources.json)。

| 包内文件（YOLO26缝合/ultralytics/nn/newsAddmodules/） | SHA256                                                           |
| ----------------------------------------------------- | ---------------------------------------------------------------- |
| DPCF_INFFUS2025.py                                    | f4a0606dc51ecb6f80feb65febb68f70f9d097c449144664acc00f0d7da9dd10 |
| RLAB_fusion_2025CVPR.py                               | 9e166e61f96cdfe6588679092de06960ad427d127a80c066c0d538b4a07a6d62 |

DPCF：参考双输入跨尺度组织和通道投影，不采用AdaptiveCombiner标量混合、四块拼接、方形双线性插值及BN。
RLAB：参考Q/K/V/out投影和关系计算组织，不采用DSUB/EUB/ResidualRB/refine/PixelShuffle、
实际N×N矩阵和forward内\_proj_if_needed创建可训练层。未整体覆盖模块包、未复制其现成模块或引入依赖。
本次没有额外论文检索或未经核实的文献效果声明。

## 机制诊断与正式评估

沿用固定排序前16张验证图和best.pt，同图同权重保留residual on/off原始one-to-one变化及IoU0.5/0.75匹配。
另存corner/edge/interior的center weight、noncenter总权重、entropy、H_norm、C、G、K_valid、
expected correspondence displacement，均包含count/mean/std/min/p50/p90/max。
位移定义为粗尺度邻域偏移的加权期望向量的范数，不是GT alignment error。

残差保存每位置通道L2范数的R_raw/(U+eps)、R/(U+eps)，eps=1e-6；同时记录0.5*G。
actual_over_ungated_same_correspondence比较相同v2对应下的原始残差；非零raw处应等于0.5*G。
actual_over_v1_no_prior_same_projections另去掉中心先验，仍使用本次投影权重，只是机制理论比较，
不是加载v1 best，也不是重新训练的消融；该比率同时受correspondence变化影响，不要求小于0.5。
零分母单独计数，ratio为null，不用NaN。所有统计断言有限。

训练后只用val选出的best.pt，固定FP32完整val/test、P/R/AP50/mAP50-95/AP75、PR/P/R/F1曲线、混淆矩阵、
predictions JSON、数据/权重/环境/源码provenance及同条件b19比较。
补充R@P>=0.85只在val冻结阈值，再一次应用test，不替代标准mAP、不用test挑参数。
关注test Recall从v1 0.762182恢复并超过b19 0.783792，mAP50-95超过0.507770，AP75修复0.492964并争取超过0.536748。
这些是观察目标，不是实现验收的虚构涨点承诺。

## Packaging fix（独立基础设施修复）

原归档把transient attempt日志和console链接混入主包，tarfile校验会追索缺失link target而失败。
删除这两个收集入口；run内*.attempt.*目录和失效符号链接不作归档输入；有效的最终文件链接物化为普通文件，保留正式train/evaluation/provenance日志。
临时attempt仍原地保留用于排查，不依赖其console作为主包成功条件。成功stage exit receipt仍在package入口核验。
写tar时dereference=True把已选择的文件物化为普通成员，重复硬链接也不生成依赖其他成员的link。
验证前要求成员isfile，再逐一读内容校验SHA256/大小；读取gzip直到尾部验证CRC。
输出package_manifest.json、SHA256 sidecar、package_result.json和关键成员列表。
旧package_result不回流到新包。此修复不计入CCA模型创新。

Deleted: 删除归档对transient attempt全目录/console别名的收集；将v1硬编码实验身份替换为显式参数。
Reused: v1模型投影/局部索引与b19训练/评估/归档审计基础设施。
归档修复需要少量新增普通成员校验与回归测试以证明链接缺失不再影响正式结果，单纯删除不能验证完整性。

## 本地验证与服务器待执行

本地环境 Python 3.11.15 / PyTorch 2.7.1+cu118 / RTX 2060 6GiB，区别于 b19 的3.12.3 / 2.8.0+cu128 / RTX4090。
[完整生命周期记录](evidence/cca_v2_local_validation.json)保存逐tensor/输出误差、CPU/CUDA/AMP梯度更新、EMA/reload/fuse及Validator证据。
[112项配置与数据manifest核验](evidence/cca_v2_configuration_comparison.json)已通过：训练8414张/10243实例，val2404/2985，test1202/1477；全部内容摘要与原记录相等。

| 检查                                  | 实测结果                                                                                                     |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| native公共状态                        | 708项全部逐tensor严格相等                                                                                    |
| 原始预训练加载                        | 606/708 native；606/712 v2，新增仅4项CCA投影                                                                 |
| one-to-many / one-to-one 初始误差     | 最大误差0                                                                                                    |
| v1 / v2 CCA参数                       | 14,336 / 14,336，新增0                                                                                       |
| CPU FP32 / CUDA FP32合成任务          | Wo第1步有效更新，其余投影第2步解锁                                                                           |
| CUDA AMP合成任务                      | GradScaler真实跳步被记录；Wo第4次尝试更新，其余投影第6次尝试有效更新，未改LR/AMP/detach                      |
| 生命周期                              | 非零EMA、FP16保存、新进程FP32严格重载、native对照fuse及Validator均通过                                       |
| 合成batch1模型分辨率                  | 640×640及640×960，CUDA FP32/AMP loss与反向通过                                                               |
| 固定16图diagnose与FP32评估            | 实际入口运行通过；包含v2统计及PR/P/R/F1/混淆矩阵/预测JSON，使用合成数据                                      |
| 归档                                  | 硬链接物化、attempt排除、完整v2 package及缺少b19比较时拒绝均通过；CRC/逐文件SHA通过                          |
| dangling symlink真实创建用例          | Windows账户无创建权限，明确skip；已运行的硬链接/attempt排除用例证明临时目录不入主包，Linux需复验符号链接用例 |
| Ruff 0.12.12 / bash -n / diff --check | 通过；bash语法检查不等于Linux tmux/flock执行                                                                 |

单模块合成batch32/40×40与20×20特征的FP32前后向约12.56ms，峰值分配332,752,896字节，包含存活张量。
该测量不是完整训练资源消耗，且不使用THOP输出宣称精确GFLOPs；不是服务器真实batch32预检。

API reference生成器已执行；Windows默认GBK解码失败后使用PYTHONUTF8=1重跑成功。
生成器的Windows导航路径与默认上游源码链接会失真，仅保留本实验所需的v2引用页和导航，修正为仓库实际分支链接；未改生成器。
Markdown使用已存在的Prettier 3.6.2、tab-width=4、print-width=120格式化，没有新增依赖。
当前未声明任何正式实验精度。
服务器命令见[操作模板](b19_cca_fusion_v2_server.md)。服务器真实batch32/640、匹配b19环境的preflight、
Linux tmux/flock、正式200e、best.pt val/test/diagnostics/比较及正式打包均需后续实际执行。

最终联合回归：**30 passed，1 skipped**（Windows符号链接权限），无失败；耗时62.98秒。
本地检查命令（所有路径均在v2 worktree；Python使用实际yolo26解释器）：

```bash
python -m pytest -o addopts='' tests/test_cca_fusion.py tests/test_cca_fusion_v2.py -q
python -m tools.experiments.verify_b19_cca_fusion_v2 --weights /path/to/original/yolo26n.pt --output artifacts/local_verify_cca_v2
ruff check ultralytics/nn/modules/cca_fusion.py ultralytics/nn/modules/cca_fusion_v2.py ultralytics/nn/modules/__init__.py ultralytics/nn/tasks.py tools/experiments/run_b19_cca_fusion.py tools/experiments/run_b19_cca_fusion_v2.py tools/experiments/finish_b19_cca_fusion.py tools/experiments/finish_b19_cca_fusion_v2.py tools/experiments/verify_b19_cca_fusion.py tools/experiments/verify_b19_cca_fusion_v2.py tests/test_cca_fusion_v2.py
bash -n tools/experiments/server_b19_cca_fusion_v2.sh
bash -n tools/experiments/deploy_b19_cca_fusion_v2.sh
git diff --check
```

v2服务器test/package要求存在完成的同条件b19比较，缺少时失败；无任何模型或配方额外改动。
