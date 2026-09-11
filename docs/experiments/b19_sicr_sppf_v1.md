# b19 SICR-SPPF v1 实施与审计

本实验仅把 YOLO26n 第 9 层原生 SPPF 替换为 Stage-Increment Context Refinement SPPF（分级增量上下文精炼空间金字塔池化）。这是一个固定、尚待正式训练检验的 v1；没有多候选、test 调参、组合创新或自动 v2。

## 审计先于实现

- 工作分支：`codex/exp-yolo26n-b19-sicr-sppf-v1`。
- 基点：`4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6`，来自原始 b19 归档的 `environment/git_state.txt`。
- 独立 worktree：`E:/PycharmProjects/Tunnel_Disease_YOLO26/.worktrees/exp-yolo26n-b19-sicr-sppf-v1`。
- 远端：`https://github.com/supershaojie/Tunnel_Disease_YOLO26.git`。只推送实验分支，PR 基分支为同一 b19 基点的 `exp-dataset-augfirst-diverse5x-randomsplit`。
- 原工作区分支为 `exp-dataset-augfirst-diverse5x-randomsplit`，有未跟踪 `.worktrees/` 和 `artifacts/`；未删除、移动或覆盖这些用户内容。
- 原始 b19 归档：`E:/ditieyolo26跑结果/8.12离线在线/b19 200e/b19_yolo26n_e200_train_val_test_20260823_224153`。
- 实读 `train_run/args.yaml`、`results.csv`、训练和测试日志、数据 YAML、实验说明、Git 和框架环境记录。训练 CSV 有 200 行，最高 val mAP50-95 的记录为 epoch 196，值 0.50285。
- `args.yaml` SHA256：`b08b915756bf85c91d3356586a651a37156d71867b84b6e8493d75c9568642b9`。
- `results.csv` SHA256：`44795068d0ebc687c29ff7b7b8cbf337469546b2d4dd515cb0616c270bdf9507`。
- b19 原 shell 启动、finish/package 脚本未收录于归档。已实际读取历史 SIR server/runner、CCA v2 finish/deploy 和 NDP 公共审计；`b19_launcher_expanded.txt` 是历史用户文档第 9 节的展开记录，经过全字段核对，不能称为从服务器取回的原始 shell。

原始来源和实际文件哈希见 `tools/experiments/sicr_sources.json`；完整 b19 配方见 `b19_reference.json` 和按字节保留的 `b19_archived_args.yaml`。没有在实现中重新手抄配方。

## 既往 SPPF 经验审计

实际读取 SIR v1、SIR v2 和 NDP v1 的模块代码及结果 tar 中 JSON。下表均为同口径 FP32 归档评估数据；没有重训或以历史权重初始化本实验。

| 模型   | Val mAP50-95 | Test P      | Test R      | Test mAP50  | Test mAP50-95 | Test AP75   |
| ------ | ------------ | ----------- | ----------- | ----------- | ------------- | ----------- |
| b19    | 0.502648280  | 0.874590232 | 0.783792155 | 0.870056744 | 0.507769939   | 0.536747589 |
| SIR v1 | 0.508045202  | 0.857508452 | 0.800270819 | 0.876482625 | 0.499716584   | 0.511854785 |
| SIR v2 | 0.504029520  | 0.878820135 | 0.779959377 | 0.872795211 | 0.506946685   | 0.531789182 |
| NDP v1 | 0.499187593  | 0.884960866 | 0.762356127 | 0.858303561 | 0.497944375   | 0.521134397 |

SIR v1 用空间/通道 router 产生 `0.5*tanh(logit)*increment`，再沿池化阶段累积修正；SIR v2 保留 router，改为各阶段独立修正。NDP v1 是 16 通道投影、同一 U 的 5/9/13 局部标准化分布加权池化，再由零初始化输出投影注入原生输出；它不是 SIR router。SICR 不继承这些模块代码，只借用经过审计的实验基础设施。

SIR v1 的 val 增益伴随 test mAP50-95/AP75 下降是实测现象；“动态重分配可能导致泛化或定位扰动”属于待验证解释，不能据此证明因果。NDP 归档已存在正式结果，因此没有沿用旧实现文档中“尚无正式结果”的过时表述。

## 模块包读取证据

ZIP：`D:/7.21yolo26改/YOLO26缝合.zip`。

SHA256：`a0c197f38e1510291a9a762fc7dc0a8c13694742fff9e3132c50a3ee32ddadcf`，与任务指定值一致。ZIP 顶层中文名存在编码显示差异，实际按唯一相对后缀定位成员并读取字节，不依赖解压目录的名字。

| 实际读取的成员（相对 YOLO26 缝合目录）                | SHA256                                                             | 采用与未采用                                                                  |
| ----------------------------------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------------- |
| `ultralytics/nn/modules/block.py` 的 SPPF             | `0086ef587d984f657f28adda99a6186f69bf27b77a05650fca3b99995ed78fc8` | 核实 cv1/cv2/m/n/add；复用当前仓库原生 SPPF                                   |
| `ultralytics/cfg/models/26/yolo26.yaml`               | `b1d1fa0c69eced64f9939536038bd8d697c32e72200b34187a767de431af7ef0` | 核实第 9/10 层和 P5 concat；实验配置从当前 b19 YAML 生成                      |
| `ultralytics/nn/newsAddmodules/HLKConv_TGRS2025.py`   | `ab4407a9cae37bf68b7409a18b85e652046c2aafd7d0f990d90c37b561216d06` | 借鉴 depthwise 低成本空间建模；不采用整块 HLKConv、dilation 搜索或额外 concat |
| `ultralytics/nn/newsAddmodules/StripConv_AAAI2026.py` | `a37cdb8da4c496b45da48240eaab37b235d59791e30bc0be3b7a2fa2626744b9` | 借鉴方向分解卷积；不采用 `x * attn`、19 核、attention、DropPath、timm/mmcv    |

未复制 add26 YAML，未覆盖 Ultralytics，未加入 mmcv/mmengine/timm 依赖。文件名中的年份/会议名只用于定位用户材料，不作为出版证明。

## Motivation

Conventional SPPF repeatedly pools nested feature maps, which effectively enlarges receptive fields but also introduces strong redundancy among adjacent pooling stages. For thin tunnel cracks, aggressively reweighting the entire high-level feature may improve coarse detection while disturbing localization and generalization.

640 输入在 P5/32 对应 20×20 特征。该处目标是组织已有的弱裂缝响应与上下文，不宣称恢复原始高分辨率细纹理。

## SICR idea

SICR-SPPF explicitly decomposes the **incremental contextual information** introduced by each consecutive pooling stage:

```text
Z0 = cv1(X)
Z1 = MaxPool5(Z0); Z2 = MaxPool5(Z1); Z3 = MaxPool5(Z2)
D1 = Z1 - Z0; D2 = Z2 - Z1; D3 = Z3 - Z2
Ri = Refine(Di, ki), ki = [3, 5, 7]
alpha_i = 0.10 * tanh(theta_i), theta = [0, 0, 0] initially
Zi' = Zi + alpha_i * Ri
Y = cv2(Concat[Z0, Z1', Z2', Z3']) + X
```

It performs **scale-specific directional refinement** only on these increments. The refined increments are injected back through **bounded learnable residual coefficients**, preserving the original SPPF representation as the dominant path.

每个 `SICRIncrementRefine` 依次是 `Conv(c,c,(1,k),g=c)`、`Conv(c,c,(k,1),g=c)`、`Conv(c,c,1,act=False)`。前两层含 BN+SiLU，最后一层含 BN、无激活。三支都读未经修正的相邻池化差值，没有递归传播修正、abs/softmax、归一化概率或 image-dependent gate。

`SICRSPPF` 直接继承原生 `SPPF`，保留 cv1/cv2 的字段、形状、激活、shortcut。新增卷积初始化放在 `torch.random.fork_rng(devices=[])` 内，保证后续原生层使用相同 RNG 起点；theta 三个全局参数归零。拒绝 k≠5、n≠3、alpha_max≠0.10 的构造请求。

`|alpha|≤0.10` 是系数约束，不等于修正特征范数一定低于 Zi 的 10%；Ri 无硬性幅度上界。FP32 中 0.10 表示为约 0.10000000149，属于同一浮点常数的表示误差。

## Claimed advantages to test empirically

- lower redundancy between pooling stages;
- directional context suitable for elongated crack structures;
- larger receptive context with low compute at P5;
- identity-like initialization relative to native SPPF;
- bounded correction to reduce overfitting/localization disturbance.

这些是待检验假设，没有提前宣称显著提高或证明有效。

## 拓扑与复杂度

| 审计项                                   | b19                    | SICR v1                       |
| ---------------------------------------- | ---------------------- | ----------------------------- |
| scale / nc                               | n / 1                  | n / 1                         |
| layer 8                                  | C3k2                   | C3k2                          |
| layer 9                                  | SPPF [1024,5,3,True]   | SICRSPPF [1024,5,3,True,0.10] |
| layer 10                                 | C2PSA                  | C2PSA                         |
| layer 9 输入/输出                        | 256 / 256              | 256 / 256                     |
| cv1 隐藏通道                             | 128                    | 128                           |
| 640 输入层 9 形状                        | B×256×20×20            | B×256×20×20                   |
| P5 bottom-up concat                      | [-1,10]                | [-1,10]                       |
| Detect 输入 / stride                     | [16,19,22] / [8,16,32] | 相同                          |
| end2end / reg_max                        | True / 1               | 相同                          |
| 未融合参数                               | 2,504,190              | 2,559,489                     |
| 640 GFLOPs（项目 get_flops/THOP，MAC=2） | 5.771776               | 5.817856                      |

增加 55,299 参数（2.208258958%）和 0.046080 GFLOPs（0.798367781%），满足 <3% / <5%。THOP 未完整计入差值、tanh、访存和部分逐元素操作，GFLOPs 不等于实测延迟；没有捏造 4090 耗时或显存。

`tasks.py` 只在 import 和 `base_modules` 注册新类，不加入 `repeat_modules`。所有非第 9 层的完整模块字符串和所有 from 索引一致，YAML 结构比较只豁免这一层；没有改 loss、C3k2、C2PSA、neck、Detect 或优化器实现。

## 预训练、配方和运行契约

原始 checkpoint：`/root/autodl-tmp/projects/Tunnel_Disease_YOLO26/yolo26n.pt`，SHA256：`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`。本地同名文件哈希一致。

Checkpoint 共 708 tensors；606 项迁移到单类模型。b19 共有 708 个 state 项保持相等；SICR 新增 55 个 state 项，其中有 28 个参数张量（27 个 refine Conv/BN 参数和 theta）。原生 layer 9 的 cv1/cv2 全部加载，剩余 102 个不匹配是与 b19 相同的 COCO 80 类→crack 1 类 Detect 适配，无异常 backbone/neck missing。逐键清单写入 preflight 的 `checks.json` 和正式 `provenance/weights.json`。

正式 trainer 复用 `DetectionTrainer.get_model` 和 `BaseTrainer.build_optimizer`，在真实模型重建处检查共享参数和 606 迁移数，在 `on_pretrain_routine_end` 检查最终参数、AMP、batch 和 MuSGD 分组。theta 自动进入原生一维带衰减权重组，二维以上 refine 卷积进入原生 Muon+SGD 组，BN 沿用原规则；不改学习率、分组算法或优化器。

固定数据：`Tunnel_Crack_AugFirst_Diverse5x_RandomSplit_7_2_1_seed42`。训练/val/test 图片 8414/2404/1202，目标 10243/2985/1477。审计每张图片的大小、内容 SHA256 和标签 SHA256，汇总值与已核实 b19 manifest 一致；不修改划分或标签。

正式条件：200 epoch、patience60、640、batch32、workers8、device0、seed42、deterministic=True、AMP=True、cache=False、MuSGD、lr0=.01、lrf=.003、momentum=.937、weight_decay=.0005、warmup3/.8/.1、cos_lr=True、close_mosaic10、nbs64，以及原始 args 中全部 loss/增强/default 字段。

HSV .024/.84/.535、degrees11、translate.17、scale.735、shear3.5、perspective.00055、flipud0、fliplr.5、mosaic1、mixup.135、cutmix.03、copy_paste0、randaugment、erasing.4 均由真实 args 读取。只允许 model/project/name 和经过哈希证明的等价数据/原始预训练路径表示变化；save_dir 由原生输出位置派生。任何其他有效训练字段变化均失败。

本机环境是 Windows、Python3.11.15、torch2.7.1+cu118、RTX2060 6GB；与 b19 的 Python3.12.3、torch2.8.0+cu128、RTX4090、Ultralytics8.4.98 不同。正式入口要求 b19 环境，不能在本机以较小 batch 冒充正式实验。

`train` 先在独立解释器执行 B32/640 synthetic preflight，再从原始 checkpoint 和原生 seed 重新建立正式 trainer。预检不是数据集小训练，不产生可用来继续训练的 checkpoint。原生 OOM 重试在修改 batch 前重新抛出异常；正式停止只服从原生200 epoch / patience60，不自动续训、降 batch 或换结构。

## 评估、诊断与归档

`finish --stage test` 用本实验 best.pt 跑完整 val/test，同时用 SHA256 匹配正式 b19 的 best.pt 跑同环境 test。统一640/batch32/workers8/device0、FP32、conf=.001、iou=.7、max_det300、rect=True、augment=False。输出 P/R/mAP50/mAP50-95/AP75、逐 IoU AP、原生 PR/F1/P/R 曲线、混淆矩阵、预测 JSON 和实际评估 args。比较表含 Params/GFLOPs、delta、b19 rerun 与历史精确 reference 的差异。

成功最低条件只看 test mAP50-95 > b19，期望 ≥.510，同时查看 Recall/AP75；不会仅凭 val 或 mAP50 宣称成功。

`--stage diagnose` 读取固定字典序前16张 val 图片，640、FP32 eval、不融合，按样本统计每个 Di/Ci 的 mean_abs、RMS、相对 Zi 的 L2；另记 `||Ri||/(||Di||+eps)`。三个 stage 各输出 mean/median/p95/max、raw theta/alpha、固定文件列表/哈希。参数和 BN buffers 在前后全量哈希相等；只保存 JSON/MD 标量摘要，不保存原始 feature tensor，不做训练或 test 选择。

解释预案只供观察：alpha 接近0且指标接近 b19，可能忽略分支；饱和且 AP75 下滑，可能修正过强；mAP50 上升但高 IoU 下滑，可能召回和定位存在取舍；Recall/mAP50-95 提升且 AP75 不降是希望看到的结果。不会自动放大 alpha、改核或启动新版本。

`--stage package` 要求训练完成凭据、相同代码/数据/权重、正式 preflight、test 和诊断文件完整。归档 canonical run，包括 best/last、args/results、图像曲线、所有已有 train/val batch 图、comparison/diagnostics、完整提交源码 `source.tar`、source manifest、Git SHA 和 diff。遍历前剪除 `.attempt.*`，跳过 symlink，hardlink 用普通成员实体化。归档逐成员读取并核验 SHA256/大小、gzip CRC，产生外部 `.sha256`。test/diagnose/package 各自记录阶段状态；package 异常不会重写 test 成失败。

## 本地验证与限制

- 原生等价性：theta=0，alpha=0.1\*tanh(theta)=0；逐 tensor 核对 cv1/cv2 权重、BN weight/bias/running_mean/running_var/num_batches_tracked，两个 Conv 的 bias 均不存在。实际 forward hooks 记录同一输入、Z0/Z1/Z2/Z3、cv2 输入和最终输出，在模块与实际 layer9 输入上均逐项严格相等（atol=rtol=0），没有首个差异点。整网640输出仍保留原 rtol1e-5 / atol1e-6。
- CPU以及RTX2060 CUDA FP32/AMP的B1/640 synthetic detection loss 前向/反向有限；theta三项有梯度；theta更新后3个 refine 分支获得非零有限梯度；全局系数有界。
- 3/5/7方向核、group、线性末层、矩形/1×1边界和 shortcut 关闭均覆盖。
- YAML通过 `YOLO(path)` 构建；n/nc/end2end/reg_max、所有连线、共有参数、原始预训练加载和参数/GFLOPs审计通过。
- 原生MuSGD成员、两步梯度、EMA非零、FP16 checkpoint跨解释器恢复、真实trainer重建与setup callback有专门回归测试。
- 非零alpha的完整checkpoint生命周期复用NDP：同一模型复制到CPU、保存原生FP16 EMA快照，再把该快照转回FP32作为独立参考；新进程经YOLO加载后，state、全部raw输出（含one2many/one2one和top-k结果）、Detect shape/stride/anchors/strides等属性逐项严格相等（atol=rtol=0）。这不是用两个独立随机模型做恢复比较。
- 新进程在CPU上分别对原生b19 control与SICR做融合检查，完整保留one2one的boxes/scores/feats，atol=rtol=1e-4。按既往NDP方法比较top-k之前同anchor的全部解码坐标/概率，原坐标atol=32e-4、rtol=1e-4及概率atol=1e-6、rtol=1e-4均未改变。CPU开发快照SICR raw最大误差4.76837158203125e-5，同anchor坐标最大0.00152587890625像素；CUDA smoke生成的快照分别为4.291534423828125e-5和0.0013427734375像素。未把CPU融合结果宣称为CUDA直接融合前后等价。
- 实际部署另走原生AutoBackend的CPU融合→目标设备FP32推理，640输入、完整one2one结构和所有输出均检查；非零alpha对boxes/scores仍有实际影响。CUDA FP32/AMP backward继续在原GPU路径执行；正式preflight仍B32/640。
- 已在固定16张真实val图上用明确标识为 `lifecycle_only.pt` 的开发快照运行诊断，并验证权重/buffers不变；这不是正式best.pt或性能结果。
- 打包测试覆盖真实hardlink、canonical best/last及MD、逐成员读取和缺失必要文件拒绝；Windows符号链接权限为WinError1314，真实symlink用例明确SKIP，独立故障注入用例验证不会对dangling路径做stat/open。Linux用相同测试会执行真实symlink用例。
- `compileall`、Ruff、shell语法、四个入口 `--help` 均作为交付检查。API reference由项目脚本生成；Windows生成器产生的导航反斜杠已仅在本次生成结果中规范为原目录约定，未修改生成器。
- 尚未执行AutoDL正式B32/640预检、200epoch训练或完整test；因此没有SICR正式涨点结论和正式结果包。

## 正式服务器preflight故障修复

旧提交 `e3766a00f2d7b8f5d90872e3d9809661581a1c62` 的失败断言位于融合阶段：`before`来自完成FP32/AMP synthetic backward后的同一个CUDA模型，`after`来自该模型的deepcopy直接在CUDA上融合。它不是零初始化native-equivalence比较，也不是save/reload比较。两次train-mode forward确实更新BN buffers，但deepcopy继承同一份state；没有证据表明两侧BN状态不同。

可确认的实现问题是预检偏离既往SIR/NDP及本实验正式checkpoint评估生命周期：原生 `load_checkpoint` 和非Jetson的 `PyTorchBackend` 对CPU模型先fuse、再to(device)，旧verify却在训练smoke对象所在CUDA设备直接fuse。现在删除这一职责混用，把融合放回同一checkpoint的CPU恢复审计，并保留GPU训练smoke与实际部署检查。复用 `b19_common.audit_weights/assert_close_tree/computation_conditions`，在common中集中记录推理属性；没有复制SIR中调整backend策略的context，也没有新增CUDA_VISIBLE_DEVICES、TF32或CUBLAS设置。

同时修正了另一个经原生b19 control实测复现的问题：融合前后top-k分数极微小变化可交换行序，按行直接比坐标会把不同anchor的框相减。本地原生control只有2行交换，post-top-k最大假差异231.471649像素，同anchor实际最大差仅0.001220703125像素、raw score最大差1.90735e-6。改用NDP已有的 `_inference` 比较全部anchor，修正的是对应关系，原断言容差没有放宽；save/reload自身仍严格检查包括top-k在内的全输出。

服务器提供的0.0411229是旧CUDA融合路径的观测值。本地RTX2060/PyTorch2.7.1+cu118无法复现RTX4090/PyTorch2.8.0+cu128的同一个数值失配，因此不能将其具体算子根因归为TF32、cuDNN或BN；本次修复证实并纠正了验证生命周期和坐标对应关系。AutoDL重新运行通过之前，不声明正式服务器preflight已PASS。

修复回归还覆盖：故意破坏共享BN应定位到cv1.bn.running_mean；故意破坏pool应定位到Z1；破坏恢复state、Detect stride或one2one raw均必须失败；子进程preflight失败不得构造trainer或建立训练run。模型实现、注册、YAML、run/finish/diagnose/deploy/server脚本及b19全部正式参数均未修改。

Deleted (preflight fix): 删除训练smoke中的CUDA原位融合比较、post-top-k按行坐标比较，以及只验证load/fuse有限性的弱checkpoint测试；复用NDP快照生命周期与同anchor解码、SIR推理属性审计和已有common严格比较。新增行用于用户要求的逐tensor证据、跨进程恢复和失败回归，仅删除旧逻辑不能覆盖这些必要审计。

Deleted: 用SICR替换实验YAML第9层原生SPPF条目；复用基础设施时删去NDP专属模块/身份、训练批次观察和不需要的生命周期函数，删除CCA归档对best-only的选择与transient目录依赖。基点没有实验基础设施，新增固定模块、审计和交付入口无法仅通过删除实现；复用原生SPPF/Conv/Trainer/MuSGD/Validator，并集中共用b19审计以避免重复。

服务器复制命令见 [AutoDL操作说明](b19_sicr_sppf_v1_server.md)。最终提交和远端一致性以实施报告及 `git rev-parse HEAD` / `git ls-remote` 为准。
