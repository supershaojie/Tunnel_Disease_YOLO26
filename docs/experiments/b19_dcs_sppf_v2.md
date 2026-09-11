# DCS-SPPF v2：相对残差范数约束

本轮只在 DCS v1 最终残差相加前增加每张图的范数预算，保留 v1 三尺度特征、concat、fuse、BN 和单一 theta。未启动 AutoDL 正式训练；本地检查不能替代服务器原生 B32 预检。没有 v2 正式 Val/Test 指标，也没有自动调参或下一版本。

## 来源与实验身份

| 字段                 | 值                                                                 |
| -------------------- | ------------------------------------------------------------------ |
| Branch               | `codex/exp-yolo26n-b19-dcs-sppf-v2`                                |
| Code base SHA        | `d4a3b83940bdb306d008ede322b1e45bcd08cafb`                         |
| B19 model/recipe SHA | `4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6`                         |
| Class                | `ultralytics.nn.modules.dcs_sppf_v2.DCS_SPPF_V2`                   |
| RUN                  | `yolo26n_b19_dcs_sppf_v2`                                          |
| WORK                 | `/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_DCS_SPPF_v2`      |
| Layer 9              | `DCS_SPPF_V2, [1024, 5, 3, True, 0.05, 0.000001]`                  |
| 原始初始化 SHA256    | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |

已核对两个 Git 对象并从已验证 v1 提交建立独立 worktree；没有从 MGC/LBI/FDV/CCA 工作树叠加，也没有修改主 checkout 或旧实验目录。附件本轮规格取代三路独立投影/三个 0.03 系数草案。按本轮附件不创建 PR。

本地实际找到 v1 归档 `yolo26n_b19_dcs_sppf_v1_d4a3b83940bd.tar.gz`，其 SHA256 为 `d637e0733f885e839e46c58131175dfbbb332bcfcd5d7063dc3f234dfb7c9d3e`，与附件指明的上传包一致。已读取比较、独立 val/test、原 b19 args、预检梯度、诊断、results.csv 和 source.tar；source.tar 中实验工具及 v1 模块与代码基准提交逐字节一致。

模块包 `YOLO26缝合.zip` 的 SHA256 为 `a0c197f38e1510291a9a762fc7dc0a8c13694742fff9e3132c50a3ee32ddadcf`。已读取原 SPPF、HLKConv_TGRS2025、DPCF_INFFUS2025：只核对 SPPF 接口、depthwise/dilation 写法与标量混合不等于残差范数约束这一差别；未复制大核、sigmoid 混合、插值或融合拓扑。ZIP 的旧 GBK 文件名与可读标识均记录在去敏的 [reference_audit.json](evidence/dcs_sppf_v2/reference_audit.json)。文件名中的期刊标签不构成论文验证。

## v1 的直接证据与假设边界

下表是同条件 Test；差异单位为百分点，使用未四舍五入原值计算。DCS v1 为历史结果，未在本任务重新训练或重新评估。

| 指标    |       b19 Test |    DCS v1 Test |  v1−b19 / pp |
| ------- | -------------: | -------------: | -----------: |
| P       | 0.874590231545 | 0.882042929953 | +0.745269841 |
| R       | 0.783792155484 | 0.779661417960 | −0.413073752 |
| AP50    | 0.870056743733 | 0.871518202434 | +0.146145870 |
| AP75    | 0.536747589145 | 0.519516099482 | −1.723148966 |
| AP50:95 | 0.507769938902 | 0.501253408673 | −0.651653023 |

DCS v1 独立 Val 的 AP50:95 为 0.5075305512175974、AP75 为 0.5343302940711554；不拿它与 b19 Test 混比。历史 b19 Test 重评相对参考五项 delta 均为 0。全部逐 IoU 数值保存在 [v1_evidence.json](evidence/dcs_sppf_v2/v1_evidence.json)，未来 v2 test 入口会输出 b19 / v1 历史 / v2 三列及 AP50–95 的逐 IoU 比较。

v1 诊断仅针对文件名排序前 16 张验证图，原生 LetterBox 640×640、auto=False、RGB/255、FP32 fused。theta=−1.6279296875，alpha=−0.09257661551237106；`||alpha*R||₂/||Y0||₂` 均值约 0.126334、范围 0.116221–0.141539。这是范数比，能量比为其平方；不是 Test 全量或随机总体估计。负 alpha 不代表减去真实裂缝，尺度 std 也不能用于贡献排名。

CSV 的 Val AP50:95 最高点在 epoch 200，约 0.50730，191→200 总体仍在上升。记录没有显示训练崩溃，但不能证明已经充分收敛或延长训练无效。200e 保持不变是为了固定 b19 对照预算。

本实验只检验：直接减少实际注入幅度，能否保留粗检测收益并缓解高 IoU 回退。残差幅度尚未被证明是唯一原因；定位、分类排序、匹配和共享层训练轨迹仍可能影响 AP。0.05 是固定工程假设，不是扫描后的最优值，也不保证 AP 或 Recall 提升。Test 已参与开发观察，不能作为从未被观察的盲测描述。

## 结构与数值性质

主路径 cv1（act=False）只执行一次，接三次 MaxPool5、concat、cv2 及原 shortcut。分支保留：

```text
C5/C9/C13 = M5/M9/M13 − AvgPool5/9/13(Z0)
R5/R9/R13 = 原 DW3(dilation=1/2/3) + BN + SiLU
R = 原 fuse(concat(R5,R9,R13))，1×1 + BN，act=False
alpha = 0.10*tanh(theta)，theta 初始为 0
raw = alpha*R
s = sqrt(mean(stopgrad(y0)^2, dims=(C,H,W)))
b = 0.05*s
q = b/sqrt(b² + mean(raw², dims=(C,H,W)) + (1e-6)²)
injected = q*raw
Y = y0 + injected.to(y0.dtype)
```

每张图一个 `[1,1,1]` 标量，整个 q 为 `[B,1,1,1]`；非负且不大于 1，保持 raw 的方向和相对排列。只有预算 reference detach；主 y0、theta、R、raw、u2、q 的必要梯度链保留。预算 detach 不保证共享主干在训练中永远不改变幅值。y0=0 时注入为 0；eps 在分母平方和内，避免 0/0 和零点反向 NaN。

令 t=RMS(raw)/RMS(y0)，对非零 y0 有：

```text
RMS(injected)/RMS(y0)
= 0.05*t/sqrt(0.05²+t²+(eps/RMS(y0))²) ≤ 0.05
```

这约束每张图整体张量的范数比，不约束每个位置的最大偏移或模型预测误差。小残差时 q 接近 1；大残差趋于预算，径向梯度会减弱。这是平滑约束的取舍，不用硬边界切换，也不以删除 bound 来补偿。

控制运算在局部禁用 autocast 后以 FP32 执行；卷积继续原生 AMP，未改变全局 AMP/TF32、优化器或 GradScaler。FP32 bound 容差为 2e-8；FP16/BF16 转换及最终加法的 observed ratio 另外记录，并用实测舍入张量的范数核对三角不等式，不能要求低精度最终差值严格小于十进制 0.05。

原 AvgPool 的 `count_include_pad=True`、全部卷积/BN 名字和初始化、单一 theta 与分支 RNG 隔离保留。没有 double-zero 初始化，没有 theta=0 的分支跳过。meta shape 检查使用 CPU autocast 策略描述形状，不执行真实数据计算。

## 配方与工程复用

原 b19 args SHA256：`b08b915756bf85c91d3356586a651a37156d71867b84b6e8493d75c9568642b9`。逐项读取、比较全部 112 个原字段并验证展开 launcher；没有按附件摘要重建近似配方。保留 200e、patience60、B32、640、workers8、device0、seed42、deterministic、AMP、MuSGD、warmup、EMA、梯度累计、loss 和全部训练增强。`augment:false` 没有被用来关闭训练增强。

数据完整内容清单核验为 train/val/test=8414/2404/1202 图，targets=10243/2985/1477。训练仅从原 `BASE/yolo26n.pt` 加载；不接受 best/last、v1 续训或 resume。运行时仍要求原 Python 3.12.3、Torch 2.8.0+cu128、Ultralytics 8.4.98、RTX 4090，不升级依赖。

v2 的三个 Python 入口显式传入 MODEL、精确 block_type 和实验名。公共函数保留 v1 默认值；没有重设 common.MODEL、另造优化器/训练器体系或复制整个预检。来源收据绑定模块文件、YAML、常数、Git SHA、全部源文件哈希、数据 YAML 和内容 manifest。数据入口同时验证 YAML hash，避免相同图片下更改类别语义后仍借用缓存。

预检继续独立子进程，父进程接收完整匹配收据后才重新构建正式 trainer。recipe 审计的 CPU RNG 前后哈希必须相等；临时 baseline 构图隔离 RNG，候选构图后的 RNG 与原生一致。正式 seed 仍由原生 `BaseTrainer.__init__` 的 `seed+1+RANK` 设置（单进程 RANK=−1 即 42），不会从预检 checkpoint、BN、optimizer、scaler 或 EMA 接着训练。

服务器 native 预检仍为真实 B32/640、原数据/native loss/AMP/MuSGD 和最多 64 批窗口。只读 hook 保留实际 scaler step/skip、clip、momentum、零任务梯度对照及逐元素 FP32 舍入证明。四个 BN gamma 的 sub-ULP 只按原成功判据接受；不能把非零梯度单独当更新，也不能扩大为所有参数的豁免。0/16/32/48/64 阶段日志保留在普通 console.log；失败保存证据并退出。

## 本地验证口径

本地环境为 Windows、Python 3.11.15、Torch 2.7.1+cu118、RTX 2060。完整数值与机器可读 args_diff 在 [local_validation.json](evidence/dcs_sppf_v2_local_validation.json)。测试原始日志、逐 tensor/逐步记录与首次失败证据仅保留本地 artifacts。

| 检查              | 口径                                                                                                        |
| ----------------- | ----------------------------------------------------------------------------------------------------------- |
| 原生/v2 zero-init | CPU FP32、CUDA FP32/AMP；模块和整网 train/eval，含完整 one2one/one2many 与逐层首差定位                      |
| BN 状态           | cv1/cv2 running_mean、running_var、num_batches_tracked 一次更新一致                                         |
| Controller        | 84 组边界/shape/dtype 用例每设备；逐样本 max_abs、ratio、finite、梯度和舍入                                 |
| 梯度              | 固定 reference 的 FP64 raw gradcheck；预算梯度不泄漏，raw/q 未 detach                                       |
| dtype             | FP32、FP16、BF16 控制张量；CPU BF16 autocast 和 CUDA FP16 autocast 卷积                                     |
| 共享权重          | 708/708 精确；606/606 兼容预训练 tensors 继承，102 个 shape 不兼容均属于 COCO80→crack1 Detect               |
| 新参数            | 13 个 parameter tensors，全体 optimizer 唯一注册；相对 v1 参数增量 0                                        |
| 状态生命周期      | 非零 theta 的保存/同进程/新进程加载；常数、精确类和 source binding；EMA；Conv-BN fuse                       |
| 融合行为          | 按原生 Detect.fuse 明确核验 one2many 删除，比较完整 one2one 与最终预测；不取消 zero-init 的 one2many 检查   |
| 原生 loss + MuSGD | 本地 B2/640 合成数据，分别 FP32 与 AMP；不是服务器预检                                                      |
| B32 形状          | meta 张量构图，不是实际 B32 训练通过                                                                        |
| 诊断              | 固定 16 张验证图，与 v1 图像/标签哈希一致；未训练非零 theta 测试副本，不是精度结果                          |
| 归档              | 实际 git archive/source.patch、best/last fixture、逐成员 hash/gzip CRC、断链/attempt 排除及必需证据拒绝测试 |

数学控制边界采用 FP32 与独立 FP64 参考核对；低精度 observed 单独测量。RTX 2060 不具备原生 CUDA BF16 矩阵计算支持，不把控制张量的 BF16 测试当作 BF16 CUDA 训练验证。本地无真实 AutoDL B32 结果，不能保证服务器预检一定通过。

Unfused nc1 参数量实测 b19=2,504,190，v1=v2=2,607,231，增加 103,041；本地 THOP 为 b19 5.771776、v2 5.8555392 GFLOPs。THOP 可能漏掉控制器，因此不能声称 FLOPs/延迟不变：每图 N=C×H×W 的两组 square+sum reductions、N 次 scale 乘法、两次 sqrt 和标量运算均有开销，640 输入下 layer9 N=102,400。实测时间、dtype、融合状态、硬件、重复次数与占卡说明写入验证 JSON；不是服务器性能结论。

## 服务器同步与运行

先把交付 deployment_manifest.json 中实际 `Local commit SHA` 复制到下面的 EXPECTED。脚本只做同步和源码核验，不启动训练。远程拉取失败立即停止，不使用无效 FETCH_HEAD；目标 worktree 已存在时停止，以保留旧工作。

```bash
set -euo pipefail
BASE=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_DCS_SPPF_v2
BRANCH=codex/exp-yolo26n-b19-dcs-sppf-v2
EXPECTED='替换为交付的40位Local commit SHA'
[[ "$EXPECTED" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid SHA'; exit 2; }
cd "$BASE"
exec 9>"$BASE/.yolo26n_b19_dcs_sppf_v2.lock"
flock -n 9 || { echo 'Deployment or stage already active'; exit 3; }
if ! git cat-file -e "$EXPECTED^{commit}" 2>/dev/null; then
    timeout 180s git -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=30 \
        fetch --no-tags origin "refs/heads/$BRANCH"
    FETCHED="$(git rev-parse --verify FETCH_HEAD)"
    [[ "$FETCHED" =~ ^[0-9a-f]{40}$ && "$FETCHED" == "$EXPECTED" ]] || { echo 'SHA mismatch'; exit 4; }
fi
ACTUAL="$(git rev-parse --verify "$EXPECTED^{commit}")"
[[ "$ACTUAL" == "$EXPECTED" ]] || exit 4
[[ ! -e "$WORK" ]] || { echo 'Preserving existing WORK'; exit 3; }
git worktree add --detach "$WORK" "$EXPECTED"
[[ "$(git -C "$WORK" rev-parse HEAD)" == "$EXPECTED" ]] || exit 4
printf 'Verified source: %s\n' "$EXPECTED"
exec 9>&-
```

下列正式命令仅供用户之后运行，本任务不执行：

```bash
cd /root/autodl-tmp/projects/Tunnel_Disease_YOLO26_DCS_SPPF_v2
bash tools/experiments/server_b19_dcs_sppf_v2.sh train
bash tools/experiments/server_b19_dcs_sppf_v2.sh test
bash tools/experiments/server_b19_dcs_sppf_v2.sh diagnose
bash tools/experiments/server_b19_dcs_sppf_v2.sh package
```

train 自动执行 runtime/source/recipe audit → 独立 preflight → 匹配收据 → 全新原权重训练。保留阶段锁、PIPESTATUS、普通日志、失败退出和已有 RUN 保护；不新增 CUDA_VISIBLE_DEVICES/TF32/CUBLAS 设置，不假定 GPU1，不 kill 任何既有任务。不要求用户手工更改已固定配方。

test 固定 FP32、imgsz640、B32、workers8、device0、conf0.001、iou0.7、max_det300、rect=True、augment=False、quantize=None，实测并断言模型/输入 dtype；保留 end2end 推理。同环境重评原 b19 best.pt 并核对历史 hash；v1 列明确来自上述历史包。

diagnose 记录 theta/alpha/rho/eps、每图 RMS、raw/injected/observed norm ratio、q、预算比例、C/R/fuse/raw/injected 的统计及位置 p95/p99/max（每位置通道 RMS / 整图 y0 RMS）。零参考单列 flag；分母 floor 只用于诊断展示。JSON/CSV 和汇总均保存，控制计算与真实输出交叉核验。

package 要求真实完成标记、完整匹配的预检/recipe/weights/optimizer/日志及 val/test/诊断证据；保留 source.tar、source.patch、source_manifest、best/last、曲线、预测和比较。跳过 transient attempts/无效别名并记录；重要必需文件不能跳过后仍称完整。输出在 RUN 外，不覆盖历史包；逐成员 hash 和 gzip 完整性验证仍复用成熟逻辑。

若范数受控但高 IoU 仍下降，报告单一约束不足；若 q 很小而 raw 持续增大，记录顶着预算学习的现象。均不自动扫描 rho、改 loss/尺度/BN、延长 epochs 或做 v3。

## 审计与交付

Deleted: 移除共享脚本中写死 v1 的模型类型、来源文件、测试文件 glob、RUN/入口引用；替换为保留原默认值的显式参数。复用 SPPF、DCS v1、Conv、native DetectionTrainer、MuSGD、原梯度重放和 archive 流程。新增内容是独立 v2 模块、薄入口及本规格要求的可审计验证；保留 v1 和 b19 排除了通过删除旧实验来实现本功能的可能。

独立 reviewer 负责 Core Principles/重复代码/生产可用性/性能完整 diff 审计。已修复 canonical name 循环变量覆盖和数据 YAML 缓存绑定问题，并增加成功/失败收据 fixtures；不以新 guard 掩盖模型误差。最终审计 SHA、真实本地/远程 SHA、push 状态及未运行项记录在交付的本地 deployment_manifest.json，避免把提交自身 SHA 硬编码进该提交。

技术依据仅限实现语义：[PyTorch 2.8 AMP](https://docs.pytorch.org/docs/2.8/amp.html) 支持局部 FP32；[detach](https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.detach.html) 说明梯度分离；[ReZero 原论文](https://arxiv.org/abs/2003.04887) 表明零初始化残差系数是已有思想。它们不证明 DCS v2 的任务有效性，也不支持首创或必然涨点的声明。
