# FDV-C2PSA v1 实施与审计报告

## 身份与边界

- Branch：`codex/exp-yolo26n-b19-fdv-c2psa-v1`。
- Base SHA：`4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6`。
- 独立 worktree：`.worktrees/exp-yolo26n-b19-fdv-c2psa-v1`。
- RUN：`yolo26n_b19_fdv_c2psa_v1`。
- 正式训练：**NOT STARTED**。未连接 AutoDL，未创建 PR。
- 交付 commit 与远端 push 校验值见交付报告；本文件随实现一起提交，避免自引用 SHA。

已完整读取用户指定的 FDV 指令。原始 b19 归档的 `environment/git_state.txt`、
`b19_reference.json` 与近期 SICR 独立分支的 history 均指向上述基线。
分支直接从该 SHA 建立，没有从 RPCA/MSI/RSC/CCA/SPPF 创新分支派生。
只选择性复用这些实验中已经审计的实验基础设施，未导入任何其它创新模块。

[来源、实际路径与 SHA256](evidence/fdv_source_audit.json)记录指令、模块包、b19 原始文件及被读取的代码。
[历史指标原始证据](evidence/fdv_history.json)记录归档成员路径、成员 SHA256 和指标。

## 实现前审计

原生 `Attention` 的 QKV reshape 为 `[B, heads, 2*key_dim+head_dim, H*W]`，
按 key/key/value 拆分。实际 score 为 `(q*scale).transpose(-2,-1) @ k`，
原生 softmax 后通过 `v @ A.transpose(-2,-1)` 恢复空间输出，加原生 depthwise `PE(V)`，再执行 proj。
PSABlock 的 attention residual、FFN residual，以及 C2PSA 的 cv1/split/m/concat/cv2 均已逐项审计。

下表使用各实验 200 轮后保存的 FP32 val/test 评估；不同历史环境的绝对差异不是因果证明。
FDV 后处理脚本会在同一运行环境中重新评估已固定的 b19 best.pt。

| 实验   | 机制变更                                                   | Val Recall | Val AP75 | Test Precision | Test Recall | Test AP75 | Test mAP50-95 |
| ------ | ---------------------------------------------------------- | ---------- | -------- | -------------- | ----------- | --------- | ------------- |
| b19    | Native                                                     | 0.788610   | 0.524806 | 0.874590       | 0.783792    | 0.536748  | 0.507770      |
| RPCA   | 区域概率混合；bypass 空间 gate；FFN 原生                   | 0.789693   | 0.507684 | 0.855139       | 0.783362    | 0.515664  | 0.496752      |
| MSI    | Attention 原生；FFN expansion 的 DW3/DW5/GELU 局部交互残差 | 0.777111   | 0.521469 | 0.878509       | 0.773189    | 0.504011  | 0.497786      |
| RSC v1 | 原生概率与互惠几何对称概率混合；per-head beta              | 0.778894   | 0.534610 | 0.864384       | 0.798240    | 0.519511  | 0.501479      |
| RSC v2 | 互惠不平衡的有界 logits 修正后重新 softmax                 | 0.778671   | 0.520806 | 0.877982       | 0.789215    | 0.509526  | 0.503009      |

RPCA/MSI/RSC 均未同时保持 b19 的 Test AP75 与 mAP50-95。FDV 将实验假设移到 value 高频残差，
不重复 probability/logit 校准，也不增加 FFN/local interaction 模块。

实际模块包是 `D:/7.21yolo26改/YOLO26缝合.zip`，SHA256
`a0c197f38e1510291a9a762fc7dc0a8c13694742fff9e3132c50a3ee32ddadcf`。

- RHDWT_TGRS2025：读取 Haar DWT 高低频分量与卷积残差，只参考分解思路；不使用 DWTForward、降采样或额外依赖。
- MALA_2025ICCV：读取 QKVO、RoPE/ELU 线性 attention、LEPE 与 FFN；仅参考 value 局部信息保留，不采用其 attention、gate 或 5×5 卷积。
- HMHA_2025CVPR：读取 channel regroup、temperature、cache modulation、QKV depthwise 以及 C2PSA wrapper；不采用其 head 分配或概率/调制机制。
- MultipoleAttention_2025ICCV：读取 unfold/fold 局部 attention、分层 down/up 和 C2PSA 注册；不采用多尺度 attention 或新 FFN。
- 读取包内 `tasks.py`、`newsAddmodules/__init__.py`、原生 YAML 和相关 C2PSA YAML。
  Multipole 的两个 YAML 中 layer 21 引用 `[-1,9]`，与 b19 的 `[-1,10]` 不同，因此本次没有复制这些 YAML。

## 最小模型改动

`FDVAttention` 接管已经构造好的原生 qkv/proj/pe，保留名称、shape 和 RNG。
`FDV_C2PSA` 先调用原生 C2PSA 构造，再替换内部 Attention。
无需新增 FDVPSABlock：直接保留原生 PSABlock 对象，避免复制 FFN 和 residual。

```text
Q, K, score, A, O_att, O_pe = native
V_low  = AvgPool2d(3, stride=1, padding=1)(V_spatial)
V_high = V_spatial - V_low
gamma_c = 0.10 * tanh(theta_c), theta_c.shape=[128], init=0
O = O_att + O_pe + gamma_c[None,:,None,None] * V_high
output = native_proj(O)
```

AvgPool 使用原生默认 `count_include_pad=True`；尺寸不变，无学习参数。
gamma 在 FP32 中计算，乘法前回到 value dtype，保留 native AMP 主路径的 dtype。
没有 zero-init 跳过分支、spatial gate、SE/CBAM/EMA/CPCA、FFT/DWT、router、top-k、temperature 或额外卷积/FFN。

YAML 从本基线 `yolo26.yaml` 生成，明确 nc=1/scale=n，与 b19 的实际训练模型一致。
唯一模型结构变更为 **layer 10 内部 Attention**。layer 9 SPPF 原生，layer 21 引用 layer 10，
Detect 输入 `[16,19,22]`、stride `[8,16,32]`、reg_max=1、end2end、one2one/one2many 均保持。
注册仅涉及 modules 导出和 parser 的 base/repeat 集合。

## 预训练与全部配置

原始初始化文件 SHA256 必须是
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`。
正式入口只接受该文件，并验证 COCO 80 类原生 nano 图。
`best.pt`、`last.pt` 或任何其它创新实验权重不能作为初始化。
b19 best.pt 仅用于训练后的同条件比较，另有固定历史 SHA256 校验。

112 项 b19 args、历史 launcher 展开值、数据 split/class 和完整 image/label 内容 manifest 已核验。
实际数据集 `Tunnel_Crack_AugFirst_Diverse5x_RandomSplit_7_2_1_seed42`：
train 8414 图/10243 实例，val 2404/2985，test 1202/1477。
允许的配置差异仅为经验证的 model/pretrained/data 路径及 project/name/save_dir 身份。

epochs=200、imgsz=640、batch=32、workers=8、device=0、seed=42、deterministic=True、amp=True、cache=False、
MuSGD、lr0=0.01、lrf=0.003、momentum=0.937、weight_decay=0.0005 均保持。
patience=60、warmup=3.0/0.8/0.1、cos_lr=True、close_mosaic=10、hsv=0.024/0.84/0.535、
degrees=11、translate=0.17、scale=0.735、shear=3.5、perspective=0.00055、mixup=0.135、cutmix=0.03，
以及其余全部原始字段均继承，没有新增可调实验参数。

## 验证结果

本地 Python 3.11.15、PyTorch 2.7.1+cu118、RTX 2060，与 b19 服务器环境不同。

| 检查                        | 实测结果                                                                                         |
| --------------------------- | ------------------------------------------------------------------------------------------------ |
| import/registry/model build | 通过；24 层，只有 layer 10 内部 Attention 改变                                                   |
| Attention zero-init         | CPU/CUDA，多种矩形/奇数/单点尺寸，max_abs=0、mean_abs=0、allclose=True                           |
| C2PSA zero-init             | CPU/CUDA，同输入同共享权重，max_abs=0、mean_abs=0、allclose=True                                 |
| shared state                | 708/708 逐 tensor 精确一致，100% shared coverage                                                 |
| 原始 checkpoint 加载        | 606/708 原生项（85.5932%）；新增仅 theta_c 一个 tensor，128 参数                                 |
| 未加载原始项                | 102 项均是原生 COCO 80→crack 1 类 Detect shape 适配；不存在其它 missing/unexpected               |
| layer 10 共享加载           | cv1/cv2/qkv/proj/pe/FFN 全部继承                                                                 |
| FP32 update                 | CPU/CUDA 的 B1、640 detection loss 前后向通过；theta、qkv/proj/pe、FFN 均实际更新                |
| CUDA AMP update             | 原生 GradScaler 初始 3 次溢出跳步，65536→8192；第 4 次有限梯度并完成全部受审参数更新             |
| 保存重载                    | 非零 theta，同一 FP16 snapshot、FP32 新进程 reload；709 项状态、Detect cache 和 raw 输出严格一致 |
| 原生/FDV fusion             | 沿用原生 CPU fuse，保留 one2one 和 decoded anchor 断言，均通过                                   |
| diagnose                    | 合成非零模型 16 图与真实 val 前 16 图初始化模型均通过；JSON/CSV，state 未改变                    |
| package dry-run             | 完整必需文件门禁、best/last、hardlink 物化、attempt 排除、缺失 console、逐文件 SHA256/CRC 通过   |
| 符号链接                    | Windows 账户不能创建实际 symlink，该用例明确 skip；Linux 运行同一用例复验                        |
| Ruff/syntax/bash/diff       | Ruff 0.12.12、Python 编译、bash -n、diff --check 均通过                                          |

AMP 溢出的 scaled gradients 不被算作更新；被跳过步骤的参数必须逐 tensor 未变。
所有最终参数、buffer、loss、有效更新梯度均有限。未修改 scaler、softmax、TF32、CUDA 或 CUBLAS 策略。
本地 B1 是明确标记的隔离验证，不更改正式 batch=32。

参数量：native **2,504,190**；FDV **2,504,318**；增加 **128（约 0.00511%）**。
640 下 THOP 口径 GFLOPs：native **5.771776**；FDV **5.7718784**。
THOP 不完整统计 functional matmul/逐点算子，此数值不能视为精确硬件 FLOPs。
FDV 旁路另按固定池化及逐点操作估算约 **614,656 scalar ops/图**（tanh 按一个标量操作记），不含任何新增注意力矩阵。

完整逐 tensor、梯度、重载和数据证据见 `evidence/fdv_local_validation.json`。
最终回归：**13 passed、1 skipped**（Windows 创建符号链接权限），耗时 21.46 秒。最终 commit/push 回执见交付报告。

## 服务器入口与生命周期

在指定分支的干净 worktree、b19 原环境和原数据/权重就绪后执行：

```bash
bash tools/experiments/server_b19_fdv_c2psa_v1.sh train
bash tools/experiments/server_b19_fdv_c2psa_v1.sh test
bash tools/experiments/server_b19_fdv_c2psa_v1.sh diagnose
bash tools/experiments/server_b19_fdv_c2psa_v1.sh package
```

可仅运行 `preflight`。默认 Python 为 `/root/miniconda3/bin/python`，可用 `B19_PYTHON` 指定原环境解释器；
runtime audit 仍严格比对 b19 的 Python 3.12.3 / PyTorch 2.8.0+cu128 / Ultralytics 8.4.98 / RTX 4090。
BASE 固定 `/root/autodl-tmp/projects/Tunnel_Disease_YOLO26`。

`train` 顺序为：runtime/source/data/args/weight audit → 独立新进程 preflight → 校验 PASS/commit/config/source →
全新 trainer 从原 seed 和原始 yolo26n.pt 开始正式 200e 配方。
独立 preflight 在真实 dataset 上使用 native AMP/warmup/clip/MuSGD/EMA，B32、640，
最多 32 个 batch，至少两个有效 step，并要求 theta/qkv/proj/pe 的有限非零梯度及实际更新。
失败停止，OOM 不允许自动降低 batch，正式 RUN 已存在则保护现有结果。
preflight 与正式运行日志、源代码哈希和原始配置会归档。

`test` 固定训练 val 选出的 best.pt，对候选和历史 b19 best.pt 分别执行同条件 FP32 val/test，
输出 P/R/AP50/AP75/mAP50-95、曲线、混淆矩阵、predictions、比较与 provenance。
`diagnose` 对固定排序前 16 张 val 图通过真实 forward hooks 捕获 QKV/PE/pool/proj 输入，
验证统计重构等于真实 proj 输入，并输出 theta/gamma 分布、百分位、正负通道、近零比例（阈值 1e-4）、
V_low/V_high 统计、O_att/O_pe/O_high norm 与 `||O_high||/||O_att+O_pe||`。
零分母记 null；不通过额外模块 forward 改变 BN 状态。

`package` 复用 CCA v2 修复后、经 SICR 复用的归档流程；排除 attempt 目录及 symlink，
hardlink 物化为普通成员，缺少 console.log 不影响正式结果包。
要求 best/last、args/results、curves、val/test、comparison、diagnostics、completion 和 provenance；
包括完整 source.tar、source.patch、源文件 SHA256、逐成员 SHA256/size、gzip CRC 和归档 sha256 sidecar。

## 差异审计与限制

没有更改原生 Attention/PSABlock/C2PSA 文件、trainer/loss/optimizer 实现或全局训练默认值。
API reference 生成器已运行；移除本次生成的 Windows 路径/全导航重排，只保留新增模块的引用页与一行导航。

Deleted: 原始基线没有删除项。独立新模块和自足的实验入口在该基线上不存在，无法靠删除基线代码实现；
复用原生 C2PSA/PSABlock 和成熟 b19 基础设施，省去 FDVPSABlock 及重复 FFN/wrapper，未搬入其它创新实现。

最终审计关注：新增条件均位于实验输入验证、原生 OOM 请求边界或原生 GradScaler 步骤观察；
FDV forward 没有通过条件隐藏数值错误或绕过初始化检查。

尚未执行服务器真实 B32 预检、Linux flock 全流程或正式 200e，因此没有 FDV 精度收益结论。
gamma 有界不代表特征扰动比例必然小于 10%；AvgPool 零 padding 可能产生边界高频响应。
这是按用户固定规格实现的实验假设，是否提升 Recall/AP75/Test 泛化需要后续固定协议的训练和评估。
