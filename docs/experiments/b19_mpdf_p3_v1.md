# b19 + MPDF-P3 v1

独立分支 `exp-yolo26n-b19-mpdf-p3-v1` 从原生 b19 提交
`4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6` 创建。仅候选 YAML 的第 15 层由 Concat 替换为
MPDFP3；不引入 MSI、RSC、RPCA、SIR、DCR、DSD 或其他实验结构。其他工作树、运行结果和任务均保留。

## 实现与来源

- 模块：`ultralytics/nn/modules/mpdf_p3.py`。
- YAML：`ultralytics/cfg/models/26/yolo26n-mpdf-p3-v1.yaml`。实际文件名显式带 `n`，沿用原生尺度识别。
- 解析：`ultralytics/nn/tasks.py` 中独立多输入分支；隐藏宽度 32 不缩放，输出通道只计 U 和 L。
- 启动：`tools/experiments/server_b19_mpdf_p3_v1.sh`。
- 训练：`tools/experiments/run_b19_mpdf_p3.py`。
- 评估、诊断、打包：`tools/experiments/finish_b19_mpdf_p3.py`。
- 验证：`tools/experiments/verify_b19_mpdf_p3.py`、`tests/test_mpdf_p3.py`。
- 来源、原始配置：`tools/experiments/mpdf_sources.json`、`b19_reference.json`、`b19_archived_args.yaml`、
  `b19_launcher_expanded.txt`、`b19_startup_excerpt.txt`。
- 本地审计：`docs/experiments/mpdf_audit/`、`tools/experiments/mpdf_local_validation.json`。

已完整读取 `D:/7.21yolo26改/YOLO26缝合/ultralytics/nn/newsAddmodules/` 下的
`RHDWT_TGRS2025.py` 和 `RLAB_fusion_2025CVPR.py`，并核对它们与 ZIP 内成员一致。
ZIP 实测 SHA256 为 `a0c197f38e1510291a9a762fc7dc0a8c13694742fff9e3132c50a3ee32ddadcf`。
每个参考文件、原生源码、b19 配置/日志、原始权重的实际路径和哈希在来源 JSON 中。

RHDWT 只借鉴保留四个 Haar 子带的组织方式；其本身是下采样模块。RLAB 只借鉴 depth-to-space
重排，原 DSUB 在 PixelShuffle 后还有 stride=2 卷积；其注意力生成 N×N 矩阵，不能按名称视为线性复杂度。
不复制两个模块的完整类，不增加第三方算子依赖。未核实的模块论文题名、作者和 DOI 不作补写。

[FreqFusion](https://arxiv.org/abs/2408.12879) 的 arXiv v1 页面注明 TPAMI 2024 接收，本文只借鉴融合一致性与
边界细节的动机，不复制滤波器或偏移采样。[RFP-YOLO26](https://doi.org/10.3390/app16167991) 的出版社页面为
Applied Sciences 2026, 16(16), 7991，只借鉴细长目标跨尺度传递动机，不复制反馈金字塔或 SPDConv 组合。
不继承它们的性能增益。Haar 后接自由 1×1 可视为 PixelUnshuffle 的线性换基，不能据此主张额外频率选择能力。

Deleted: 候选图第 15 层的原生 Concat 节点；工具适配中删除 MSI 架构、门控 FFN 诊断及其初始化断言。
复用同版本的 b19 全字段配置审计、原生 Trainer/MuSGD/EMA/Validator、attempt 管理及校验打包工具。
这是新增独立实验，原生代码中没有可搬迁的三系数融合实现；新增数值/生命周期检查用于验证新算子和运行边界。
固定 batch 的内存错误退出复用已验证入口：在原生 OOM 重试计数首次改变时重新抛出当前异常，以阻止其自动减 batch；
不修改框架的通用 OOM 行为。

## 固定公式与图

输入顺序为 `[U,L,H]`，图来源 `[14,4,13]`。U 是保留的第 14 层 `nearest(H)`。不插值补偿、裁剪或修补形状。
L 的左上、右上、左下、右下记作 a,b,c,d，所有子带系数均乘 **0.5**：

```text
L0=(a+b+c+d)/2; Lx=(a-b+c-d)/2; Ly=(a+b-c-d)/2; Ld=(a-b-c+d)/2
Q=cat(H,L0,Lx,Ly,Ld)
S=SiLU(Conv1x1_640_to_32(Q)); S=SiLU(DW3x3_32(S))
Bx,By,Bd=Conv1x1_32_to_384(S).split(128)
ra=( Bx+By+Bd)/2; rb=(-Bx+By-Bd)/2
rc=( Bx-By-Bd)/2; rd=(-Bx-By+Bd)/2
R=PixelShuffle(stack(ra,rb,rc,rd,dim=2).reshape(B,4*128,h,w),2)
Y=cat(U+R,L)
```

W1、DW3 无 bias，W2 有 bias；不增加 BN、门控、幅值裁剪、额外损失或阶段开关。两个 SiLU 均不原地执行。
构造使用 CPU `fork_rng`，W1/DW3 保持 Conv2d 常规初始化，W2 权重和 bias 在构造时置零；
checkpoint/EMA/fuse 的加载不再次清零。输入都不被原地修改。

原生 autocast 的 nearest 在本机可能将 U 提升为 FP32，而 L/H 仍为 FP16；模块接受这条原生混合 dtype 路径，
遵循 PyTorch 拼接/相加与 autocast 的类型规则，不强制全程 FP32，也不以 dtype 相同作为额外接线限制。

| 层  | 来源/算子          | 640 输入形状（不含 batch） | 384×672 输入形状 |
| --- | ------------------ | -------------------------- | ---------------- |
| 4   | 原生浅层 L         | 128×80×80                  | 128×48×84        |
| 13  | 原生 H             | 128×40×40                  | 128×24×42        |
| 14  | 原生 nearest(H)    | 128×80×80                  | 128×48×84        |
| 15  | MPDFP3 `[14,4,13]` | 256×80×80                  | 256×48×84        |
| 16  | 原生 C3k2          | 64×80×80                   | 64×48×84         |

共有 24 个节点；Detect 仍取 `[16,19,22]`，双分支、原生 detach、损失和选优规则不变。save 列表包含 4、13、14。
R 每个 2×2 单元的均值在实数域为零；`AvgPool(U+R)=H` 是该融合节点的逐通道约束，有限精度误差单独记录。
不做后处理投影或去均值。这不意味着全网输出、坐标、光照或平移不变，也不是无混叠保证。
下游原生路径可使 P4/P5 间接受到影响。原生 Concat+C3k2 本身能学习特征关系，尚未证明此节点就是 b19 瓶颈。

实测未融合单类参数：原生 **2,504,190**，候选 **2,537,630**，新增 **33,440**。
新增状态仅 `model.15.{reduce.weight,dw.weight,project.weight,project.bias}`。
原始 COCO 权重 SHA256 为 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`；
原生匹配 606/708，候选匹配 606/712，708 项共有参数/buffer/BN 状态逐项完全相同。
其余不匹配项是原生 nc=80→1 的 Detect 适配，并非原生漏载。
卷积增量为 52.8896M MAC/640 图，乘加两次口径为 0.1057792 GFLOPs，未计张量重排、激活和访存，不能当成实测延迟。
fuse 保留新分支的非线性和三层卷积。

## 配置与运行边界

真实 b19 args、原始启动日志与来源记录来自本地 `b19_yolo26n_e200_train_val_test_20260823_224153` 结果包。
原始 shell 脚本不在结果包内；随代码保存的是已有历史启动记录的展开命令，并对全部有效字段再核对。
服务器必须提供真实 b19 args、相同原始 yolo26n.pt、数据 YAML 和完整数据元信息；缺失或不匹配时退出。
不把归档配置当作服务器缺失 args 的默认替代。

训练全部继承真实 b19：200 epochs、patience=60、640、batch=32、MuSGD、seed=42、AMP=True、workers=8、
nbs=64、原生 warmup/累积、全部增强和优化器字段。只允许模型、输出目录及必要身份/路径表达字段有解释的差异。
`resolved.json` 保存全配置及逐字段差异，Trainer 完成 setup 后再次核对，记录共享初始化、原生参数组和 EMA。
运行环境检查继承既有工具：原 b19 为 Python 3.12.3、torch 2.8.0+cu128、Ultralytics 8.4.98、RTX 4090；
不一致明确退出并在 audit 中列出差异，不升级环境或悄悄改变配方。

`train` 自动开启独立 preflight 进程：先顺序加载原生与候选做 batch=32 的初始化对照，释放后进入原生训练循环，
真实数据增强、AMP、batch=32 不变。在最多 32 批内观察至少两次实际 optimizer.step、EMA 更新和 W2 更新，
并等待 W1/DW3 获得有限非零数据梯度。GradScaler 跳过的 step 单独记录，不当成实际更新。
固定前 32 张验证图用于真实 Trainer Validator 与重载 AutoBackend/FP32 Validator；不重复完整 val/test。
预检结束释放整个子进程，正式训练重新设 seed，从原始 COCO 权重重建，不沿用预检权重/BN/优化器/EMA。
GPU 资源不足则保留日志并退出，不减 batch、不更改累积、不影响其他实验。

## 服务器操作

主目录：`/root/autodl-tmp/projects/Tunnel_Disease_YOLO26`。
独立工作树：`/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_MPDF_P3_v1`。
部署时使用交付的完整 SHA，`git worktree add --detach "$WORK" "$SHA"`。若对象已在本机，不需要 fetch。
否则仅拉本实验分支，使用 HTTP/1.1、lowSpeedLimit=1、lowSpeedTime=60、`--progress --no-tags`，不加总超时。
已有目标目录只核对 HEAD 与 clean 状态，不删除、reset 或复用覆盖。

```bash
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_MPDF_P3_v1
bash "$WORK/tools/experiments/server_b19_mpdf_p3_v1.sh" train
```

正常只执行 train，无需先单独 preflight。默认 Python `/root/miniconda3/bin/python`，可通过 `B19_PYTHON` 指定；
从任意目录调用均定位自身工作树，核对实际 ultralytics 导入路径，拒绝未提交源码。
使用独立 tmux 会话 `y26_mpdf_v1`，启动前检查会话不存在，启用 `remain-on-exit` 后再发送 train 命令。

阶段 `preflight/train/test/diagnose/package` 共用本实验自己的 flock，不使用全局训练锁。
已有正式 run 时拒绝再次 train，保留失败/完成结果，不能自动覆盖重训。
只有预检失败且正式 run 尚未创建时，重试 train 会自动新建 attempt。
每阶段目录为 `runs/detect/yolo26n_b19_mpdf_p3_v1_<stage>.attempt.*`，包含 console.log、PID、状态和退出码。
`<name>_<stage>.current_attempt` 指向当前目录；运行中不会保留旧成功 exit_status。
`python.pid` 在 Python 入口进入阶段时产生；自动预检另有自己的 PID/attempt。
正式训练曲线和进度在 `runs/detect/yolo26n_b19_mpdf_p3_v1/results.csv`。

```bash
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_MPDF_P3_v1
tmux attach -t y26_mpdf_v1
# 以下是独立操作，可在另一个 SSH 窗口执行。
STAGE=train
P="$WORK/runs/detect/yolo26n_b19_mpdf_p3_v1_${STAGE}"
A=$(cat "$P.current_attempt")
cat "$A/process_status.json"
tail -n 80 "$A/console.log"
test ! -f "$A/python.pid" || ps -fp "$(cat "$A/python.pid")"
if test -f "$A/exit_status"; then cat "$A/exit_status"; else echo '当前 attempt 未写退出码'; fi
tail -n 5 "$WORK/runs/detect/yolo26n_b19_mpdf_p3_v1/results.csv"
```

`exit_status` 只有进程结束后才存在。异常主机断电/SIGKILL 无法执行 shell trap，此时以实际进程是否存活判断，
不能把缺失退出码当作成功。自动预检状态可将 STAGE 改为 preflight 查看。

```bash
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_MPDF_P3_v1
bash "$WORK/tools/experiments/server_b19_mpdf_p3_v1.sh" preflight
bash "$WORK/tools/experiments/server_b19_mpdf_p3_v1.sh" test
bash "$WORK/tools/experiments/server_b19_mpdf_p3_v1.sh" diagnose
bash "$WORK/tools/experiments/server_b19_mpdf_p3_v1.sh" package
```

test 只用正式训练按 val 选出的 best.pt，在两个独立进程分别做完整 FP32 val、test。
640、batch=32、conf=.001、iou=.7、max_det=300、rect=True、augment=False、quantize=None；
此源码用 quantize 字段控制精度。保存 P/R/AP50/AP75/mAP50–95、PR 等曲线、混淆矩阵、预测和权重/源码哈希。
AP75 从真实 IoU=.75 的 all_ap 列取得。诊断固定 16 张 val，保存 ID/预处理，逐样本 R/U 范数比（C,H,W 归约，
epsilon=1e-12）、系数统计/能量、参数与初始值差异、短跑实际梯度、均值误差、统一色标图和融合后 one-to-one 开关差异。
AMP 单列，不改变训练配方。关闭分支是训练后推理诊断，不是独立训练消融；正常 AP 复用 test。
没有同条件独立 FP32 重评 b19 时，不能把历史指标宣称为严格同条件对照。

package 要求 train/test/diagnose 成功、预检收据及必需产物齐全，并逐项核对内容哈希后发布压缩包。
包内含 best/last、args/results/曲线、评估和诊断、配置差异、来源/结构/初始化审计、完整 Git 源码归档、阶段日志状态和内容清单。
不含原始数据集与凭据。外部 `.tar.gz.json` 是压缩包实际路径、字节数、SHA256 和校验文件数的收据；
package 本身的最终日志/退出码在本工作树阶段 attempt 内。
默认下载目录：`$WORK/artifacts/experiments/`，包名 `yolo26n_b19_mpdf_p3_v1_<12位SHA>.tar.gz`，以实际打印输出为准。

## 本地验证边界

Windows / Python 3.11.15 / torch 2.7.1+cu118 / RTX 2060 的本地检查不等同于服务器正式预检。
运行 `python -m pytest tests/test_mpdf_p3.py -q -o addopts=""`；原始预训练文件可用 `B19_TEST_PRETRAINED` 指定。
覆盖独立矩阵 Haar/显式逐通道相位参考、正常及大幅系数、FP64/FP32/AMP 数值、输入错误、RNG、
两种完整尺寸的前后向、Trainer 单类重建、共享 BN、MuSGD 数据梯度/更新、EMA、恢复 optimizer state、
完整权重独立进程重载/fuse/AutoBackend、真实 CPU Validator、合成 16 图诊断、失败重试状态与缺件打包拒绝。
合成 Validator/诊断仅验证接口和产物，不作为检测精度结论。服务器 batch=32 的真实增强短跑、资源峰值及正式训练尚未执行。
