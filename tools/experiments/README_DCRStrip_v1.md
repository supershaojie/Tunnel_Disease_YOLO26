# YOLO26n / b19 / A1 DCR-Strip v1

## 状态与源码边界

本次在已有 worktree 中接续实现和补充核对，没有重复建立实验：

- 分支：`exp-yolo26n-b19-dcrstrip-v1`。
- 源码基座：`4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6`，由 b19 归档的 `environment/git_state.txt` 确认。
- 本地 worktree：`E:/PycharmProjects/Tunnel_Disease_YOLO26/.worktrees/exp-yolo26n-b19-dcrstrip-v1`。
- 正式实验：`yolo26n_b19_a1_dcrstrip_v1`；本地仅产生检查报告，尚未启动完整训练，也没有 A1 的 best.pt/results.csv。
- 当前会话未连接服务器，无法确认服务器上由其他会话启动的任务；本次没有终止任何进程。
- 实现与本说明一同提交；交付时记录实际 commit SHA。复核可执行 `git log -1 --format=fuller`，用 `git show <SHA>` 固定版本。

## b19 的真实来源

实际读取本地归档：

`E:/ditieyolo26跑结果/8.12离线在线/b19 200e/b19_yolo26n_e200_train_val_test_20260823_224153/`

核对了 `train_run/args.yaml`、配套 `results.csv`、训练日志开头的模型/优化器/参数输出、
`config/data.yaml`、`config/experiment_info.txt`、`environment/git_state.txt`、
`framework_versions.txt`、`python_version.txt` 与 `pip_freeze.txt`。
可移植的原始参数与环境摘要保存在 [b19_reference.json](b19_reference.json)，没有用其他实验参数补齐。

| 项目          | b19 实际记录                                                                                                                                                   |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 模型 / 初始化 | YOLO26n / `yolo26n.pt`，COCO 80 类初始模型，训练适配为 crack 单类                                                                                              |
| 预算          | epochs=200、patience=60、imgsz=640、batch=32、workers=8                                                                                                        |
| 优化器        | MuSGD，日志确认 lr=0.01、momentum=0.937；lrf=0.003、weight_decay=0.0005                                                                                        |
| 调度          | cos_lr=True、warmup_epochs=3、nbs=64、close_mosaic=10                                                                                                          |
| 随机性 / 精度 | seed=42、deterministic=True、AMP=True、device=0                                                                                                                |
| 在线增强      | 从真实 args 全量继承，包括 hsv=(0.024,0.84,0.535)、degrees=11、translate=0.17、scale=0.735、shear=3.5、perspective=0.00055、mosaic=1、mixup=0.135、cutmix=0.03 |
| 数据          | `datasets/Tunnel_Crack_AugFirst_Diverse5x_RandomSplit_7_2_1_seed42/data.yaml`，train=8414、val=2404、test=1202                                                 |
| 环境          | Python 3.12.3、Ultralytics 8.4.98、torch 2.8.0+cu128、RTX 4090                                                                                                 |

数据按原来的离线增强版本和划分继续使用，不重新增强或划分。原数据采用文件级随机划分，同一原图的变体可能跨 split；
这是 b19 原协议的属性，不能据此声称是独立无泄漏测试。本轮只按 validation 选择权重，不自动运行 test。

归档缺少原始启动脚本/完整启动命令，也没有记录初始权重当时的 SHA-256。日志能确认原生训练器的有效参数，
但不能排除日志之外的自定义 callback。入口要求补充原始 native `yolo [detect] train ...` 启动记录
（`--baseline-launcher`），才会放行完整预检和正式训练；不要把新生成的命令冒充历史记录。
如果原来使用自定义 Python trainer/callback，需要先补读该源码，本入口会明确拒绝未核实的启动方式。

## 已阅读的原型及实质差异

2026-09-06 实际读取，而非仅按文件名推测：

1. `D:/7.21yolo26改/YOLO26缝合/ultralytics/nn/newsAddmodules/StripConv_AAAI2026.py`
   - SHA-256：`a37cdb8da4c496b45da48240eaab37b235d59791e30bc0be3b7a2fa2626744b9`
2. `D:/7.21yolo26改/YOLO26缝合/ultralytics/cfg/models/add26/yolo26_StripConvC3k2.yaml`
   - SHA-256：`88c500ca42ee6af2de4aa104e21e0e3bb9f0bc06f07f03e1f4664d42b128315b`
3. 为核实接口，补读参考包 `nn/modules/block.py` 的 C3k2 签名及 `nn/tasks.py` 的 C3k2 注册段。

参考文件内标注的论文链接为 [arXiv:2501.03775](https://arxiv.org/abs/2501.03775)，配置注释标注“Ai缝合怪 改进”。
这里记录的是本地适配文件及其标注来源；文件名中的会议/年份不作为论文归属或新颖性的证明。
没有提取或复制原型实现，正式代码没有 timm/mmcv/参考包依赖，也不会读取 D 盘目录。

| 对照项       | 已读原 StripConv / 封装                                           | 既定 DCR-Strip v1                            | 当前实现与核对                                       |
| ------------ | ----------------------------------------------------------------- | -------------------------------------------- | ---------------------------------------------------- |
| 卷积拓扑     | 5×5 DW → 1×19 DW → 19×1 DW → 普通 1×1，串联                       | Reduce 后四个方向并行，k=7                   | 保留；水平、垂直、主/副对角，d=32                    |
| 对角线       | 无                                                                | 每通道七个有效可训练系数，固定方向掩码       | 保留；buffer 掩码组装 7×7，梯度到系数                |
| 输入作用方式 | `x * attn`，无方向 softmax                                        | 有符号方向响应，经位置相关方向融合后输出残差 | 保留；没有替换成输入乘法                             |
| 法向对比     | 无                                                                | `T - 0.5*(T_plus+T_minus)`；两侧差只供门控   | 保留；四个法向索引与 clamped-grid 参考一致           |
| 边界         | Conv 内置默认零 padding，卷积默认有 bias                          | 显式 replicate padding，无卷积 bias          | 保留；常量对比含边界均为零，无 roll 环绕             |
| 门控         | 普通卷积生成乘法响应                                              | 两张统计图、共享 2→1 gate、dim=1 softmax     | 保留；FP32 统计、初始均匀 1/4                        |
| 残差         | Attention/StripBlock 另有残差及 0.01 通道 layer scale             | `F + alpha*Expand(R)`；可训练标量 alpha=0.05 | 保留；Expand 非零初始化，无 BN/激活                  |
| C3k2 内部    | 重建 `self.m`，将 Bottleneck 内卷积换成 StripConv                 | 保留原 `cv1/cv2/m`，仅块输出后加 `self.dcr`  | 保留；旧键名和初始化逐项相同                         |
| 参数接口     | StripConvC3k2 封装漏掉 attn，位置调用可能把 g/shortcut 错传给父类 | 按真实 b19 接口传参                          | 父类参数均使用关键字，显式包含 attn                  |
| YAML 接入    | 具名 YAML 实际所有位置仍为普通 C3k2，没有真正启用 StripConv       | 仅 backbone 第 4 层 P3/8                     | 已真正启用一次，64→128；Detect 输入仍为 16/19/22     |
| 调试/消融    | 原型无这些控制入口                                                | enabled、use_contrast、adaptive_fusion       | 保留；禁用对比时不采样两侧，固定融合时 gate 可无梯度 |

借鉴的是条形深度卷积的方向响应思路；四方向法向对比、位置相关方向融合、保持原基线的块后残差是本候选新增机制。
包导出、parser 的 base/repeat/scale/legacy 处理、父类 attn 参数、CPU RNG 保留属于工程接口与公平初始化适配。
这些差异不证明涨点或学术新颖性，也不表示能恢复输入中不存在的细节。

## 结构、初始化与检查证据

- 只替换 `model.4`：`C3k2(64,128,1,False,0.25)` → `C3k2_DCRStrip`。
  640×640 时输出 `[1,128,80,80]`，640×960 时输出 `[1,128,80,120]`。
- Detect 三个尺度及顺序仍是 P3/8、P4/16、P5/32；end2end、SPPF、C2PSA、其他 C3k2、loss 和标签分配保持原定义。
- 单类基线参数 2,504,190；A1 参数 2,513,345；增加 9,155（约 0.366%）。
  不报告不完整 GFLOPs：对角线虽只有七个有效系数，实际 `F.conv2d` 仍执行密集 7×7，通用 profiler 会漏计。
- 本地候选初始文件为原项目根目录 `yolo26n.pt`，SHA-256：
  `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`。
  已核对其 nc=80、scale=n、YOLO26 架构；未使用 b19 的训练后权重。
  这个当前文件哈希仍须与服务器原始文件核对，不能补写成历史归档中本不存在的证明。

| 加载分组 | 基线/A1 共同键 | 共同张量元素 | 从同一初始权重加载键 |  加载元素 |
| -------- | -------------: | -----------: | -------------------: | --------: |
| Backbone |            240 |    1,373,880 |                  240 | 1,373,880 |
| Neck     |            228 |      903,334 |                  228 |   903,334 |
| Detect   |            240 |      246,082 |                  138 |   155,708 |
| 总计     |            708 |    2,523,296 |                  606 | 2,432,922 |

102 个未直接加载的旧张量均为 nc=80→1 的 Detect 形状适配，与原生 b19 初始化一致；每个键及原因写入检查报告。
全部 708 个共同初始化张量（包括未加载的随机初始化头部）在 baseline/A1 中逐项相同；目标原块全部旧权重与来源一致。
新增 10 个参数张量仅位于 `model.4.dcr`。构造新分支时局部保存/恢复 CPU RNG；原生 trainer 先在 CPU 构造，随后移至设备，
这段构造不消耗 CUDA RNG。`AuditedTrainer.get_model` 审计真实重建路径，训练前 callback 再检查最终模型和 optimizer。

本地已执行 17 项模块/入口检查，全部通过；随后补充的目录冲突与原生最终 setup/OOM 检查 2 项，以及逻辑 GPU 与物理 UUID 映射检查 1 项也已通过，共 20 项：

- 四方向核有效位置、法向采样、边界、常量零对比、奇数/矩形/极小块输入、c1≠c2。
- `forward` / `forward_split` 一致；disabled 时整网 640×640、640×960 原始输出旁路等价。
- CPU FP32 与 CUDA AMP 的 YOLO26 检测损失/backward、所有完整模式新增参数非零有限梯度、MuSGD 更新和 EMA。
- 两张真实训练图片及标签，沿用 b19 在线增强，进行独立 batch=2、imgsz=640 的 CUDA AMP 检测 loss/backward/optimizer step。
  此项仅为本机诊断，未冒充 b19 的完整 batch=32 预检。
- 在新 Python 进程中正常 `YOLO(checkpoint)` 加载、预测、fuse 后数值一致；不需要导入实验脚本或参考包。
- 原生 `_setup_train` → 最终 optimizer/callback → 第一批入口，CPU 两图诊断中注入 OOM，确认 batch 不变、pipeline 只构建一次、原异常传播，未跑完任何 epoch。
- 同名正式目录在 trainer 构造边界原子占用；冲突立即失败，不覆盖原结果，不产生 name2。

检查文件为 [tests/test_dcr_strip.py](../../tests/test_dcr_strip.py)。本地报告位于 worktree 的
`runs/detect/yolo26n_b19_a1_dcrstrip_v1_preflight/`；一次性测试检查模型位于 `runs/dcr-tests-*`，均不纳入 Git。

尚未完成：b19 原服务器环境中的 batch=32、workers=8、AMP 全配方预检；原始启动命令核对；服务器原始权重身份核对；正式 A1 训练。
本机 Python 3.11.15 / torch 2.7.1+cu118 / RTX 2060 6 GB 与 b19 不同，入口报告这些限制并拒绝自动改变配方。

## 补充核对带来的最小修正

模块数学定义可以保留，补读原型不构成重写理由。此次完善的是首次交付前的工程边界：

1. 原训练循环首轮 OOM 默认减 batch。将已有错误处理搬迁为 `_handle_train_memory_error`；普通训练器保留原行为，
   A1 覆写后在任何 batch 变更前传播原异常。没有伪造重试计数、额外开关或复制训练循环。
2. 输出目录由 A1 trainer 原子创建，再复用已有 `save_dir` 参数，消除预检期间另一启动创建目录导致 name2 的竞态。
3. 自动 preflight 子进程使用规范化 CLI 参数；支持 `--stage=train`。父进程在子进程预检前不占用 GPU context。
4. 初始权重重定位须与仍存在的原始文件逐字节匹配；旧位置缺失时要求原文件历史 SHA-256，不能自行换权重。
5. 正式日志同时保存 LOGGER、进度/print、异常；fresh-process 检查输出也会回收。无静默 OOM 降级。
6. GPU 占用查询使用所选 PyTorch 设备的 UUID，正确处理 CUDA_VISIBLE_DEVICES 的逻辑/物理编号映射。
7. 删除 Windows 文档生成器造成的无关导航路径改写，仅保留新模块一行导航；删除新模型 YAML 中不再适用的原模型性能注释。

未产生修正前 A1 完整实验结果，因此当前没有需要重跑的 A1。历史 b19 的代码和结果没有被修改，仍可作为约定协议下的 baseline。
未来若改动计算、训练或初始化，必须另建输出并从同一初始预训练重新进行公平消融，不能混用旧结果或从旧 best.pt 续训冒充重跑。

## 服务器操作（当前未执行）

先使用运行 b19 的原环境，不猜 Conda 名称、不升级共享环境。首次建立独立 worktree；已有目录/会话先检查，不重复执行创建。

```bash
set -euo pipefail
BASE=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_DCRStrip_v1
cd "$BASE"
git status --short
git fetch origin
git worktree add --detach "$WORK" origin/exp-yolo26n-b19-dcrstrip-v1
cd "$WORK"
git log -1 --oneline
tmux new -s y26_dcr_a1
```

在 tmux 中执行。原始 b19 launcher 不在现有归档中：先在原目录、原会话记录或脚本中找回，设置其真实路径。
下面 `read` 接受历史命令/脚本文件路径，不能填入新拼出的假历史记录。

```bash
set -euo pipefail
BASE=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_DCRStrip_v1
cd "$WORK"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_AUTOINSTALL=false
python -c "import sys,torch,ultralytics; print(sys.executable); print(ultralytics.__file__); print(torch.__version__); print(torch.cuda.is_available())"
nvidia-smi
sha256sum "$BASE/yolo26n.pt"
read -r -p '原始 b19 启动命令/脚本文件绝对路径: ' B19_LAUNCHER
test -f "$B19_LAUNCHER"
B19_ARGS="$BASE/runs/detect/b19_y26n_diverse5x_e200_i640_b32_musgd_b8b9hybrid_s42/args.yaml"
COMMON=(--baseline-root "$BASE" --baseline-args "$B19_ARGS" --baseline-launcher "$B19_LAUNCHER")
python -u tools/experiments/run_b19_dcrstrip.py "${COMMON[@]}" --stage preflight
python -u tools/experiments/run_b19_dcrstrip.py "${COMMON[@]}" --stage train
```

也可省略 `--baseline-args` 自动查找；不同候选并存时必须显式指定。需要定位同一文件副本时两阶段都带 `--pretrained`；
旧原始位置已不存在时还需 `--pretrained-sha256`，其值必须来自原始 b19 初始权重记录。
入口配置差异只能是模型、实验名称/目录、同一权重/数据的路径映射及已分类运行元数据；任何其他差异均停止。
正式 train 将检查匹配当前代码、配置、环境、权重和数据元数据指纹的 preflight，通过后重新初始化。

预期正式输出目录：`$WORK/runs/detect/yolo26n_b19_a1_dcrstrip_v1/`。
日志：该目录 `train.log`；审计/配置/源码信息：`provenance/`。预检单独写入同级 `_preflight/` 目录。
程序异常返回非零，不缩短 epochs、不减 batch、不切换 AMP/优化器，不抢占已有 GPU 作业。
8 GiB 可用显存是保守的预检门槛，不是对峰值显存的测量或可训练保证；完整 batch 预检仍可能 OOM 并退出。

用 `Ctrl+B` 后按 `D` 分离；回看使用 `tmux attach -t y26_dcr_a1`。本次补充没有自动启动第二个训练任务。
完成后归档 args.yaml、results.csv、曲线、模型 YAML、provenance、Git 提交与日志；权重按原项目打包惯例归档，不进 Git。
只基于 validation 评估本轮，再决定后续候选；锁定方案后才统一测试。

## 后续固定约定

每次改进前，先到 `D:/7.21yolo26改/YOLO26缝合` 查找并阅读对应原型代码和配置，记录实际来源、读取版本和实质差异。
找不到时明确报告；不整体覆盖参考包的 tasks.py/loss.py/tal.py，不引入整包依赖。正式模型不能依赖该本地目录。
此约定已合并到仓库 AGENTS.md 的现有检索规则，没有另加一组重复规则。
