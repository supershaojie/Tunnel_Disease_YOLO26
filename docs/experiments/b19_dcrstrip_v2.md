# b19 / DCR-Strip v2：唯一候选执行记录

本轮只实现、检查和交付一个 v2 候选。不安排 D1/D2、多种子、baseline 重训或短训筛选。
起点是已训练 v1 提交 `be5e2bfc013e902e74451b066da1ec24ff555b2e`；baseline 源码为
`4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6`。实验分支为 `exp-yolo26n-b19-dcrstrip-v2`，不合并到 baseline/main。

## 机制与边界

只把 backbone 第 4 层替换为 `C3k2_DCRStripV2`，nano 为 64→128 通道，640 输入输出 80×80 P3 特征。
继承当前 C3k2 的 `n,c3k,e,attn,g,shortcut` 顺序，保留 `cv1/cv2/m` 及 chunk/split 两条原始 forward。
第 4 层只增加 v2 分支，没有先跑 v1 再叠 v2；P3/P4/P5 和 Detect 索引 `[16,19,22]` 不变。

两项变化：

1. 新增方向分支保留内容 `U=mean(T_H,T_V,T_D,T_AD)`，再加入有符号补偿 `beta*V`。
   v1 的原 C3k2 主路径本来就完整保留，不能说 v1 删除了主干内容。
2. 以 FP32 归一化统计 `q=u/(u+v+1e-6)` 替换无约束 1×1 gate，方向权重为温度 1 的 softmax。
   在 u 固定时，法向两侧差异 v 增大只会降低 q；q 是人为定义的相对对比指标，不是标定后的裂缝概率。

方向顺序 H、V、D、AD，法向分别 `(1,0),(0,1),(1,-1),(1,1)`。沿用 v1 的 7 点对角核、k=7、r=1、
replicate padding 和 `d=max(8,ceil(C/32)*8)`；复用其 `DiagonalStrip`、`contrast` 和 `directional_responses`。
没有引入卷积 CUDA 扩展、SE/CBAM、P2、额外检测头、损失或增强变化。

`C=T-(T_plus+T_minus)/2` 保持符号；仅门控使用绝对值。通道均值和 replicate 3×3 池化、q 与 softmax
均在禁用 autocast 的 FP32 内计算，不 detach；权重在内容计算前转换为 C 的 dtype。全零统计得到 q=0、w=0.25。
`M=U+sigmoid(beta_raw)*sum(w_i*C_i)`，Expand 是无 bias 的 1×1 Conv，无新增 BN/激活。
`beta_raw=log(0.25/0.75)`，可学习 alpha=0.05 不限幅，Reduce/Expand/方向核非零初始化。
只保留 `enabled=False` 调试旁路，不生成额外训练开关组合。

相对 b19：2,504,190→2,513,344 参数，增加 **9,154（约 0.36555%）**，比 v1 少 1 个参数。
v2 没有旧 gate 参数，多一个 beta_raw 标量。通用 profiler 漏算函数式对角 conv2d（实际使用稠密 7×7 工作量），
因此启动日志里的 GFLOPs 不是精确总开销，本报告不据此给出精确算力或速度收益。
内容表达和方向选择可能改善迁移是本轮待验证假设；不宣称已证明 v1 退步的原因，也不承诺 v2 涨点。

## 已读取的原型与来源

2026-09-06 实际读取：

- `D:/7.21yolo26改/YOLO26缝合/ultralytics/nn/newsAddmodules/StripConv_AAAI2026.py`
- `D:/7.21yolo26改/YOLO26缝合/ultralytics/cfg/models/add26/yolo26_StripConvC3k2.yaml`
- 同包 `ultralytics/nn/tasks.py` 的 `C3k2_Class` 注册；当前仓库 v1 PY/YAML/runner/tests。

- `StripConv_AAAI2026.py` SHA-256：`a37cdb8da4c496b45da48240eaab37b235d59791e30bc0be3b7a2fa2626744b9`
- `yolo26_StripConvC3k2.yaml` SHA-256：`88c500ca42ee6af2de4aa104e21e0e3bb9f0bc06f07f03e1f4664d42b128315b`

原型 `StripConv` 是 5×5 depthwise→1×19→19×1→1×1 的串联条形聚合，卷积使用普通零 padding，最后乘输入。
维护者用 `StripConvC3k2` 替换内部 bottleneck；其构造接口不含当前仓库的 attn 参数，不能按位置照搬。
**所读 YAML 的实际节点全部仍为原 C3k2，没有出现 StripConvC3k2**；文件名并不代表已经接入。

[Strip R-CNN 论文（AAAI 2026）](https://ojs.aaai.org/index.php/AAAI/article/view/38217)
与[作者仓库](https://github.com/HVision-NKU/Strip-R-CNN)提供细长目标条形聚合动机。
作者的 StripNet 使用串联正交条卷积，基于 MMRotate；本地包中的 YOLO C3k2 适配来自包维护者（YAML 标记 Ai缝合怪），
不能写成作者的原生 YOLO 实现。v1/v2 的四方向法向对比和本次 U+beta\*V 都不是原论文方法的原样复现。

作者仓库当前 [LICENSE](https://github.com/HVision-NKU/Strip-R-CNN/blob/main/LICENSE) 为 CC BY-NC 4.0；
本仓库为 AGPL-3.0，原型 PY 无独立许可证头，YAML 带 Ultralytics 许可标记，不能据此推断整包重许可。
本次独立实现任务给定公式，复用仓库 v1 几何代码，没有复制作者/原型完整代码，也没有引入 timm/mmcv 依赖。

后续约定：修改前先读该包对应 PY/YAML，再针对性修改并记录差异；ZIP 备用路径是
`D:/7.21yolo26改/YOLO26缝合.zip`，WSL 对应 `/mnt/d/7.21yolo26改/YOLO26缝合`，须检查实际挂载。
本轮真实读到了 D 盘文件；远端不存在此路径时应如实记录，不能捏造访问经历。

## 公平初始化与训练入口

原始 `yolo26n.pt` SHA-256：
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`。
正式 v2 只从该 COCO 初始化文件开始，不从训练后的 b19/v1 best.pt 开始。

共同的 708 个参数/BN buffers 全部逐项相等；其中 606 个源张量加载一致，102 个 Detect 形状差异遵循原生
nc=80→1 适配，不额外重置整个头。新增参数仅在 `model.4.dcr`，包括 alpha/beta_raw；局部 CPU fork_rng
保留后续网络初始化序列。原生 Trainer.get_model、最终 setup 后模型和最终 MuSGD 均有审计。

共享 v1 runner 只增加显式模型、类、入口和回调绑定；默认仍是 v1。v2 不复制 800 行 runner。
删除旧重载子进程的 v1 类/属性硬编码，改为传入实际类路径；baseline 搜索排除 DCR 目录。
源码指纹包含所有包 PY/YAML、共享 runner、v2 入口和 b19_reference.json；v1 凭据不能用于 v2。
通过的同一指纹可复用；正式 train 缺少有效 v2 凭据时先在新进程 preflight，之后重新建种子/模型/优化器/EMA/loader。
OOM 传播原异常，不降低 batch 或切 CPU；输出原子占用，不覆盖旧结果或悄悄改名。

`b19_reference.json` 从已训练 v1 原样保留。新增 `b19_launcher_expanded.txt` 是用户本轮给出的**历史 b19
启动记录**，文件中标明来源；服务器记录已存在时核对它，不替换它，不重新训练 b19。
原始 args.yaml 优先，200 轮上限、patience=60、seed=42、640、batch=32、workers=8、MuSGD 和全部增强/损失保持不变。
允许差异仅模型、同一预训练权重的等价表示、输出路径，以及跨机器同一数据路径映射。

## 本地检查与真实阻塞

运行环境：Windows，Python 3.11.15，torch 2.7.1+cu118，Ultralytics 8.4.98，RTX 2060 6 GB。
正式 b19 环境是 Python 3.12.3、torch 2.8.0+cu128、RTX 4090。

已执行定向 v1/v2 测试共 35 项（v1 23 项，v2 12 项；其中后处理/打包使用明确标记的测试夹具）：形状/边界、forward_split、RNG、旁路全网 640×640/640×960、归一化门控、
CUDA AMP 内容 dtype、全部新参数梯度、原生检测 loss/MuSGD、EMA、fresh-process reload/fuse/predict。
另核对真实 b19 args、launcher、同一权重哈希和 Trainer 重建。
原生最终 `_setup_train`、final_model_audit 和开始标量回调实际通过；两张真实增强图片在 640 输入 CPU 下完成
检测 loss/反传/MuSGD step（一次性 batch=2，不是正式预检）。第一次 64 输入增强后目标消失，检查如实失败，
改用 640 输入后通过；未因此修改正式配方。真实历史 v1 best.pt 重载/fuse 前后推理一致也已通过。
本地报告在 `runs/detect/yolo26n_b19_a1_dcrstrip_v2_preflight/`，包括 `structural.json`、
`local_weight_audit.json`、`resolved.json`、`missing.json` 和日志；均不进 Git。

本地正式 preflight 因 Python/PyTorch 不匹配、可用显存小于继承的 8 GiB 门槛及占用检查未通过而停止，
没有 `passed.json`，没有正式训练。小 batch/随机图检查不是 batch=32 真实增强 AMP 通过记录。
没有发现 SSH 配置、可复用 SSH 进程或可调用服务器连接，因此未在服务器启动 tmux 或训练。
本地可访问的历史 v1 权重只用于兼容性重载，原 tar.gz、b19/v1 权重、结果和历史工作树全部保留。

## 服务器：建立独立 worktree 并启动唯一训练

在原 b19 Python 环境执行以下整段。已存在目录只接受干净且与远端实验分支同提交的工作树，任何冲突停止，不覆盖。

```bash
(
    set -euo pipefail
    BASE=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26
    WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_DCRStrip_v2
    BRANCH=exp-yolo26n-b19-dcrstrip-v2
    cd "$BASE"
    git status --short
    git fetch origin "$BRANCH"
    if [ -e "$WORK" ]; then
        test "$(git -C "$WORK" rev-parse --show-toplevel)" = "$WORK"
        git -C "$WORK" status --short
        test -z "$(git -C "$WORK" status --porcelain)"
        test "$(git -C "$WORK" rev-parse HEAD)" = "$(git rev-parse "origin/$BRANCH")"
    else
        git worktree add --detach "$WORK" "origin/$BRANCH"
    fi
    cd "$WORK"
    git log -1 --oneline
    if tmux has-session -t y26_dcr_v2 2>/dev/null; then
        echo 'y26_dcr_v2 already exists; inspect it, do not start a duplicate.' >&2
        exit 1
    fi
    tmux new-session -d -s y26_dcr_v2 -c "$WORK" 'bash tools/experiments/server_b19_dcrstrip_v2.sh train'
    sleep 3
    tmux capture-pane -pt y26_dcr_v2 -S -50
)
```

`train` 自动新进程预检，通过后启动正式训练。脚本设置当前 worktree 的 PYTHONPATH，打印真实导入路径/版本，
校验原始权重与配方。tee 保留管道失败状态；尚未通过预检不代表训练成功。
只预检可用 `bash tools/experiments/server_b19_dcrstrip_v2.sh preflight`，无需先后重复同一批检查。

查看实际进度：

```bash
tmux attach -t y26_dcr_v2
# Ctrl+B, D 分离
tail -n 40 /root/autodl-tmp/projects/Tunnel_Disease_YOLO26_DCRStrip_v2/runs/detect/yolo26n_b19_a1_dcrstrip_v2_train.console.log
```

正式目录是 `$WORK/runs/detect/yolo26n_b19_a1_dcrstrip_v2`，内部 `train.log`、`results.csv`、`provenance/`、
`weights/best.pt`。外部 `yolo26n_b19_a1_dcrstrip_v2_train.exit_status` 在脚本退出时记录状态，0 才是完成。
预检启动失败时可能没有正式目录，查看 console.log 和 `_preflight/`。

## 训练结束：单次 test、验证诊断和打包

等 train 退出且状态为 0 后执行一次：

```bash
(
    set -euo pipefail
    cd /root/autodl-tmp/projects/Tunnel_Disease_YOLO26_DCRStrip_v2
    test "$(cat runs/detect/yolo26n_b19_a1_dcrstrip_v2_train.exit_status)" = 0
    bash tools/experiments/server_b19_dcrstrip_v2.sh test
    bash tools/experiments/server_b19_dcrstrip_v2.sh diagnose
    bash tools/experiments/server_b19_dcrstrip_v2.sh package
)
```

test 只加载验证集选出的 `best.pt`，一次 `YOLO.val` 导出 P/R/mAP50/mAP50-95 原始浮点数、速度、完整 args、
图表、predictions.json、test.log。设置固定为 split=test、640、batch=32、workers=8、device=0、conf=0.001、
iou=0.7、max_det=300、rect=True、augment=False。8.4.98 的 cfg 将旧 half=False 映射到 quantize=None；
validator 仅在 quantize==16 时启用 fp16。本入口显式 quantize=None，即原 test 常规 FP32，无 TTA/INT8。
test 子目录存在即拒绝再次评估，失败目录也保留以供检查；打包入口不会偷偷补跑 test。

训练中只在开始、50/100/150 轮及结束读取 alpha/beta 标量到 `dcr_v2_scalars.jsonl`，不额外取 batch 或 forward。
诊断单独加载 best.pt，在 eval/no_grad FP32 下对排序固定的前四张真实 val 图执行无增强 640 letterbox，记录
图片路径/哈希、权重/源码版本、方向权重均值/标准差、q 范围、U 与 beta\*V 范数/比例、每图残差/主路范数比例。
没有真实数据或 best.pt 时失败，不用随机张量冒充验证诊断。

打包输出为 `$WORK/runs/detect/yolo26n_b19_a1_dcrstrip_v2.tar.gz`，包含本轮 args/results/train.log/best.pt、
test 和验证图表/精确指标、诊断、标量记录、预检/来源、所需源码与 b19_reference.json、Git source.tar、环境。
排除数据集、模块包整体、周期权重和 last.pt；逐文件核对 SHA-256 并读完整 gzip 验证 CRC。
目标已存在即失败，不覆盖旧包。package 独立阶段可在完成诊断/test 后单独运行。

## 历史对照（不重新评估）

来源为本次用户提供的结果与可访问历史归档，test 是控制台取整值；不能当作本轮未取整精确指标。

| 指标      | b19 val 最佳第 196 轮 | v1 val 最佳第 200 轮 | b19 test | v1 test |
| --------- | --------------------: | -------------------: | -------: | ------: |
| Precision |               0.86477 |              0.85837 |    0.875 |   0.872 |
| Recall    |               0.78559 |              0.80402 |    0.784 |   0.772 |
| mAP50     |               0.86072 |              0.87055 |    0.870 |   0.863 |
| mAP50-95  |               0.50285 |              0.50540 |    0.508 |   0.501 |

val 2404 图/2985 目标，test 1202 图/1477 目标；两组 200 轮、seed=42。val 混淆矩阵 FN 449→419、
FP 690→692；test FN 215→239、FP 361→359，不据此声称大量新增背景误检。v1 严格 AP 持续同轮优势主要在
182–200 轮，因此仍保留原 200 上限与 60 patience。v1 alpha≈0.831543、gate≈[0.317627,0.959473]
不单独证明残差过强、参数过多或两侧差异受惩罚。本轮没有内部消融，因果判断仍未建立。
