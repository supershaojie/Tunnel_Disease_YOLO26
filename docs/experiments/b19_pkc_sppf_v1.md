# b19 PKC-SPPF v1 实验交付

## 核对依据与范围

从 b19 的 `4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6` 创建独立工作树，分支为
`codex/exp-yolo26n-b19-pkc-sppf-v1`。原工作树、其他实验和结果保持原状；本次不创建 PR、不执行 SSH。

完整阅读的设计文档副本为 [YOLO26_b19_PKC_SPPF_v1_design.md](YOLO26_b19_PKC_SPPF_v1_design.md)。
真实 b19 args、原始预训练权重、模块包及三个参考源码的路径和 SHA256 见
[b19_pkc_sppf_v1_audit.json](b19_pkc_sppf_v1_audit.json)。
本机 `D:/7.21yolo26改/YOLO26缝合.zip` 的 SHA256 与设计文档中的 UUID 名模块包一致，
并逐字节核实三个解压源码与 zip 成员相同。已读取 HLKConv、MRFAConv 和 SMMM 源码，但没有导入这些模块。
历史启动命令从既有 SIR 任务附件第 9 节核对，仅将其作为 b19 记录，未执行其中的其他实验指令。

## 唯一结构变化

nano 单类配置只将第 9 层 `SPPF` 替换为 `SPPF_PKC`，参数保持 `[1024, 5, 3, True]`。
原生 `yolo26.yaml`、`block.py`、Conv、Detect、detach、损失、分配器、后处理和训练循环均未修改。
新类继承 SPPF，保留 `cv1/cv2/m/n/add` 及原 state_dict 路径。

```text
Z = cv1(X)                         # 原生 Conv+BN, act=False
Y0 = cv2(cat(Z,M1,M2,M3)) + X       # 仅当原生 add=True 才加 X
U = Conv1x1(128,32)+BN+SiLU(Z)
L5 = DW5(d=1,p=2)+BN+SiLU(U)
L9 = DW3(d=2,p=2)+BN+SiLU(L5)
L13 = DW3(d=2,p=2)+BN+SiLU(L9)
Y = Y0 + Project1x1(cat(L5,L9,L13))
```

Project 是无偏置、无 BN、无激活的 96→256 普通卷积，仅该投影权重初始化为零。
其他新增卷积正常初始化，BN gamma=1、beta=0；全部 stride=1、bias=False，空间卷积 groups=32。
本轮仅提供 nano r=32 实验，没有核组合或宽度扫描、门控、方向卷积或其他实验结构。
新增分支构造在 CPU RNG fork 中完成，从而保持后续原生层（包括未匹配检测头）的初始化随机流。
实测总参数 2,534,494，比原生 2,504,190 增加 30,304（约 1.21%）。
640 输入的新增卷积静态预算为 0.0240384 GFLOPs，未计 BN/激活/访存，不是延迟测量。

Deleted: 新实验 YAML 中第 9 层原生 SPPF 选项被替换；从复用脚本中删除 SIR 路由公式、诊断和版本包装依赖。
原生源码没有可删除的故障路径；新结构必须新增独立模块，训练/评估/打包复用并整合已有实验契约，避免修改其他实验。

## 配置与预检

所有有效训练字段由真实 b19 完整 args 快照继承，并逐项比对。只允许结构与实验输出位置变化，
数据和预训练文件允许解析到同一内容的绝对路径。b19 原 args 或启动记录在服务器缺失时，使用有明确来源的归档快照并记录来源。
原始 `yolo26n.pt` 必须匹配 SHA256
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`，不用 b19 的 best.pt 初始化。

预检及正式训练始终 batch=32、imgsz=640、MuSGD、AMP=True；epochs=200、patience=60、seed=42、workers=8，
学习率、warmup、所有增强和其他参数完整继承 b19。原生早停行为保留。OOM 不降低 batch，不修改优化器或精度。
原生训练周期内的验证 loader 行为也保持 b19；独立预检 Validator 和训练后的独立 val/test 配置均为 batch=32。

`train` 自动以独立子进程执行缺失的预检，成功后新建 Trainer，从原始权重和原 seed 开始。
预检最多使用 16 个真实 batch32，沿用原生 warmup/accumulation、GradScaler、MuSGD、clip=10 和 EMA 更新顺序，
记录溢出跳步；必须观察到投影及所有上游新增参数具有有限任务梯度和实际更新。
另用独立模型进行 FP32 batch32 反向检查，随后验证 EMA、保存重载、fuse 和真实独立 Validator。
预检模型不用于正式训练，预检精度不作为最终结果。共享张量、源码、初始权重、数据清单和环境指纹均记录。
同名训练目录已存在时保留该目录并退出；失败预检和各命令日志保存在独立 attempt 目录中。

## 本地验证与服务器待验证项

本地 Windows / Python 3.11.15 / torch 2.7.1+cu118，所有已交付数值测试明确使用 CPU。
本地检查包括：640×640/640×960 全模型零投影输出完全相等；所有 708 个共有状态完全一致；
原始预训练 606 个可匹配状态正确加载；非零投影公式和条件残差；5/9/13 连续支持；
合成标注 batch32、64×96 输入的真实原生 MuSGD 两步更新；首步仅投影梯度非零，第二步上游梯度和权重更新；
BN、EMA、FP16 快照的新进程 FP32 重载，以及独立副本上的原生/PKC 同条件 fuse 检查。
另有 16 张合成图片的原生 Dataset 诊断流程测试、PR/FPPI 曲线空预测与同分数测试、固定 batch OOM 行为测试。
这些合成数据仅用于代码验证，不代表隧道真实数据效果。

重载比较使用完全相同的独立快照，要求 state/raw 精确相等。
fuse 比较保留 atol=rtol=1e-4，并记录原生对照。
64×96 下所有 126 个 anchor 都被保留；接近相同的分数可能使 top-k 行排序改变，故按实际 anchor 对齐完整
box/score/class，并同时比较全部未排序解码结果，没有放宽阈值或修改模型后处理。

**待服务器验证：** 原 b19 Python 3.12.3 / torch 2.8.0+cu128 / RTX4090 环境，
真实数据 batch32/640 的 AMP、FP32、MuSGD、EMA、Validator 预检，正式训练，最终 val/test 和真实特征诊断。
未执行服务器训练，未声称模型涨点或首创。

本地可复核命令（从本实验工作树运行，不调用服务器预检）：

```bash
PYTHONPATH="$PWD" PKC_TEST_PRETRAINED=/path/to/original/yolo26n.pt \
  python -m unittest discover -s tests -p test_pkc_sppf.py -v
python tools/experiments/run_b19_pkc_sppf.py --help
python tools/experiments/finish_b19_pkc_sppf.py --help
bash -n tools/experiments/server_b19_pkc_sppf_v1.sh
```

## 服务器部署与 tmux

以下命令由用户在服务器执行。交付消息给出完整提交 SHA；部署时可将 PKC_SHA 固定为该 SHA。
下方默认解析刚 fetch 的实验分支，并打印实际完整 SHA。新工作树路径已存在时 Git 会停止，不覆盖其他工作。

```bash
BASE=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_pkc_sppf_v1
BRANCH=codex/exp-yolo26n-b19-pkc-sppf-v1
git -C "$BASE" fetch origin "$BRANCH"
PKC_SHA=$(git -C "$BASE" rev-parse FETCH_HEAD)
printf 'PKC source: %s\n' "$PKC_SHA"
git -C "$BASE" worktree add --detach "$WORK" "$PKC_SHA"
cd "$WORK"
export B19_PYTHON=/root/miniconda3/bin/python
PYTHONPATH="$WORK" "$B19_PYTHON" -c 'import sys,torch,ultralytics; print(sys.executable); print(torch.__version__); print(ultralytics.__file__)'
git rev-parse HEAD

# 可选：先单独预检；train 会自动补做尚未通过的有效预检。
bash tools/experiments/server_b19_pkc_sppf_v1.sh preflight

# 启动唯一一组正式训练；tmux 名存在时不再启动。
tmux new-session -d -s y26_pkc_v1 -c "$WORK" && \
  tmux set-option -t y26_pkc_v1 remain-on-exit on && \
  tmux send-keys -t y26_pkc_v1:0.0 'B19_PYTHON=/root/miniconda3/bin/python bash tools/experiments/server_b19_pkc_sppf_v1.sh train; rc=$?; printf "\nPKC train exit=%s\n" "$rc"' C-m
```

脚本从自身位置定位代码，通过 PYTHONPATH 使用实验工作树，不修改原环境的 editable 安装。
预训练和数据仍读取原 b19 根目录。不会终止其他 GPU 任务，也不做自动等待或后台监控。

## 进度查询及训练后命令

```bash
WORK=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26_pkc_sppf_v1
NAME=yolo26n_b19_pkc_sppf_v1
RUN="$WORK/runs/detect/$NAME"
tmux ls
tmux capture-pane -pt y26_pkc_v1:0.0 -S -80
tmux attach-session -t y26_pkc_v1
# 退出 tmux 查看模式：Ctrl-b 后按 d，训练继续。
tail -n 60 "$WORK/runs/detect/${NAME}_train.console.log"
[ ! -f "$RUN/results.csv" ] || tail -n 5 "$RUN/results.csv"
[ ! -f "$WORK/runs/detect/${NAME}_train.exit_status" ] || cat "$WORK/runs/detect/${NAME}_train.exit_status"
nvidia-smi

# 训练成功结束后，依次运行；失败则保留证据并停止后续步骤。
cd "$WORK"
bash tools/experiments/server_b19_pkc_sppf_v1.sh test && \
  bash tools/experiments/server_b19_pkc_sppf_v1.sh diagnose && \
  bash tools/experiments/server_b19_pkc_sppf_v1.sh package
(cd "$WORK/runs/detect" && sha256sum -c "${NAME}.tar.gz.sha256")
```

`test` 包含独立 FP32 val 和 test 各一次：imgsz640、batch32、conf0.001、iou0.7、max_det300、
rect=True、augment=False、workers8、device0、quantize=None，保留原生 end2end。
记录完整 args、split、best.pt SHA256、源码与数据指纹、P/R/AP50/AP75/mAP50-95、PR 曲线、混淆矩阵、
预测 JSON 和同一次 native matching 得到的 Precision/Recall/FPPI 操作曲线。
相同 Precision 或 FPPI 的 b19 对比必须使用相同 split 与评估条件的 b19 数据；脚本不伪造对比收益，
不根据 test 选择阈值。若以后需要运行阈值，只在 val 选择并冻结到 test。

`diagnose` 固定按文件名排序的前 16 张 val 图，记录三级特征及任务梯度统计、
`||Delta||/(||Y0||+1e-6)`，以及同一训练模型分支开启/关闭的 TP/FP/FN 差异（conf0.25、匹配IoU0.5）。
梯度来自独立 eval 模型和原生有标注检测损失，不更新参数或 BN；保留 one2one detach。
这只是推理旁路诊断，不等于重新训练消融，也不是相对 b19 的性能增益。

`package` 只打包本次 run 的已有有效结果、best.pt、源码归档/patch、配置、日志、环境及检查证据，
验证包内文件哈希和 gzip CRC，生成 `${NAME}.tar.gz` 与 `.sha256`。
不包含原始数据集、其他实验、整个模块包、.git、Conda、last.pt 或 epoch\*.pt。
模型需要本次分支中的自定义类，不能把自定义 best.pt 当作任意原生环境均可独立加载的文件。
