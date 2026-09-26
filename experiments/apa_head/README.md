# APA-Head / b19 独立实验

正式训练状态：**NOT_STARTED**。本次仅交付实现、有限工程检查和服务器后续训练入口；不以工程检查代替精度实验，不保证涨点。本实验不组合 QCA 或其他创新分支。

## 固定来源

- b19 代码锚点：`4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6`；分支 `exp/apa-head-b19`。
- 完整训练参数来源：`E:/ditieyolo26跑结果/8.12离线在线/b19 200e/b19_yolo26n_e200_train_val_test_20260823_224153/train_run/args.yaml`；源文件 SHA256 `b08b915756bf85c91d3356586a651a37156d71867b84b6e8493d75c9568642b9`。
- 同一归档中 `environment/git_state.txt` 确认上述锚点；`logs/b19_y26n_diverse5x_e200_i640_b32_musgd_b8b9hybrid_s42.log` 的完整 trainer 启动参数与该 YAML 一致。未找到原始 shell 启动脚本。
- 原始初始化：`E:/PycharmProjects/Tunnel_Disease_YOLO26/yolo26n.pt`；实测 SHA256 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`。禁止使用任何实验 best、smoke 或 EMA 权重。
- 本地数据：`E:/PycharmProjects/Tunnel_Disease_YOLO26/datasets/Tunnel_Crack_AugFirst_Diverse5x_RandomSplit_7_2_1_seed42/data.yaml`，`nc=1`，类别 `crack`；图像和标签实数均为 train/val/test `8414/2404/1202`。YAML SHA256 `1f18760508e9dbf2332cd7102ee9c15e11e08f3d15fc4202c8ee0ed1bb785b12`。
- 数据沿用历史按文件随机划分；同一原图的不同增强版本可能跨集合，不能声称按原图来源完全独立。test 只作最终固定协议报告，不选结构、超参数或 checkpoint。
- 历史服务器为 Python 3.12.3 / PyTorch 2.8.0+cu128 / RTX 4090；本地可用 `D:/miniconda3/envs/yolo26/python.exe` 为 Python 3.11.15 / PyTorch 2.7.1+cu118 / RTX 2060。当前服务器路径、运行环境和文件内容未远程核验。

## 配方与入口

`b19_train.yaml` 保存全部 112 个历史有效参数。与来源仅有以下五个值不同：

| 字段     | 历史值                                                  | 本实验值                                                                      |
| -------- | ------------------------------------------------------- | ----------------------------------------------------------------------------- |
| model    | `yolo26n.pt`                                            | `ultralytics/cfg/models/26/yolo26n-apa.yaml`                                  |
| data     | 历史服务器母仓库中的绝对数据 YAML                       | `datasets/Tunnel_Crack_AugFirst_Diverse5x_RandomSplit_7_2_1_seed42/data.yaml` |
| project  | 历史服务器母仓库 `runs/detect`                          | `runs/apa_head`                                                               |
| name     | `b19_y26n_diverse5x_e200_i640_b32_musgd_b8b9hybrid_s42` | `apa_head_b19_e200_s42`                                                       |
| save_dir | 历史 b19 运行目录                                       | `null`，由原生 Trainer 根据 project/name 生成                                 |

无训练数值改动。`train.py` 将路径解析为绝对路径，允许显式指定 data、weights、project、name、device。`pretrained: true` 在调用时解析为经 SHA256 验证的原始权重路径，让原生 Trainer 直接从公共权重做 nc 适配；始终 `resume=False`。模型 YAML 决定新结构，沿用原生 DetectionTrainer、loss、分类名字适配和 MuSGD 分组；不修改环境、不冻结骨干、不使用 smoke 状态。

关键配方为 200 epochs、patience 60、640、batch 32、MuSGD、seed 42、lr0 0.01、lrf 0.003、momentum 0.937、weight_decay 0.0005、AMP、cosine LR、workers 8。完整 warmup、增强、loss 和评估参数以 YAML 为准。`nbs=64`，正式梯度累积随 warmup 从 1 到 2，不能把 smoke 的累积设置带入正式实验。b19 原生 O2M 权重按 epoch 从 0.8 线性降到 0.1，O2O 为其补数；O2M topk=10，O2O topk=7/topk2=1。

`deterministic=true` 沿用 b19 的 `torch.use_deterministic_algorithms(True, warn_only=True)`；CUDA `grid_sample` 反向可能提示非确定性。这不等于完整可重复保证，不能为了消除警告修改全局设置。ONNX/TensorRT 导出和长时间性能基准不属于本轮交付。

## 实现与计算来源

`ultralytics/nn/modules/apa.py` 中的 `DetectAPA` 继承原生 `Detect`，保留 cv2/cv3、O2O detach、唯一一次解码、stride 乘法与后处理。`apa` 和 `one2one_apa` 各含 P3/P4/P5 三个 `AxisPairedAlignment`，总计六组独立参数。模型入口是 `ultralytics/cfg/models/26/yolo26n-apa.yaml`；原生参数形状、BN buffer 和键路径不变。

每组从分类/回归最终预测层之前的隐藏特征预测两个二维偏移，分类 logit 不变，但定位损失可以经 APA 回传分类隐藏塔。边界场、完整框有效性掩码、归一化采样及零网格差分使用 FP32；最后增量转回原生 raw dtype。有限无效中心保留原值，非有限 raw 显式报错。没有零偏移快捷返回，没有新增监督或学习门。

单类别 n 模型原生参数为 2,504,190，APA 为 2,510,166，新增 **5,976**（每组 996）。原生融合删除 O2M 后，新增 O2O 参数为 2,988。640 输入三个尺度共有 8,400 个位置，仅新增 1×1 卷积每分支约 8.064M MAC，双分支约 16.128M MAC（按一次乘加算两个 FLOP 为 32.256M FLOP）。这些数值**不含** SiLU/tanh、掩码、除法、网格生成和插值。

简单实现每组四次 `grid_sample`，每次采样两条边及同一个有效质量通道；双分支共 24 次，融合后 12 次。还有非有限输入检查造成的设备同步、FP32 临时张量和每次在当前 device 构造网格的开销。未缓存网格，不增加 state_dict buffer。未做长时间延迟基准，不能将常规 FLOPs 统计漏算的 functional 操作视为零成本。

## 有限检查结果

结果见 `validation.json`，入口为 `validate.py`。本地 Python 3.11.15 / PyTorch 2.7.1+cu118 / RTX 2060，实际 import 指向本实验 worktree。

- A：3×5 仿射场、border、1×5 退化尺寸均通过，最大误差约 `4.77e-7` feature-grid。
- B：FP32/FP16 有效与有限无效 raw 零态误差均为 0；显式同步 708 个共同参数/buffer 后，两分支 top-k 前 boxes/scores 误差均为 0。无效邻居归一化采样通过，NaN/Inf 均明确失败。
- 预训练：606 个形状兼容的原始 tensor 逐键一致，102 个缺口属于原生 nc 适配，48 个状态键属于 APA，意外丢失原生键为 0；完整清单在 JSON。
- C：原生 Trainer 的 YAML 重建、dataloader、loss、MuSGD 分组和 optimizer_step 路径完成 3 次真实更新，loss 为 `16.24148 → 14.94914 → 14.14677`。所有参数恰好注册一次；六组 APA 在独立空间变化特征检查中均获得梯度，O2M 可回传输入而 O2O 不可。小尺寸合成检测标签下实际更新了 P3/P4 的四组，两组 P5 未更新；不把“可训练”写成“全部实际更新”。首步投影梯度为零，后续投影获得梯度。
- CUDA AMP 原生 loss 前后向有限；保留 `deterministic=True, warn_only=True` 并实测得到 `grid_sample` CUDA 反向非确定性警告，不能承诺逐位可重复。
- D：非零 APA 对 top-k 前 raw 的最大改变为 `1.824706` feature-grid；分类、deepcopy、保存重载误差为 0。EMA raw 最大误差约 `1.29e-5`，融合 raw 最大误差约 `2.05e-5`；融合预测最大差异 `0.000328` 像素。融合后保留三组 O2O APA；默认 predict、非方形输入与单张 640 输出 `[1,300,6]` 均通过。

A/B/C 结果来自完整检查；D 在修复预热输入后单独重跑，重新从公共权重构造 nc1 模型并故意注入非零偏移，未把 smoke 权重当作正式初始化。检查只使用合成标签，没有执行完整训练 epoch、真实数据集 val/test、正式 200e、B32 压力测试、导出或长期延迟测试。一个临时 FP32 checkpoint 在检查后自动删除；由于 b19 文件名处理会移除 Windows 路径中的英文单引号，验证临时目录建在公共权重所在的无单引号路径下。

复现检查（权重路径需按环境填写）：

```bash
YOLO_AUTOINSTALL=false python experiments/apa_head/validate.py --weights /path/to/original/yolo26n.pt
```

## 服务器交接

默认预测预热有一处必要的原生兼容修复：`ultralytics/nn/autobackend.py` 将 dummy 输入从 `torch.empty` 替换为 `torch.zeros`。b19 确定性模式在当前 PyTorch 下会将未初始化浮点张量填充为 NaN，导致 APA 按设计拒绝该输入。修复发生在创建预热输入的位置，未关闭确定性或吞掉非有限输入错误，未改变检测 loss、数据增强或正式训练参数。

以下命令供用户后续执行，本轮未连接或操作服务器。将 `APA_SHA` 替换为最终交付中已推送并核验的完整提交 SHA；其余路径先在服务器确认。使用一个尚不存在的独立 worktree 目录，保留现有实验。

```bash
set -euo pipefail
MOTHER=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26
APA_SHA='<填写最终交付的完整提交 SHA>'
EXP=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26-apa-head-b19
DATA="$MOTHER/datasets/Tunnel_Crack_AugFirst_Diverse5x_RandomSplit_7_2_1_seed42/data.yaml"
WEIGHTS="$MOTHER/yolo26n.pt"
PROJECT=/root/autodl-tmp/apa_head_runs

git -C "$MOTHER" fetch origin exp/apa-head-b19
git -C "$MOTHER" cat-file -e "$APA_SHA^{commit}"
git -C "$MOTHER" worktree add --detach "$EXP" "$APA_SHA"
cd "$EXP"
test "$(git rev-parse HEAD)" = "$APA_SHA"
export PYTHONPATH="$EXP${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_AUTOINSTALL=false
export NO_ALBUMENTATIONS_UPDATE=1
python - <<'PY'
from pathlib import Path
import ultralytics
expected = Path.cwd() / "ultralytics" / "__init__.py"
actual = Path(ultralytics.__file__).resolve()
print("ultralytics.__file__:", actual)
assert actual == expected.resolve(), (actual, expected)
PY
test -f "$DATA"
test -f "$WEIGHTS"
printf '%s  %s\n' 9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef "$WEIGHTS" | sha256sum -c -
python experiments/apa_head/train.py \
  --data "$DATA" --weights "$WEIGHTS" --project "$PROJECT" \
  --name apa_head_b19_e200_s42 --device 0
```

训练按相同 val 规则选择 checkpoint。未进行正式训练前，历史 b19 test mAP50–95 `50.7769939%` 仅作研究参照；归档日志显示取整后的 `0.508`，本轮未重算该全精度指标。
