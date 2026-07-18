# YOLO26n 隧道裂缝检测 baseline 实验归档

## 实验信息

| 项目 | 值 |
| --- | --- |
| 实验 ID | `yolo26n_baseline_e150_img640_b32_seed42` |
| 任务名称 | YOLO26n 单类别隧道裂缝目标检测 baseline |
| 实验日期 | 2026-07-18（训练完成；最终测试日志本地时间为 2026-07-19 00:00） |
| 实验状态 | 已完成训练与最终测试，锁定为后续模型改进的 baseline |
| 模型 | YOLO26n，COCO 预训练权重 `yolo26n.pt` |
| 分支 | `exp-yolo26n-baseline-crack` |
| baseline 源码提交 | `cbd8606758e22ab7282cd859a918d0c38ffba558`（`cbd8606`，`build offline augmented tunnel crack dataset`） |

本文件仅归档已下载的服务器实验产物。本次归档没有训练、推理、数据集重建、模型源码修改或数据修改。

## 数据集

数据集版本为 `Tunnel_Crack_AugFirst_5x`。以下数量来自
`tunnel_project/reports/crack_augfirst_build_report.json`、服务器 `data_server.yaml`、训练日志和测试日志：

| 项目 | 数量或说明 |
| --- | --- |
| 原始图片 | 20,206 |
| 原始裂缝类别 | ID `2` |
| 新数据类别 | ID `0`: `crack` |
| 原始含裂缝图片 | 2,404 |
| 有效裂缝原图 | 2,312 |
| 原始/有效裂缝框 | 2,941 / 2,815 |
| 离线增强版本 | `orig`、`hflip`、`vflip`、`bright120`、`gauss_s10`，每个有效 source 各 5 个版本 |
| 最终图片 | 11,560 |
| train | 8,092 张，9,877 个框 |
| val | 2,312 张，2,816 个框 |
| test | 1,156 张，1,382 个框 |
| 划分方法 | 完成离线增强后，以 variant 为样本、seed=42 确定性随机划分 7:2:1 |

该划分不是 source 分组划分。同一 `source_stem` 的不同增强版本允许进入不同 split；实际有 1,925/2,312
个 source（83.2612%）跨越多个 split。因此，测试集只是 variant 级留出集，不能称为完全独立来源测试集。

数据构建只保留至少含一个有效裂缝框的原图，没有加入无裂缝负样本。训练、验证和测试扫描均为 0 个
background 样本。这会限制对真实无裂缝场景中误报率的估计，也是当前 baseline 的重要局限。

服务器数据配置为：

```yaml
path: /root/autodl-tmp/projects/Tunnel_Disease_YOLO26/datasets/Tunnel_Crack_AugFirst_5x
train: images/train
val: images/val
test: images/test
names:
  0: crack
```

## 训练参数

核心参数如下；数值直接取自训练目录中的 `args.yaml`。

| 参数 | 值 |
| --- | --- |
| model | `/root/autodl-tmp/projects/Tunnel_Disease_YOLO26/yolo26n.pt` |
| data | `/root/autodl-tmp/projects/Tunnel_Disease_YOLO26/datasets/Tunnel_Crack_AugFirst_5x/data_server.yaml` |
| epochs / imgsz / batch | 150 / 640 / 32 |
| device / workers | `0` / 8 |
| optimizer | `MuSGD` |
| lr0 / lrf | 0.01 / 0.01 |
| momentum / weight_decay | 0.937 / 0.0005 |
| seed / deterministic | 42 / true |
| patience | 0 |
| cache / amp | false / true |
| mosaic / close_mosaic | 1.0 / 10 |
| fliplr / flipud | 0.5 / 0.0 |
| HSV | `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4` |
| 几何增强 | `degrees=0.0`, `translate=0.1`, `scale=0.5`, `shear=0.0`, `perspective=0.0` |
| 组合增强 | `mixup=0.0`, `cutmix=0.0`, `copy_paste=0.0` |
| save_period | 10 |

完整有效参数如下，保持 `args.yaml` 的键和值：

```yaml
task: detect
mode: train
model: /root/autodl-tmp/projects/Tunnel_Disease_YOLO26/yolo26n.pt
data: /root/autodl-tmp/projects/Tunnel_Disease_YOLO26/datasets/Tunnel_Crack_AugFirst_5x/data_server.yaml
epochs: 150
time: null
patience: 0
batch: 32
imgsz: 640
save: true
save_period: 10
cache: false
device: '0'
workers: 8
project: /root/autodl-tmp/projects/Tunnel_Disease_YOLO26/runs/tunnel_crack
name: yolo26n_baseline_e150_img640_b32_seed42
exist_ok: false
pretrained: true
cls_remap: true
optimizer: MuSGD
verbose: true
seed: 42
deterministic: true
single_cls: false
rect: false
cos_lr: false
close_mosaic: 10
resume: false
amp: true
fraction: 1.0
profile: false
freeze: null
multi_scale: 0.0
compile: false
overlap_mask: true
mask_ratio: 4
dropout: 0.0
val: true
split: val
save_json: false
conf: null
iou: 0.7
max_det: 300
quantize: null
dnn: false
plots: true
end2end: null
source: null
vid_stride: 1
stream_buffer: false
visualize: false
augment: false
agnostic_nms: false
classes: null
retina_masks: false
embed: null
show: false
save_frames: false
save_txt: false
save_conf: false
save_crop: false
show_labels: true
show_conf: true
show_boxes: true
line_width: null
format: torchscript
keras: false
optimize: false
dynamic: false
simplify: true
opset: null
workspace: null
nms: false
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
distill_model: null
dis: 6.0
box: 7.5
cls: 0.5
cls_pw: 0.0
dfl: 1.5
pose: 12.0
kobj: 1.0
rle: 1.0
angle: 1.0
nbs: 64
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
bgr: 0.0
mosaic: 1.0
mixup: 0.0
cutmix: 0.0
copy_paste: 0.0
copy_paste_mode: flip
auto_augment: randaugment
erasing: 0.4
cfg: null
tracker: tracktrack.yaml
save_dir: /root/autodl-tmp/projects/Tunnel_Disease_YOLO26/runs/tunnel_crack/yolo26n_baseline_e150_img640_b32_seed42
```

## 服务器环境

| 项目 | 值 |
| --- | --- |
| 平台 | AutoDL |
| GPU | NVIDIA GeForce RTX 4090，24 GB（日志可用显存 24,081 MiB；`nvidia-smi` 总显存 24,564 MiB） |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| CUDA runtime | 12.8（`nvidia-cuda-runtime-cu12==12.8.90`） |
| Ultralytics | 8.4.98，editable checkout 位于提交 `cbd8606758e22ab7282cd859a918d0c38ffba558` |
| NVIDIA 驱动 | 580.105.08 |
| NVIDIA-SMI | 580.105.08；显示的 CUDA 13.0 是驱动支持上限，不是本实验 PyTorch runtime 版本 |
| CUDA 关键库 | cuDNN 9.10.2.21，NCCL 2.27.3 |
| 视觉/数据依赖 | NumPy 2.3.2，OpenCV 5.0.0.93，Pillow 11.3.0，Polars 1.42.1，PyYAML 6.0.2 |
| TorchVision / THOP | torchvision 0.23.0+cu128，ultralytics-thop 2.0.20 |

## 训练成本与模型规模

- 150 轮训练耗时 4.190 小时（15,084 秒）。
- 训练前模型摘要为 2,504,190 个参数、5.8 GFLOPs；融合后的最终模型摘要为 2,375,031 个参数、5.2 GFLOPs。
- 本地 `best.pt` 为 5,383,045 字节，即 5.13 MiB；服务器日志按十进制四舍五入显示约 5.4 MB。
- 本地 `best.pt` SHA256：`24b25d33381fa48d29749a7848dbf6889eb85cac9ee26fd6ccb840c76542fdb7`。

## 结果

### 验证集

`results.csv` 共 150 轮。第 150 轮同时是最终轮和 `mAP50-95` 最高轮，因此最终值与本次最佳值一致：

| Images | Instances | Precision | Recall | mAP50 | mAP50-95 | 最佳轮次 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,312 | 2,816 | 0.93193 | 0.86540 | 0.93845 | 0.69469 | 150 |

训练日志随后加载验证集选择得到的 `best.pt` 并复验，日志以三位小数显示为
`P=0.932, R=0.865, mAP50=0.938, mAP50-95=0.694`。

### 测试集

最终测试使用验证集选出的 `best.pt`，没有使用测试集选择权重。独立下载的测试日志给出：

| Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,156 | 1,382 | 0.932 | 0.852 | 0.929 | 0.682 |

测试 PR 曲线标注 `mAP@0.5=0.929`，验证 PR 曲线标注 `mAP@0.5=0.938`，与日志和 CSV 的四舍五入结果一致。
测试归一化混淆矩阵在其固定绘图阈值下显示 crack 单元格约 0.90；该图使用的阈值口径与汇总 P/R
工作点不同，不应直接拿 0.90 与汇总 Recall 0.852 等同，二者不构成结果冲突。

## 训练曲线分析

- `train/box_loss`、`train/cls_loss`、`train/dfl_loss` 和对应验证损失整体下降。
- 验证损失在训练后期继续缓慢下降，没有出现持续反弹；当前曲线未显示明显过拟合。
- `mAP50-95` 的最高值出现在第 150 轮，尚无提前饱和后退化的证据。
- `close_mosaic=10` 在最后 10 轮关闭 Mosaic。第 141 至 150 轮训练 box/cls/dfl loss 分别下降约
  0.20293、0.15823、0.00158，曲线上表现为明显下降；这是训练分布改变后的预期现象，不能单独解释为泛化性能突增。

由于测试集存在 source 跨 split 且没有无裂缝负样本，以上高指标只应作为当前数据协议下的 baseline，不能外推为
完全独立隧道来源或真实负样本环境下的最终性能。

## 等价复现实验命令

原始 shell 命令未包含在下载产物中。下面命令依据 `args.yaml` 和日志重建，可复现相同的核心训练设置；服务器路径按实际目录记录。

```bash
yolo detect train \
  model=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26/yolo26n.pt \
  data=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26/datasets/Tunnel_Crack_AugFirst_5x/data_server.yaml \
  epochs=150 imgsz=640 batch=32 device=0 workers=8 \
  optimizer=MuSGD lr0=0.01 momentum=0.937 weight_decay=0.0005 \
  seed=42 deterministic=True patience=0 cache=False amp=True \
  mosaic=1.0 close_mosaic=10 fliplr=0.5 save_period=10 \
  project=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26/runs/tunnel_crack \
  name=yolo26n_baseline_e150_img640_b32_seed42
```

测试命令同样是依据最终测试目录和日志重建的等价命令：

```bash
yolo detect val \
  model=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26/runs/tunnel_crack/yolo26n_baseline_e150_img640_b32_seed42/weights/best.pt \
  data=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26/datasets/Tunnel_Crack_AugFirst_5x/data_server.yaml \
  split=test imgsz=640 batch=32 device=0 plots=True \
  project=/root/autodl-tmp/projects/Tunnel_Disease_YOLO26/runs/tunnel_crack \
  name=yolo26n_baseline_e150_img640_b32_seed42_test
```

## 本地证据文件

以下路径均相对于项目根目录，位于已被 Git 忽略的 `runs/` 中；权重和运行产物不纳入 Git：

- 训练结果目录：
  `runs/server_downloads/yolo26n_baseline_extracted/runs/tunnel_crack/yolo26n_baseline_e150_img640_b32_seed42/`
- 完整训练参数与逐轮指标：上述目录中的 `args.yaml`、`results.csv`
- 环境与源码证据：上述目录中的 `environment_pip_freeze.txt`、`nvidia_smi.txt`、`git_commit.txt`
- 数据配置证据：上述目录中的 `data_server.yaml`、`dataset_build_config_crack_augfirst.yaml`
- 权重：上述目录中的 `weights/best.pt`（仅就地读取并计算哈希，没有复制到 Git 跟踪目录）
- 训练日志：
  `runs/server_downloads/yolo26n_baseline_extracted/logs/train/yolo26n_baseline_e150_img640_b32_seed42.log`
- 测试日志：`runs/server_downloads/yolo26n_baseline_e150_img640_b32_seed42_test.log`
- 测试图形目录：
  `runs/server_downloads/yolo26n_baseline_extracted/runs/tunnel_crack/yolo26n_baseline_e150_img640_b32_seed42_test/`
  （含 `BoxPR_curve.png`、`BoxP_curve.png`、`BoxR_curve.png`、`BoxF1_curve.png`、`confusion_matrix.png` 和
  `confusion_matrix_normalized.png`）

## 结论

该实验在当前增强后 variant 级划分协议下完成了可复核的训练和测试，现锁定为后续模型改进所使用的
YOLO26n baseline。后续改进实验应保持本归档的训练设置或明确记录变量，同时优先补充 source 隔离测试和无裂缝负样本评估。
