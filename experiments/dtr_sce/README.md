# b19 + DTR-C2PSA + SCE-Fusion

分支：`exp/dtr-sce-b19`。组合正式训练：**NOT_STARTED**。本目录实现第四组组合消融，保留两个已完成单模块实验的冻结结构，不报告组合精度提升。
单模块旧 README 的 `NOT_STARTED` 仅为历史开发记录。

## 冻结来源与接线

| 来源              | 固定提交                                   |
| ----------------- | ------------------------------------------ |
| b19               | `4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6` |
| DTR               | `8cf1b0d09f2488d320525f131f7966cf20792345` |
| SCE（新分支起点） | `57b59daee717b5cb048f226adb8ad23909a6e95a` |

`ultralytics/nn/modules/dtr.py` 原样来自固定 DTR 提交；`sce.py` 保持固定 SCE 文件。
只在内存中 CRLF→LF 后计算 SHA256，分别为：

```text
dtr.py d429ea673adf36ca24361df31cf4eaa5c2fb326990abcfd15e8c38277604f53f
sce.py cb9194f0563f9ac5f22d0ad23a38f7e3644bcbe33c30321b6ef0ebb3c750c5ad
```

28 节点图沿用完整 SCE YAML，仅将层 10 换为 `C2PSA_DTR [1024,0.5,2,2,5]`。
第 0–9、11–22 层不变；`[16,19,22] → SCE23 → Index24/25/26 → Detect27`。
640 输入时 DTR 为 B×256×20×20，内部两个独立 128 通道块、两头、400×50 注意力；
SCE 在实际 P4 的 40×40 网格交换，输出恢复 64×80×80、128×40×40、256×20×20。
仅支持 n/RGB/nc1/crack、reg_max=1、end2end=True；原生 Detect、detach、E2ELoss、EMA、增强和早停不变。

复用 SCE parser 的多输出元数据与 Index 审计，不改普通 Concat/Index/Detect/C2PSA 行为。
受限加载沿用 `_SafeLoad` 对模块子包的自动安全类登记，入口设置 `ULTRALYTICS_SAFE_LOAD=true`。

## 原生构建与迁移

真实 `DTRSCETrainer.setup_model → BaseTrainer.setup_model → get_model` 先调用原生 DetectionTrainer，完成原始 nc80→nc1 适配。
组合图构建在 RNG 隔离范围内；恢复 CPU/CUDA/Python/NumPy 在原生构建边界的状态，而非重新 seed42。
共享 SCE 的迁移实现增加声明式删除/新增前缀，避免复制两套迁移逻辑；原 SCE 入口默认行为保留。

| 实测状态分项                           | 张量数 |  参数元素 | buffer 元素 |
| -------------------------------------- | -----: | --------: | ----------: |
| 原生 nc1 总计                          |    708 | 2,504,190 |      19,106 |
| 原始预训练继承并保留                   |    576 | 2,298,488 |      15,005 |
| 原生类别适配缺口（从参考模型原样继承） |    102 |    88,070 |       2,304 |
| 删除旧 `model.10.m.*`                  |     30 |   117,632 |       1,797 |
| 全部应保留状态                         |    678 | 2,386,558 |      17,309 |
| 其中保留层 10 的 cv1/cv2               |     12 |   132,096 |       1,026 |
| Detect23→27 完整前缀迁移               |    240 |   241,566 |       4,516 |
| 新 DTR 内部                            |    122 |   432,366 |       2,060 |
| 新 SCE                                 |    211 |   280,891 |       4,763 |

逐键检查形状、父模块类型、数值及独立存储；意外遗漏、重复映射为零。参考模型不注册到训练模型，构建后释放。
旧 Detect23 的任何状态都不会按同名加载到 SCE23。参数元素与 buffers 分开统计。
显式组合 `.pt` 由独立 checkpoint 路径加载，保存的 `pretrained` 不能覆盖已学状态；不会重新做原生→组合迁移。
正式新实验仍只允许 SHA 为 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` 的原始 `yolo26n.pt`，禁用 resume。

## 训练契约与入口

`baseline_args.yaml` 原样复制自固定 DTR 提交的完整 b19 存档；正式命令仍必须传入实际 `--baseline-args`。
全部非运行字段逐字段比较，不由摘要或新版本默认值拼配方；保持 batch32、640、200 上限、patience60、MuSGD、seed42、AMP。
正式入口核对 Python3.12.3、`/root/miniconda3/bin/python`、Torch2.8.0+cu128、TorchVision0.23.0+cu128、CUDA12.8、8.4.98、RTX4090、无 Albumentations。
环境不符明确终止；不安装、升级或降级包。`YOLO_AUTOINSTALL=false`；原生 AMP 检查使用同名本地已核验权重，失败则终止。

MuSGD 复用 SCE 的临时参数名称视图：真实 0–22，Detect27→临时23，SCE23→临时24；无参数 Index 不加入视图。
视图不前向、不保存在真实图中。567 个参数张量各入组一次，351 个保留原生参数的组类别、LR、decay、MuSGD 标记与 b19 一致；
O2M/O2O 分类分支保持 3×LR。LayerNorm、gate/router/lambda/位置偏置均按原生规则入组。

OOM 复用 SCE 的 `_oom_retries` setter，在原生降低 batch 前拒绝重试；不添加另一套重试上限，不修改基类。
输出目录原子创建，已存在即失败；显式 save_dir 防止自动追加数字。报告必须在正式输出目录之外。
dry-run/verify 使用临时目录和独立副本，不占正式输出名、不修改数据集。

| 入口             | 必填                                        | 可选                                                                                           |
| ---------------- | ------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `train.py`       | `--data --weights --baseline-args`          | `--project --name --device --dry-run --report`                                                 |
| `verify.py`      | `--data --weights --baseline-args --report` | `--device`                                                                                     |
| `validate.py`    | `--weights --data --project --name`         | `--device --split {val,test} --variant {b19,dtr,sce,dtr_sce} --diagnostic-samples {0,1,2,3,4}` |
| `launch_tmux.sh` | 首参数 `train` 或 `evaluate`                | 后续参数原样交给对应 Python 入口；`--help`                                                     |

所有 Python 入口支持 `--help`，device 默认 `0`，评估默认 val/dtr_sce/diagnostic-samples=0。
report 是 JSON 文件路径。默认 project 为当前 worktree 的 `runs/detect`，正式名为 `dtr_sce_b19_e200_i640_b32_s42`。
后续服务器 worktree 约定为 `/root/autodl-tmp/projects/Tunnel_Disease_YOLO26-dtr-sce-b19`，本次未创建服务器目录或连接服务器。

launcher 使用 Bash 数组和 `%q` 引用，先建立交互 shell，再送入独立任务脚本；额外开启 remain-on-exit。
管道结束紧接复制完整 PIPESTATUS，分别保留 Python/tee 退出码；打印结果、摘要与日志位置，任务结束后返回 shell。
默认会话为 `dtr-sce-b19` / `dtr-sce-eval`；同名会话拒绝覆盖并给查看命令。
重跑用明确的新 `DTR_SCE_SESSION` 和输出名。可用 `DTR_SCE_PYTHON` 指定已有绝对解释器路径、`DTR_SCE_LOG_DIR` 指定日志目录。
允许 GPU0 并发，不等待空卡、不停止其他实验。Linux/tmux 实际会话保留行为 **UNVERIFIED**，本地仅检查 Bash 语法。

## 参数量与本地有限验证

| 模型            | 静态预期 Unfused / Native fused | 本次组合实测 Unfused / Native fused |
| --------------- | ------------------------------- | ----------------------------------- |
| b19 + DTR + SCE | 3,099,815 / 2,968,160           | **3,099,815 / 2,968,160**           |

Unfused 公式为 2,504,190 − 117,632 + 432,366 + 280,891。Native fuse 同时折叠 BN 和移除端到端 Detect 的 O2M，
参数减少不能全部归因于 BN；不据此推断速度。原生日志中的通用 FLOPs 估计不作为完整自定义模块计数。

实际本机：Windows、Python3.11.15、Torch2.7.1+cu118、TorchVision0.22.1+cu118、RTX2060，已安装 Albumentations2.0.8。
本地 data YAML SHA 与历史 `1f18760508e9dbf2332cd7102ee9c15e11e08f3d15fc4202c8ee0ed1bb785b12` 相同；
8414/2404/1202 图与 val/test 2985/1477 框核对通过，不代表已比较服务器全部图像字节。

| 验证                                                                             | 状态       |
| -------------------------------------------------------------------------------- | ---------- |
| 生产构建、迁移、独立冷进程全部 RNG、优化器逐参数对照                             | PASS       |
| B1 640×640、608×864 图/缓存/三尺度与原生 O2O detach                              | PASS       |
| CPU FP32 3 个有效任务更新（3 微批）、CUDA AMP 3 个有效更新（9 微批，跳步不计）   | PASS       |
| EMA、deepcopy、原生 FP16 checkpoint 保存、真实 YOLO 新进程受限重载、eval/predict | PASS       |
| 原生 fuse、重复 fuse、已学 DTR/router/lambda 不复位                              | PASS       |
| 两张复制非空样本 B2/160 的 FP32 评估 schema、原生图、组合诊断清理                | PASS       |
| 配方变更/输出冲突拒绝、真实 OOM 不降 batch、生产对象不被 smoke 污染              | PASS       |
| 服务器 batch32/640、Linux/tmux 实际保留行为、全量 val/test                       | UNVERIFIED |

smoke 使用独立副本、非空合成框、真实 E2ELoss/MuSGD，每精度上限24微批，到3个有效任务更新即停。
相同优化器历史的“当前梯度置零”对照排除仅 weight decay/历史动量造成变化；不为理论零梯度添加辅助损失。
重载以同样 FP16 量化后转 FP32 的 EMA 为参考，逐张量相等；比较同一 O2O 原始输出。
本次 FP32 fuse 最大绝对误差约 7.01e-5，容差 atol=3e-4、rtol=2e-4；重复 fuse 误差0。
未执行完整训练、全量精度评测、导出、参数扫描或公平测速。

## 后续统一评估

`validate.py` 以实际图节点、模块、YAML、Detect 位置、nc/stride核验四组身份：b19/DTR 为24节点，SCE/组合为28节点。
统一 FP32、640、batch32、workers8、conf0.001、iou0.7、max_det300、rect=True、augment=False、quantize=None、end2end=True、native fuse。
seed42、最高 float32 matmul 精度、CUDA matmul TF32=False、cuDNN TF32=True、benchmark=False、deterministic=True，确定性算子警告照常保留。
实际 model/input dtype、原生对 CPU workers 的调整及其他实际配置均记录。

保存 `metrics_exact.json`、`curves.npz`、原生 PR/P/R/F1 图、混淆矩阵和批次预测图。
JSON包含 P/R/F1/AP50/AP75/mAP50–95、各 IoU AP、图像/目标数、checkpoint SHA、代码提交、数据SHA、导入路径、环境、参数量和原生分段耗时。
AP75按真实IoU数组查找；最佳阈值按 `smooth(mean F1,0.1).argmax()` 的相同索引读取并核对原生P/R/F1。
conf0.25/0.50为 `curve_estimate`，不伪造TP/FP/FN；混淆矩阵的不同工作点单独记录。
同卡并发耗时不能当公平论文测速。val用于固定方案/部署阈值，test仅用于固定后的评价，本轮未新增阈值搜索。

诊断默认关闭，最多4张；复用DTR聚合统计和SCE临时钩子，不保留整套激活，结束恢复DTR诊断状态并移除钩子。
历史单模块 test 的 mAP50–95：b19 50.7770%、DTR 51.8056%、SCE 52.2797%；组合仍待实验，不能相加预测。
