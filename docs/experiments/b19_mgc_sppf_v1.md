# MGC-SPPF v1：固定设计与审计

本实验为 YOLO26n 隧道裂缝检测提供一个待验证的形态间隙上下文假设，仅替换 layer 9。
正式训练尚未启动；本地测试不是 AutoDL B32 preflight 证据，不预先声称 AP/Recall 改善或统计显著性。

| 身份     | 固定值                                                  |
| -------- | ------------------------------------------------------- |
| Branch   | `codex/exp-yolo26n-b19-mgc-sppf-v1`                     |
| RUN      | `yolo26n_b19_mgc_sppf_v1`                               |
| 基线源码 | `4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6`              |
| 基线 RUN | `b19_y26n_diverse5x_e200_i640_b32_musgd_b8b9hybrid_s42` |
| 模型     | `ultralytics/cfg/models/26/yolo26n-mgc-sppf-v1.yaml`    |
| 类       | `MGC_SPPF`                                              |

## 来源

实际读取用户模块 ZIP，SHA256 为
`a0c197f38e1510291a9a762fc7dc0a8c13694742fff9e3132c50a3ee32ddadcf`。
清单中三个指定文件是 `ultralytics/nn/modules/block.py`、
`ultralytics/nn/newsAddmodules/CGhalfConv_2025ESWA.py`、`ultralytics/cfg/models/26/yolo26.yaml`。
ZIP 的旧编码目录名保留在本地审计，解码后的显示名亦记录。原生 SPPF AST 与 canonical b19 完全一致。
CGhalfConv 只提供低成本部分处理的思路；没有复制分组卷积、C3k2 替换、安装脚本或包内 YAML。

脱敏摘要为 `tools/experiments/mgc_reference_audit.json`；实际路径、完整清单和私有归档信息仅在忽略的
`artifacts/source_audit/` 中保存。模块 ZIP 不提交。继承代码采用仓库 AGPL-3.0；CGhalfConv 文件没有独立许可头，
未复制其实现，不据文件名补造论文信息。

已从用户 CCA v2 结果归档重新读取 `run/provenance/b19_original_args.yaml`，112 项与保存的 b19 记录一致。
原始 args 的 SHA256 为 `b08b915756bf85c91d3356586a651a37156d71867b84b6e8493d75c9568642b9`。
原始 `yolo26n.pt` 的 SHA256 为 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`。
历史启动命令是已保存的展开命令记录；原始 launcher shell 文件不在 b19 归档中，这一限制保留在 provenance。

## 数学定义

继承 native SPPF 构造，保留 `cv1/cv2/m/n/add` 路径与旧 checkpoint 的 `getattr` fallback。
`cv1` 为原生 `Conv(c1, c1//2, 1, 1, act=False)`，只计算一次，主支路和新增支路共享同一个 `z`，
不 detach、不就地修改。原生 maxpool 级数 `n=3` 不参与 YAML repeats 的深度缩放。

```python
u = gap_in(z)
d = max_pool2d(u, k, stride=1, padding=k // 2)
closing = -max_pool2d(-d, k, stride=1, padding=k // 2)
gap = closing - u
r = gap_out(cat([gap3, gap5], dim=1))
y = y_native + r.to(dtype=y_native.dtype)
```

固定 `rank=16`、`k=3/5`，普通无 bias 点卷积；不增加 BN、激活、归一化或门控。
MaxPool 采用隐式负无穷 padding 与 `ceil_mode=False`，不手工零填充、不降采样。
闭运算的两次邻域操作分别可能依赖原输入 5×5、9×9 的范围，与 native SPPF 的 5/9/13 分支不同。
并列极值使用 PyTorch 原生次梯度，不保证不同平台的梯度选择顺序完全相同。

Closing 和 black-hat 是已有形态学算子，见 [OpenCV 官方定义](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)。
padding 语义见 [PyTorch 2.8 MaxPool2d](https://docs.pytorch.org/docs/2.8/generated/torch.nn.MaxPool2d.html)。
深层特征的 gap 不是裂缝概率、像素恢复或去噪保证。人工缺口与孤立峰值测试只支持其特定构造。

## 参数与初始化

| 新增 tensor              | nano 实际 shape | 参数  |
| ------------------------ | --------------- | ----- |
| `model.9.gap_in.weight`  | `[16,128,1,1]`  | 2,048 |
| `model.9.gap_out.weight` | `[256,32,1,1]`  | 8,192 |

共 10,240 个新增参数。640 正方输入时 layer 9 是 `[B,256,20,20]`。
两层点卷积增加约 0.008192 GFLOPs（multiply-add 算两次），不包括池化比较或访存。
THOP 数字按相同口径保存；真实速度需要服务器实测。

`gap_in` 保留 `nn.Conv2d` 默认初始化。仅新建模块中的 `gap_out.weight` 置零。
仅这两个新增卷积的构造使用 `torch.random.fork_rng(devices=[])`；不扰乱后续原生层的 RNG 流。
加载、EMA、reload、fuse、predict 不重新置零。只保证初始化时退化为原生输出，不保证训练后残差幅度上界。

模型从原始 COCO `yolo26n.pt` 经 native `DetectionModel.load` 和 `cls_remap` 初始化；
nc=1 不兼容的 Detect tensor 由同 seed 的 canonical native 模型初始化，逐 tensor 对齐并解释。
参数和 BN buffers 的 shared count、exact matches、源 checkpoint 形状失配都实际统计，不硬编码历史 708/606 为通过。
禁止以 b19 训练后 best 或其他创新权重初始化。

## 唯一拓扑差异

```yaml
- [-1, 1, MGC_SPPF, [1024, 5, 3, True, 16]]
```

parser 仅将 `MGC_SPPF` 加入通道类，不加入 `repeat_modules`。rank 不经 width multiplier 缩小。
native YAML 完全不改；layer 10 C2PSA、12/15/18/21 Concat、23 Detect 的索引和输入保持原样。
`scale=n`、`end2end`、`reg_max=1` 和 nc 适配保留。审计输出完整 24 层模型表。
未合并其他创新模块/YAML/loss。canonical b19 不包含其他实验 YAML；它们留在各自 worktree，
不将其实现合入本实验。额外原生 YOLO11 构建验证公共 parser 接口。
另在只读子进程中对 DCS、FDV、SICR、CCA v2 原 parser 内存副本应用同一个 MGC 注册增量，
逐一验证原生及各自创新 YAML 的模型结构和全部初始化状态不变；不改它们的源码文件。

## 分层验证

`tests/test_mgc_sppf_v1.py` 覆盖 CPU B1/B2、20×20/13×17、train/eval 原 BN 更新、常量正负零、
有效邻域边界参考、人工缺点细线、独立峰值、原生 Detect detach、权重 reload、EMA、参数/拓扑和归档。
Windows 无 symlink 权限时明确跳过真链接测试；不算作 Linux 链接通过。

`verify_b19_mgc_sppf.py --local` 使用指定原始权重、完整 args 与同一数据 manifest 做独立本地验证：

- CPU 同算子零残差模块和整网输出要求 `atol=rtol=0`，报告 max/mean abs、dtype、shape；检查 one2many/one2one。
- 模块两步普通 SGD 只证明代数关系；另用 B2/640 合成检测 batch、原生 detection loss 和 native MuSGD 验证更新。
- `gap_out=0` 时上游任务梯度为零；输出投影先实际离开零，后续上游出现非零有限梯度和任务更新。
- native MuSGD 精确重放与零任务梯度反事实分离任务更新、weight decay 和旧 momentum；新增权重必须在 optimizer 中且只出现一次。
- FP32 state/checkpoint 往返和新进程输出严格比较。FP16 checkpoint 采用同量化 snapshot 对照，独立保存误差影响。
- EMA、CPU fuse、AutoBackend 部署保留非零投影。fuse raw one2one 使用原 verifier 的 `1e-4/1e-4`；
  decoded 坐标按 stride32 使用 `atol=32e-4`，先比较全部 anchors，避免 top-k 的近似并列排序影响。
- CUDA autocast 检查预先采用 `atol=1e-6, rtol=1e-5` 整网判据，模块共享算子比较仍要求严格一致；
  不改 CUDA、TF32、CUBLAS 配置来通过。

服务器 preflight 运行真正 `BaseTrainer._do_train`，最多 64 个原生 B32/640 train batch。
通过 subclass 的只读 step 观察和 callbacks 记录，不复制 warmup 循环、不替换 loss 或优化器。
参数分组、LR、WD、累积、loss scaling、unscale、clip、step 和 EMA 由 native trainer 执行。
合法 step pre/post hooks 读取裁剪后的未缩放梯度，不重复 unscale。
记录 name/shape/dtype/requires_grad、group/LR/WD、grad norm/max、step delta、首次梯度和更新 batch、optimizer buffers。
累积未请求 step、AMP overflow 跳步和有效 step 分开记录；未观察到任务更新则预算到达时失败，不扩大预算。
本实验两 tensor 没有 BN gamma；删除旧 DCS 的专用 sub-ULP BN 判据。若 Muon 权重不能得到可观测任务更新，
保持失败，不用 decay 或浮点舍入推测替代证据。

完整 AMP 语义参考 [PyTorch 2.8 AMP examples](https://docs.pytorch.org/docs/2.8/notes/amp_examples.html)。
本地 CPU/RTX2060 或 synthetic B2 通过均不代表 native AutoDL B32/AMP 已通过。

## 固定配方、诊断和归档

完整 b19 args 为唯一来源，112 项逐键审计。只允许模型、实验名、输出与等价来源路径改变。
epochs=200、patience=60、B32、640、workers8、MuSGD、seed42、AMP 与全部增强原样保留。
不重分数据、不重增强、不升级依赖、不改 MuSGD/loss/原生训练代码。
沿用的 AugFirst Diverse5x RandomSplit 来源和局限保留，不据本实验宣称不存在划分相关性。

诊断固定前 16 个按路径排序的 validation 图像，记录 ID、顺序和 SHA。
只在独立 eval 进程临时挂 hooks，并移除全部 hooks/cache；训练循环默认没有诊断 hooks。
JSON/CSV 记录两投影 norms、u/g3/g5 的 mean/std/min/max/zero_fraction、r/native 的 RMS/L2、
每图残差 L2 比、两尺度能量分数，以及宽度4的边界区和内部统计。对照前后全部参数/buffer hash。

test 在 val 选择 best 后固定评估。`baseline_comparison/mgc_val`、`mgc_test`、`b19_test` 明确分开。
报告 P/R/mAP50/mAP50-95/AP75、图像目标数、权重 hash、split 和实际评估配置。
AP75 使用 `all_ap[:,5]`，不推算。未获得的 DCS/FDV 正式检测结果继续视为未知。

正式 package 要求 best/last、args/results/curves、val/test/b19 comparison、诊断、preflight 和 provenance。
用 `git archive HEAD` 包含全部 tracked 源码及工具依赖，附 baseline 到 HEAD diff、源码 manifest。
只归档指定 RUN；不夹带历史 runs 或 datasets。临时 attempt、符号链接和缺失旧 console 都明确记录。
硬链接以正规文件物化。回读每个 tar member 校验内容 hash，校验 gzip CRC，再写压缩包 SHA256 sidecar。

## 工具来源与删除审计

详见 `tools/experiments/mgc_utility_sources.json`：

- DCS `d4a3b83940bdb306d008ede322b1e45bcd08cafb`：b19 provenance、严格配方、native 初始化、MuSGD 重放。
- SICR `d6e90097fd58e48f80e28503aab7d282ea58326f`：同量化重载、缓存/fuse、finish/archive、只读诊断流程。
- CCA v2 `05c79c6f2beb9a5e38d728838916f7729a0ab3b4`：归档内实际核验的 `.attempt.*` server entry。

Deleted: 移植版本中的 DCS/SICR 门控、refine/BN 专属断言、手写 warmup 预检循环、固定历史 tensor 通过数、
旧实验诊断公式和精度目标阈值；正式包中的瞬态链接改为显式排除记录。canonical b19 没有可删除的 MGC 实现，
故独立特性和必需审计工具增加文件；复用 native SPPF/parser/trainer 和成熟中性工具，未复制其他创新架构。

## 本地执行示例

```bash
python -m pytest tests/test_mgc_sppf_v1.py --override-ini=addopts= -q
python tools/experiments/verify_b19_mgc_sppf.py --local --local-device cpu \
    --baseline-root /path/to/original/local/base --baseline-args /path/to/b19/args.yaml \
    --output artifacts/local_cpu
# 可用的本地 CUDA，仍然只是 local tests：
python tools/experiments/verify_b19_mgc_sppf.py --local --local-device cuda:0 \
    --baseline-root /path/to/original/local/base --baseline-args /path/to/b19/args.yaml \
    --output artifacts/local_cuda
```

失败的 JSON/traceback 和日志保留；具体本次命令、通过范围、限制及 Git SHA 在外置实施报告中记录，
避免把自身最终 commit SHA 提交进同一 commit 的递归问题。
