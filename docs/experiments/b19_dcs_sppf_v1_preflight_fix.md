# DCS-SPPF v1：gated branch preflight 修复报告

修复对象：`codex/exp-yolo26n-b19-dcs-sppf-v1`，修复前 HEAD 为 `36f7336960aa84cf5e7171c14e9e783661f35c63`；原 b19 Base 为 `4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6`。

**模型代码未修改。功能代码只修改 verifier，另增回归测试及审计文档。正式训练：NOT STARTED。**

本地复现、实现及验证已完成。服务器实际 64 个 B32 batch 的逐步日志和 SSH 连接尚未提供，因此本报告不能签发服务器 native B32 PASS，也不能给出该次服务器运行的真实梯度 norm 或首次更新索引。下列数值明确来自本地合成验证。

## 根因与证据边界

原 verifier 并未要求所有新增参数在第一个 step 更新；它累计观察最多 64 批。真正的判定缺陷是把“任务梯度非零且 FP32 parameter delta 严格大于零”作为所有参数统一的必要条件，没有识别 warmup 下 BN gamma 的舍入损失，也没有读取裁剪后的梯度及任务对 optimizer 状态的真实贡献。

复现使用未修改的模型、原始 `yolo26n.pt`、原生 detection loss、AMP 和 MuSGD，在 RTX 2060 上使用本地合成 B2/640 输入，并重放原 b19 B32 的 warmup 调度：8414 张训练图对应 264 批，warmup 为 792 批；BN/weight/muon 组学习率从 0 起，bias 从 0.1 起。它不是原数据集的真实 B32 训练。

64 次复现恰好留下四个 BN gamma 未发生可观测 tensor 更新，与用户报告的失败集合一致。theta 在 batch 1 已有有限非零梯度，但该批因其他梯度溢出未接受 optimizer step；batch 4 theta 真正更新，alpha 离开 0；batch 5 全部 12 个支路参数获得有限非零任务梯度，conv/bias 出现任务导致的可观测更新。不存在 theta 始终为 0 的现象。

实际初始化是 theta=0、四个 gamma 全为 1、四个 beta 全为 0；fuse 使用正常初始化，没有 gamma=0 的双重死支路。独立模块回归同时验证 step 0 精确 identity 和后续所有参数的更新。

在 1 附近，FP32 向下相邻间隔为 `5.960464477539063e-8`，向上为 `1.1920928955078125e-7`；相应半间隔为 `2.9802322387695312e-8` 和 `5.960464477539063e-8`。本地 gamma 的单步请求更新只有约 `1e-10`，真实 FP32 加法仍舍入为 1。beta 从 0 起且 warmup 学习率不同，能表示更小的更新；卷积还走 Muon+SGD 路径，不能直接用它们的变化尺度要求 gamma。

用户提供的失败集合与旧 verifier 代码共同说明，其余九个参数曾进入旧审计的通过集合；但该错误消息本身没有四个 gamma 的梯度记录，无法单凭错误字符串断言服务器此次也已证实舍入原因。已证实的是上述 verifier 缺陷及同集合的本地复现；服务器原因的最后闭环仍需新预检日志。

## 全部新增参数的本地 64 步结果

索引均从 0 开始。“首个 delta”包括 decay，因此卷积的 batch 4 不能作为支路学习证据。“首个有效任务 step”已排除零任务梯度重放。所有参数均为 FP32、optimizer 唯一注册、grad 从未为 None，全部被接受 step 的梯度有限。

| 参数                 | 首个有限非零梯度 | 首个 tensor delta | 首个有效任务 step | 接受 step 中最大 postclip grad max_abs | 最大 tensor delta |
| -------------------- | ---------------: | ----------------: | ----------------: | -------------------------------------: | ----------------: |
| theta                |                1 |                 4 |                 4 |                            2.425802e-2 |       2.517382e-5 |
| refine.0.conv.weight |                5 |                 4 |                 5 |                            1.233853e-6 |       7.873029e-5 |
| refine.0.bn.weight   |                5 |                无 |       5，舍入证明 |                            4.847145e-7 |                 0 |
| refine.0.bn.bias     |                5 |                 5 |                 5 |                            1.600123e-7 |       2.563053e-8 |
| refine.1.conv.weight |                5 |                 4 |                 5 |                            1.409660e-6 |       8.755922e-5 |
| refine.1.bn.weight   |                5 |                无 |       5，舍入证明 |                            3.807374e-7 |                 0 |
| refine.1.bn.bias     |                5 |                 5 |                 5 |                            1.797553e-7 |       3.085208e-8 |
| refine.2.conv.weight |                5 |                 4 |                 5 |                            1.348705e-6 |       7.431209e-5 |
| refine.2.bn.weight   |                5 |                无 |       5，舍入证明 |                            3.836030e-7 |                 0 |
| refine.2.bn.bias     |                5 |                 5 |                 5 |                            1.931838e-7 |       3.193241e-8 |
| fuse.conv.weight     |                5 |                 4 |                 5 |                            4.146292e-6 |       4.209206e-5 |
| fuse.bn.weight       |                5 |                无 |       5，舍入证明 |                            4.827756e-7 |                 0 |
| fuse.bn.bias         |                5 |                 5 |                 5 |                           1.802842e-10 |      3.629627e-11 |

theta 所在组为 weight，四个卷积为 muon，四个 beta 为 bias。四个 gamma 均在 index=3 的 bn 组，weight_decay=0、use_muon=False、nesterov=True，momentum 依原 warmup 调度变化。

以下为 batch 63 的实际 postclip 梯度及参数记录，64 步逐批明细见证据 JSON；四个 gamma 的 before/after **每个元素都为 1**。

| gamma              | 元素数 | grad None? / finite? |   grad norm | grad max_abs | step 前/后 | delta max_abs | 请求更新 max_abs |
| ------------------ | -----: | -------------------- | ----------: | -----------: | ---------- | ------------: | ---------------: |
| refine.0.bn.weight |    128 | False / True         | 1.102200e-6 |  3.439553e-7 | 1 / 1      |             0 |     4.174289e-10 |
| refine.1.bn.weight |    128 | False / True         | 1.107890e-6 |  3.491295e-7 | 1 / 1      |             0 |     4.415308e-10 |
| refine.2.bn.weight |    128 | False / True         | 1.146790e-6 |  3.073386e-7 | 1 / 1      |             0 |     3.538218e-10 |
| fuse.bn.weight     |    256 | False / True         | 1.541893e-6 |  3.454331e-7 | 1 / 1      |             0 |     4.274589e-10 |

该批 BN lr=`0.0007954545454545455`；64 步内四者最大请求更新分别为 `5.834735e-10`、`4.654531e-10`、`4.862004e-10`、`7.287202e-10`。逐元素请求更新最多只占相应半 ULP 的 2.446%，低于舍入边界，全部原生 FP32 重放与实际参数及 optimizer 状态精确一致。

这些结果证明梯度链及当前 optimizer 的有效任务贡献，不能宣称 gamma 在这 64 步已发生物理 tensor 更新，也不能保证未来训练何时发生可观测变化。

## 修复后的判定

复用了此前 CCA/NDP 的有限梯度、optimizer 成员、真实 step hook、零任务梯度重放思路；参考了 SIR 对零初始化支路延迟梯度的处理。没有继承 SIR 独立 probe 的 unit-scale AMP 设置，当前原生 GradScaler 保持不变。

1. 在实际 optimizer pre-hook 读取完成 unscale/clip 的梯度；原 trainer 仍负责 unscale、clip、MuSGD、scaler、zero_grad 和 EMA。
2. 每个参数必须唯一加入 optimizer、requires_grad=True、dtype=FP32；接受的任务梯度必须有限且非零。
3. 以同类型原生 MuSGD、同参数/组/状态分别重放实际梯度和零任务梯度；实际参数及 momentum 必须与前者逐 tensor 精确一致。可观测更新还必须不同于零任务梯度结果。
4. 仅对四个正常初始化的 BN gamma，允许记录 `verified_sub_ulp_task_step`：正学习率、实际任务梯度和 momentum 贡献非零、请求更新及任务分量非零、每个元素严格低于其有方向半 ULP、FP64 诊断相加再转回 FP32 恰等于更新前、真实 FP32 重放完全吻合。任何一项未证明均不通过。这是明确的数值证明，没有添加 tolerance。
5. theta 必须先发生真实任务更新并使 alpha 离开 0；支路有效任务 step 必须来自后续 batch，alpha=0 时有限支路任务梯度必须为零。全部 13 个参数保留，不允许 decay 或旧 momentum 单独充当学习证据。
6. 服务器观察窗口仍为 64 个原生 B32 batch；本次按审计要求记录完整 64 批，即使较早证据齐全也继续记录，不扩大窗口。输出 `native_steps.json` 和 `native_gradient_summary.json`，未覆盖全部参数仍触发 assertion。

Deleted: 替换旧的 pre-clip max_abs + exact-delta 单一判定，以及仅凭 scale 变化推断 step 的逻辑。新增行数用于用户要求的逐参数数值证据、原生/零任务重放和阶段状态审计；单纯删除或迁移旧条件无法证明 FP32 舍入和任务贡献。没有给模型添加压制失败的条件。

## 验证与约束

| 检查                                                | 结果                                                                                             |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| CPU/CUDA/AMP module zero-init                       | 精确一致，max_abs=mean_abs=0，原严格容差保留                                                     |
| 独立模块正常幅度后续更新                            | 原有 3 步全部 13 参数真实梯度及 tensor 更新检查保留并通过                                        |
| B32 模块 staged-update 回归                         | 64 步；batch 0 identity，theta 梯度非零；batch 1 unlock；batch 2 支路梯度及有效更新；无缺失参数  |
| 故障反例                                            | 断梯度、全零梯度、lr=0、缺失 optimizer 成员、空 step、gamma=0、仅旧 momentum、丢失状态更新均拒绝 |
| shared-weight / pretrained                          | 708 个共享 tensor 精确一致，606 compatible tensors 全部正确迁移                                  |
| FP32 detection loss + MuSGD                         | 本地 B2/640，全部 13 个参数阶段审计通过                                                          |
| AMP detection loss + MuSGD                          | 本地 B2/640，5 次原生 overflow skip 后通过；跳步时所有参数不变                                   |
| save/reload                                         | 733 个 state tensor、固定输出及新进程重载精确一致                                                |
| 同输入修复前后 64 步对照                            | loss、theta、alpha、scale/step 接受轨迹、13 参数 FP32 delta maxima 逐步精确一致                  |
| tests                                               | 39 passed，含 package dry-run、SHA-256/gzip CRC/源码 archive 检查                                |
| Ruff / format / compileall / bash -n / diff --check | PASS                                                                                             |
| 模型与正式配置                                      | 没有变化；仅 layer 9 的原 DCS 结构，theta=0，alpha=0.10\*tanh(theta)，其余主干/neck/Detect 不动  |
| 参数量 / GFLOPs                                     | 原生 2,504,190 / 5.771776；DCS 2,607,231 / 5.8555392，未变化                                     |
| 正式训练                                            | **NOT STARTED**                                                                                  |
| AutoDL 真正 B32 复验                                | **未执行，等待连接及该次日志；不得标记 PASS**                                                    |

epochs=200、imgsz=640、batch=32、workers=8、MuSGD、原 lr/warmup/augmentation、seed=42、原数据集和 b19 `yolo26n.pt` 均保持不变。未改 optimizer、norm 或数值后端，未跳过预检，未创建 PR。

[审计证据 JSON](evidence/dcs_sppf_v1_staged_update_audit.json) 包含每个参数的初值/分组/首次梯度与更新、四个 gamma 的 64 批记录、momentum/ULP 证明、完整本地验证摘要以及原始日志的 SHA-256。`artifacts/dcs_after_fix_warmup.json` 留有未压缩字段的全部 13 参数逐步明细；JSON 内记录其哈希。旧实施报告中的训练效果仍无正式实验结论。

安全的服务器复验入口已经存在，无需修改 launcher：

```bash
bash tools/experiments/server_b19_dcs_sppf_v1.sh preflight
```

该 stage 完成预检后直接返回。正式入口仍为 `bash tools/experiments/server_b19_dcs_sppf_v1.sh train`，本次没有执行。真实服务器批次索引及数值必须从新 attempt 的 `native_steps.json` / `native_gradient_summary.json` 补齐，不能套用本地表格。
