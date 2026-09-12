# DCS-SPPF v2 服务器 fuse 精度审计修复

本修复基于部署提交 `b086f4dc069cb9cd37ea90f68c3adc18f11987b2` 的真实故障包。只修改独立验证和重放流程；DCS 模型、每图 5% 残差预算、b19 配方、正式训练/test 精度、固定服务器入口均不变。未启动正式训练。

旧的 [候选审计报告](b19_dcs_sppf_v2_fuse_audit.md) 是此前本机结果，不能替代本次服务器证据。最新机器可读摘要位于 [precision validation](evidence/dcs_sppf_v2_precision_validation.json)。

## 证据来源与安全读取

| 证据                                                    | SHA256                                                             |
| ------------------------------------------------------- | ------------------------------------------------------------------ |
| 原始故障包 `dcs_v2_fuse_failure_20260912_233851.tar.gz` | `a2574627db34dacd8ba1ec415598a2119c115ce2bef0ba90594d30245db57870` |
| 精简包 `DCS_SPPF_v2_Server_Precision_Evidence.zip`      | `2acbe0981939c12cff156788114db2162b54e315a3b133601c5cdcfec922dd8b` |

原包文件名与指令中的上传名不同，但 SHA256 完全一致。原包先安全解包、保留失败张量，再做修复前对照；精简包路径补充后继续核验，没有重建随机 fixture 替代现场。精简包 34 个 manifest 条目全部通过，25 个重叠文件与原包字节一致。两份精简 fixture 各有 11 个输入/参数/服务器输出张量及 BN eps 与原包精确相同。15 份可对应源码均与原故障 audit 哈希匹配，并在修改前核对了当前分支源码。

所有外来 state/tensor 使用 `map_location="cpu", weights_only=True`。未放宽安全加载、未载入含 NumPy RNG 的 `input_rng.pt`，固定输入来自 `before.pt/input_after`。重建模型只使用本地固定 native/v1/v2 架构，严格核验保存的架构签名、完整 state、BN/Detect/残差属性及类别映射。包内源码未覆盖工作树；精简 `replay_first_block.py` 经实际阅读后只在独立诊断进程执行。原权重、输入和完整日志均保留在 ignored artifacts，不提交 GitHub。

## 原始服务器失败与根因边界

原始环境：RTX 4090、PyTorch `2.8.0+cu128`、CUDA 12.8、cuDNN 91002、Python 3.12.3。五组均使用 B1×3×128×160、420 个候选、FP32 tensor、autocast 关闭。matmul precision 为 highest、matmul TF32 关闭，但 **cuDNN TF32 允许**。CPU snapshot 记录的比较设备不代表原始前向设备。

| 状态          | 服务器 raw boxes max_abs | 超差/1680 | decoded boxes max_abs | 排名变化 | 入选集合变化      |
| ------------- | -----------------------: | --------: | --------------------: | -------: | ----------------- |
| native        |             0.0189909935 |      1061 |           0.270515442 |       49 | 无                |
| v1            |             0.0189909935 |      1061 |           0.270515442 |       49 | 无                |
| v2_zero       |             0.0189909935 |      1061 |           0.270515442 |       49 | 无                |
| v2_updated    |             0.0413997173 |      1168 |           0.643272400 |       65 | 丢弃 83，加入 130 |
| v2_diagnostic |             0.0238127708 |      1099 |           0.381004333 |       44 | 无                |

五组原始结果继续保留 **FAIL**，首个已捕获超差块均为原生 `model.2.cv1`，早于 DCS 第 9 层。native/v1/v2_zero 各自融合前、各自融合后的 raw、decoded、final 完全一致。因此本次不能归因为 top-k 排序，也不是无反向 fuse 阶段缺 GradScaler。updated 来自真实全模型 MuSGD 更新，其前缀权重也已改变；审计不要求 updated 与 native 共享前缀。

已确认的工程问题是：审计用 FP32 tensor/autocast 关闭代表严格 FP32 数学等价，却继承了允许 TF32 的卷积策略。同现场输入、原 state 和保存折叠权重的 CPU/RTX 2060 对照均通过原门槛，完整重新融合 state 与保存 state 精确一致，支持卷积内部精度解释。**原 RTX 4090 上 TF32 开关的因果对照尚未执行，不能声称已实机证明 TF32 是唯一原因。**

## 首差层同源对照

每条路线使用同一份保存的 `model.1` 输入；unfused 为 Conv→BN→SiLU，fused 为保存的 folded Conv→SiLU。输出形状为 `[1,32,32,40]`，所有本地结果均为 0/40960 超差；`atol=rtol=1e-4` 不变。

| fixture    | 参考证据 CPU / torch 2.10 | 本机 CPU / torch 2.7.1 | RTX 2060 允许 TF32 | RTX 2060 禁用 TF32 |
| ---------- | ------------------------: | ---------------------: | -----------------: | -----------------: |
| native     |              4.5776367e-5 |           4.5776367e-5 |       3.8146973e-5 |       3.8146973e-5 |
| v2_updated |              6.1035156e-5 |           5.3405762e-5 |       6.1035156e-5 |       6.1035156e-5 |

表内为 max_abs。参考列是精简包已有结果；其余三列是本次实际执行。RTX 2060 compute capability 为 7.5，许可开关相同结果不能证明执行了 TF32，也不能代替 RTX 4090 对照。

## 修复内容与门的定义

`strict_fp32_equivalence` 只围住独立复制模型的融合、前向与比较：输入/参数为 FP32、autocast 关闭、cuDNN TF32 关闭、矩阵乘法 highest。作用域记录进入、执行、恢复状态，`finally` 恢复成组 matmul 设置、cuDNN、autocast 和 Python/NumPy/CPU/全部 CUDA RNG。正式 runner 仍以独立子进程完成 preflight，成功后从原始 yolo26n.pt 和 b19 原配方构建全新的训练器、优化器、EMA、scaler 和 RNG。

五种状态全部运行三种 profile，各有独立证据目录和 PASS/FAIL，汇总仅使用明确的 `strict_gate_passed`：

- **严格门必需**：五组 strict 的所有 raw boxes/logits/feats、全部 decoded 候选、完整两阶段 top-k、最终候选并集以及 DCS 残差对照通过。
- **所有 profile 的单模型不变量必需**：有限值、形状/dtype、完整 top-k 排名/唯一 ID/真实 gather、输出重建、原预算和舍入界限、state/config/source 一致性以及恢复失败均阻断。
- **有限跨路线数值误差**：native/AMP 仍写 FAIL、张量和 traceback，但其容差失败是明确的诊断项。比较现场发出专用异常，外层按类型记录诊断；不解析报错字符串或依赖“native 也失败”来豁免 v2。strict 同类失败仍阻断。
- **真实训练安全门仍必需**：既有 MuSGD 分阶段更新、AMP/GradScaler、sub-ULP gamma 证明、B32/640 真实数据检测预检均保留。数学等价通过不替代任何训练检查。

另修正审计的最终 gather 重建：保留原生 FP32 类别列引起的 dtype 提升，删除强制转为低精度的转换；实际值继续零容差比较。模型推理代码未改。

保留 DCS 的真实 `y0/R/theta/r0/b/q/r/Y/reconstructed` 捕获、FP32 控制器、逐图范数、参考支路 stopgrad、精确重建和舍入界限。原场景 v2_zero 残差比为 0；updated 约 7.109e-7；diagnostic 约 1.05153%，均在原 5% 预算内。alpha、三尺度、BN、theta 零初始化及 rho/eps 均未改。

## 验证与复用流程

最终五组整网对照均使用各自现场 source state 和同一保存输入 `[1,3,128,160]`，没有另选随机输入。本机为 Python 3.11.15、PyTorch `2.7.1+cu118`、RTX 2060（capability 7.5）。

| 状态          | CPU native / strict | CPU strict raw max_abs | RTX 2060 native / strict | GPU strict raw max_abs | CPU BF16 / GPU FP16 fuse 诊断 |
| ------------- | ------------------- | ---------------------: | ------------------------ | ---------------------: | ----------------------------- |
| native        | PASS / PASS         |           2.0980835e-5 | PASS / PASS              |           1.3828278e-5 | FAIL / FAIL                   |
| v1            | PASS / PASS         |           2.0980835e-5 | PASS / PASS              |           1.3828278e-5 | FAIL / FAIL                   |
| v2_zero       | PASS / PASS         |           2.0980835e-5 | PASS / PASS              |           1.3828278e-5 | FAIL / FAIL                   |
| v2_updated    | PASS / PASS         |           5.3405762e-5 | PASS / PASS              |           2.5987625e-5 | FAIL / FAIL                   |
| v2_diagnostic | PASS / PASS         |           2.7179718e-5 | PASS / PASS              |           1.7166138e-5 | FAIL / FAIL                   |

strict 所有 raw/feats/decoded/最终候选并集均为 0 超差，两阶段 top-k 和每个单模型精确重建全部通过。判断仍是原 `atol + rtol * abs(reference)`；例如 CPU updated decoded boxes 的 max_abs 为 0.0008544921875，但按原相对加绝对容差判据为 0 超差。所有 profile 使用同源完整 state、BN/Detect 属性和相同折叠结果；各组重新融合的完整 state 与服务器保存 state **逐元素精确一致**。30 次整网审计的 backend/autocast/RNG 均恢复，正常及故意异常退出的回归也通过。

119 个唯一回归用例已验证：整批运行 118 通过，另 1 个旧测试只因 traceback 仍期待旧 `AssertionError` 名称失败，更新为专用数值异常名称后单独重测通过（3.12 秒）；没有删测试或降低拒绝要求。原 88 项及 8 个 AMP 数值拒绝用例全部保留。新增 Conv 权重/BN running variance 污染、NaN/shape/dtype/keys、后续普通状态断言及原生诊断 RuntimeError 均能阻断所需门。

完整 `--local` 预检 PASS：708 个共享 tensor 精确一致、606 个形状兼容预训练 tensor 成功加载；CPU/CUDA 各 84 个控制器 case；B2/640 FP32 和 AMP/GradScaler 对全部 13 个新增参数取得梯度支持的有效更新，`missing_effective_parameters=[]`，保留 gamma 的 sub-ULP 证明；零初始化、BN 生命周期、非零更新 fixture、EMA、未融合及融合 snapshot 的新进程 reload 均通过。Ruff、格式及现有服务器入口的 Bash 语法检查通过。

最终结果由配套 JSON 记录，包含 CPU/RTX 2060 五组同源 profile、实际误差、首差层、8 个原有 AMP 数值拒绝测试和新增损坏/恢复负例。真实 B32/640：**NOT RUN**；RTX 4090 事故现场复测：**NOT RUN**；正式训练：**NOT STARTED**。meta B32 形状与 B2 合成检测不能写成真实 B32 PASS。

本地执行方式（已有授权证据解包后，输出必须使用全新目录）：

```powershell
$env:DCS_SERVER_EVIDENCE = (Resolve-Path artifacts/server_precision_evidence.fudbw72y).Path
python -c "import torch,pytest; torch.set_num_threads(4); raise SystemExit(pytest.main(['-q','-o','addopts=','tests/test_dcs_sppf_v1.py','tests/test_dcs_sppf_v2.py','tests/test_dcs_sppf_fuse.py','tests/test_dcs_sppf_precision.py']))"
python -u tools/experiments/replay_b19_dcs_sppf_v2_precision.py --evidence-root $env:DCS_SERVER_EVIDENCE --device cpu --threads 4 --output artifacts/precision_replay_cpu_new
python -u tools/experiments/replay_b19_dcs_sppf_v2_precision.py --evidence-root $env:DCS_SERVER_EVIDENCE --device cuda:0 --threads 4 --output artifacts/precision_replay_cuda_new
```

服务器入口仍是 `bash tools/experiments/server_b19_dcs_sppf_v2.sh train`；本次未执行。重放工具只做张量验证，没有 train 调用。既有服务器入口及更新审计未被替换。

## 修改范围与审查说明

| 文件                                                              | 目的                                                          |
| ----------------------------------------------------------------- | ------------------------------------------------------------- |
| `tools/experiments/b19_detect_fuse_audit.py`                      | 局部精度、明确 profile 门、单模型不变量、精确原生 gather 重建 |
| `tools/experiments/b19_common.py`                                 | 在原数值比较位置选择有限超界异常类型，默认语义及容差不变      |
| `tools/experiments/verify_b19_dcs_sppf_v2.py`                     | 原五组接入三 profile，保留更新/EMA/reload/控制器              |
| `tools/experiments/verify_b19_dcs_sppf.py`                        | 既有阶段增加开始/结束日志；计算及正式训练参数不变             |
| `tools/experiments/replay_b19_dcs_sppf_v2_precision.py`           | 安全、可复用的真实现场重放与硬件/执行状态记录                 |
| `tests/test_dcs_sppf_fuse.py`、`tests/test_dcs_sppf_precision.py` | 保留全部旧测试，新增精度、同源 fixture、负例与恢复验证        |
| 本报告、去敏 JSON                                                 | 分离参考证据、本机结果和未执行检查                            |

Deleted: 删除旧隐式精度的五组单次循环、最终类别列的错误低精度转换；复用已有张量统计、树比较、完整候选/残差审计、训练和归档流程。新增行多于删除行的原因：原流程没有可重放的精度状态作用域和同源现场工具；仅删除或移动代码不能记录并验证这些必要的对照与异常恢复证据。

未解决项仅是 RTX 4090 同现场精度开关复测及服务器真实 B32/640 检查尚未执行。二者由服务器实际结果决定，本文不提前授予通过结论。
