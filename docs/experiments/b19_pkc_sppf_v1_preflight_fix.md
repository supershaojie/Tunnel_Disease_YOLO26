# PKC-SPPF v1 预检修复记录（2026-09-09）

## 证据与结论

本轮依据用户的 `Codex_PKC_SPPF_v1_preflight_fix.md`，完整读取该文件和实际上传的
`pkc_preflight_debug_20260909_124618.tar.gz` 内 checks、args、优化器分组、环境、日志及实际源码。
压缩包 SHA256 为 `78e5c41ca95e79678e207e4abedbc7de3feccdf69d96c592af5c5a2ea90276fc`。
包内 runner、finish、Shell、模型 YAML、PKC 模块及原测试文件逐字节等于失败提交
`6c80a68fb20bbd1360c6cf037619fef067c2b747` 的 Git blob；Windows 工作文件的 CRLF 不用于误判源码差异。
没有把修复文档中另一 UUID 文件名当作已读取文件。本轮未引入历史模块包内任何其他结构。

失败报告位于包内
`runs/detect/yolo26n_b19_pkc_sppf_v1_preflight/bc227b30147d051bb809/checks.json`。
已证实：batch 0–7 全模型梯度非有限，scale 从 65536 降到 256，8 次都未执行 optimizer step；
batch 8–15 仅有 8 次成功更新。batch 8 上游任务梯度为零，卷积因 decay 变化不能算任务更新。
batch 9–15 四项 BN gamma 梯度有限非零，精确参数比较仍未变化；它们确实在实际 optimizer 的 BN 分组中，
不是无梯度，也不是已发现漏注册。旧 16 批预算耗尽时缺少四项 gamma 的实际更新证据，因而断言失败。

尚未证实：真实 RTX4090 上 gamma 不动是否完全由 clipped gradient、warmup LR、动量和 FP32 舍入共同解释；
128 批内是否一定能满足全部条件。旧报告缺少这些逐步数据，不能从裁剪前 L2 范数认定唯一根因。
修复后的真实数据/CUDA 预检、完整 Validator、正式训练及最终 test/diagnose 均为 **待服务器验证**。

## 修改范围与判据

只修改实验 runner，新增观察器回归测试并更新交付文档。`ultralytics/` 全部原生及 PKC 源码、模型 YAML、
`b19_reference.json`、finish 和 Shell 入口保持失败提交版本。仍是 PKC-SPPF v1，只有第 9 层 SPPF 替换，
零投影、BN gamma=1、r=32、5/9/13、参数量和 forward 均不变。
完整 b19 配方继续逐项继承：batch32、imgsz640、epochs200、patience60、MuSGD、seed42、AMP=True、原数据和原权重。

Deleted: 删除旧 16 批循环和仅有梯度范数/changed 布尔值的诊断，替换为固定预算的观察器。
复用原生 Trainer、GradScaler、clip、MuSGD、EMA、保存重载和 Validator；新增行用于必须保存的逐步诊断和故障验证，
只删除或移动旧日志无法提供此前不存在的 clipped gradient、真实 step、动量和副本更新证据。

- 默认观察预算 128 个真实 batch32，全部 13 个新增参数满足条件即提前结束，不自动追加预算。
  后续独立 FP32 batch32 反向及完整 Validator 仍单独执行，不属于 AMP 观察窗口。
- `attempted_steps`、`overflow_skips`、`successful_steps` 分开计数；真正 step 由 optimizer post hook 观测，
  不依赖 `scaler.step()` 的返回值。batch 从 0 编号，first\_\*\_step 使用从 1 开始的 attempted step 编号。
- 每个新增参数必须在成功 step 中同时具有有限非零 task gradient 和精确参数变化。
  裁剪前后梯度都记录，零任务梯度的 decay 或动量变化不会计入通过。
- 身份匹配要求参数在 optimizer 中恰好注册一次，且 float32、requires_grad=True。
  每次记录真实 group/index、lr/initial_lr、momentum、weight_decay、use_muon、nesterov。
- 记录全模型有限性、clip 前总范数及系数、各新增参数 clip 前后 norm/max_abs、精确 changed_elements/max_abs_delta，
  四项 gamma 前后范围/dtype、动量缓存 norm/max_abs，首次有限非零梯度和首次有效更新步骤、当前缺项。
- gamma 分析克隆实际参数、实际 clipped gradient 和实际 optimizer state；用仓库原生 MuSGD 的非 Muon 分组
  在副本上执行一步，要求预测结果与实际结果精确相等。另记录 FP32 update 对应的未舍入增量、
  `torch.nextafter` 方向间距和低于半间距的非零提议元素数。不会修改正式参数、优化器状态或超参数。
- 预算耗尽、setup、forward/backward、更新或后续检查失败均保存阶段、异常、预算、计数、未满足项及原因，非零退出。
  每次入口也有唯一 `*-invocation-*/checks.json`，子进程失败报告被链接并汇总计数；setup 前错误也留下记录。
  无通过凭证的重试使用新目录，旧失败文件保留。
- 原有 fingerprint 同时包含当前 commit、源码、配置、数据、权重和环境；此次 runner 源码及 commit 变化会生成新指纹，
  不复用旧 SHA 凭证。`train` 自动独立子进程预检，随后新建原生 Trainer，从原 seed/原始权重重新开始正式训练。

## 与原生循环的核对

依据本提交 `ultralytics/engine/trainer.py` 的 `_do_train` 和 `optimizer_step`：
首次 zero*grad 后进入首 epoch，scheduler.step，`_model_train`；warmup 仅当 `ni <= nw` 生效，
其中 `nw=max(round(warmup_epochs*nb),100)`，warmup_epochs=0 时为 -1。
累积数仍由原生插值及 round 计算，当前 group LR/momentum 使用原生插值。
达到 accumulation 后执行 unscale → clip_grad_norm*(max_norm=10) → scaler.step → scaler.update → zero_grad → EMA。
`last_opt_step` 在尝试更新时前移，包括 overflow 跳步；EMA 也沿用原生每次尝试更新的行为。

旧预检未调用首 epoch scheduler，并无条件执行 warmup 插值、每批 model.train。
现改为上述原生边界。b19 的 start_epoch=0、warmup_epochs=3，128 批位于首 epoch 内；
不会更改 warmup 时长、scale 初值、clip 阈值、正式训练优化器或正式训练代码。
副本分析严格使用 `ultralytics/optim/muon.py` 非 Muon 分组的 momentum/Nesterov 运算，
不错误乘入仅用于混合分组的 `optimizer.sgd` 权重。

## 本地验证

Windows / Python 3.11.15 / torch 2.7.1+cu118，数值回归均在 CPU，使用真实原生 MuSGD 和 CPU GradScaler。
原 8 项测试通过；新增 8 项观察器测试通过（初次合并运行 14 项通过，补齐入口和凭证回归后复跑新增 8 项通过）。

- 8 次初始 overflow 计为 8 次 skip、0 次 success；随后真实 step 才满足更新条件。
- 有限非零 gamma 梯度在 16 批内无精确变化时按预算失败，原因不是断梯度；128 批窗口内观察到真实变化后提前结束。
- 漏注册、重复注册、断梯度直接失败；持续零 LR 更新和仅 decay 变化均耗尽 128 批并失败。
- 观察器与原生 `BaseTrainer.optimizer_step` 在连续 12 次尝试（含 8 次 overflow）后的模型、optimizer state、
  GradScaler、EMA 状态精确一致，副本分析未污染 live state。
- setup 故障、入口结构故障写出失败 checks；两次入口故障保留两个独立记录，无 passed.json。
- 凭证回归确认同一源码复用有效记录时保留真实计数，源码及 commit 改变后重新预检，并保留旧凭证。
- 原初始化、公式、所有共有状态、真实预训练加载、任务梯度、BN、EMA、重载、fuse、诊断和固定 batch OOM 测试继续通过。
- Python 编译、Ruff format/check、原 Shell 入口及文档命令的 Bash 语法检查通过。
- Git diff 确认上述受保护路径无变化；AST 比对确认 runner 原有配方解析、AuditedTrainer、初始化审计、
  保存重载/fuse、运行环境检查及指纹等 22 个原函数/类未变化。

复核命令及服务器精确 SHA 更新、唯一 tmux 重试、进度/退出查询、test、diagnose、package 指令见
[实验交付文档](b19_pkc_sppf_v1.md)。服务器启动一次 `train` 即可，无需再手动串联预检。
