# DCS-SPPF v2 fuse 预检定向修复

本轮修复的是 FP32 fuse 预检中把最终检测表行号当作候选身份的判据。保留每个原始候选的严格数值审计，
以实际两阶段 top-k 索引证明排列及边界变化。没有修改 DCS 结构、正式 b19 配方、生产 Detect 或融合实现。
本地已经复现判据误报；未获得 AutoDL 原始截图、日志、输入或访问权限，不能据此确认服务器那次故障的根因。

基线分支为 `codex/exp-yolo26n-b19-dcs-sppf-v2`，失败版本为
`8ab1ff436b030ef2642f4ead8f9870fdc0c81f8d`。附件中 `[1,300,6]`、max_abs 约 100.96072、
outside_tolerance=743 的数值仅作为用户提供的文字摘要保存，不标作本地测量。B1 生命周期失败不代表 B32 更新已通过。

## 源码与状态来源

- 原 `state_checks()` 在同一个 eval 模型生成 before 后，对它的 deepcopy 调用 `BaseModel.fuse()`，
  再生成 after；两个 forward 之间没有 optimizer/EMA 更新，也没有 FP16 checkpoint 重载。
- 固定版本 `head.py` 的 eval 返回 `(final, {one2many, one2one})`。one2one 中的 boxes 是回归量、
  scores 是 logits；`_inference()` 先 decode 并 sigmoid，然后 `postprocess()` 进行候选筛选和 gather。
- 第一阶段按每个候选的最大类别分数选 k；第二阶段把第一阶段分数展平后再取 k，并组合原始候选 ID。
  本实验 nc=1，因此第二阶段是 k/k 全排列，没有额外丢弃候选。行号是分数排名，不能用来标识同一网格候选。
- 原生 fuse 折叠 Conv/BN，并按官方实现移除 one2many 的 cv2/cv3；one2one 仍是推理分支。
  DCS 中名为 `fuse` 的子 Conv 与模型的 `fuse()` 方法分别属于不同对象，本轮没有发现属性冲突。
- 新审计在融合前创建两个同源 eval 副本，严格比较 state，输入分别 clone，所有输出立即 detach/clone。
  比较 head 的 nc/max_det/end2end/export/xyxy/stride/anchors/shape cache 等状态，记录全模型训练模式、
  dtype、BN 非参数配置及 DCS theta/rho/eps。未更改 TF32、CUBLAS、CUDA_VISIBLE_DEVICES 或训练 AMP/GradScaler。

指定入口及其 shared runner/verifier、common、两个 DCS 模块、head/tasks/torch_utils 和测试的实际源码哈希
均保存在私有 attempt 收据与公开摘要中。原 common.assert_close_tree 和所有生产模型/配置文件保持原样。

## 新判据与失败证据

1. 首先对完整 one2one 树（原始回归量、logits、全部 feats）及 decode 后全部框和分数进行同候选比较，固定
   `atol=1e-4, rtol=1e-4`。任何失败均阻断，并记录首次超界层；不因 native 同样失败给 v2 豁免。
2. 临时实例观察器调用原生 decode/top-k。仅在一次原生 `get_topk_index()` 内使用 `TorchFunctionMode`
   记录两次真实 top-k 的输入、参数、分数和局部索引，不重新执行 top-k 猜测索引；退出时恢复观察器和 hooks。
3. 每阶段检查全部选中分数等于全局排序前 k 的多重集，验证索引唯一、范围、数量、阶段间 gather，
   及局部 ID 到原始 ID 的组合。仅检查第 k 名不能排除 `[0.9,0.8,0.7,0.7,0.1]` 误选
   `[0.8,0.7,0.7]`；本轮新增该负例。
4. 两侧最终表必须与各自捕获的原始候选、实际索引及类别逐行精确重建一致。类别固定为 0，重复、错映射、
   数量错误、漏高分均失败。原 `fusion.predictions` 的逐行误差继续保存为诊断。
5. 对集合变化，记录每个丢失/新增 ID 的两侧分数、对应截断边界 margin 和实际分数扰动，比较候选并集。
   对每对丢失项 d、新增项 a，要求 `0 <= score_before[d]-score_before[a] <= abs(delta[d])+abs(delta[a])`。
   完整 raw/decode 必须先通过，且两侧选择都已证明合法；捕获缺失或边界无法证明则 UNRESOLVED 并阻断。
6. DCS v2 捕获实际 y0、R、Y，并由同设备实际操作数重建 theta、r0、b、q、r，要求重建 Y 精确一致。
   保存各量 RMS 和前后误差；控制区仍为 FP32、每图 CHW。注入比值上界仍为 0.05+2e-8，
   低精度真实 Y-y0 沿用“注入范数加实测 cast/add 舍入范数”的原审计规则，不能据此放宽框坐标容差。
7. 除零 theta 和原 -0.7 诊断状态，还通过一次真实 native detection loss/MuSGD 的 B2、128×160 更新
   得到非零 theta，检查 fuse、EMA、同进程和跨进程重载的绑定与常量。该小批次仅用于诊断，不是 B32 预检。

每次审计使用独立 `fuse-evidence-*` 目录。`input_rng.pt` 保存 CPU 输入、Python/NumPy/CPU/CUDA RNG；
`source.pt`、`fused_state.pt` 保存可重建状态和非参数配置；`before.pt`、`after.pt` 保存候选、两阶段索引、
最终表及残差操作数；失败时保存 traceback、数值摘要。raw/decode 失败且定位重放完成时保存 `layer_outputs.pt`；
定位重放也失败时保存 `localization_traceback` 和此前取得的部分证据。这些是本机受信任诊断文件，
不自动加载外部未知 pickle，也不作为训练初始化。即使 capture 中途失败，已收集的部分证据仍会落盘。
`state_checks.json` 和 `extra_checks.json` 由各自生命周期函数在 finally 中保存。
数值行中的 `device=cpu` 表示脱离计算图后的证据存储位置；实际推理设备另记为 `input_device/output_device`。

原归档会排除 best/last 之外的所有 `.pt`，因此删除这条过滤，让 canonical provenance 诊断证据随归档保存。
既有 `.attempt.*` 临时训练目录与符号链接过滤保持不变。私有模型、输入及完整日志均未 commit/push。

## 本地结果与边界

环境为 Python 3.11.15、torch 2.7.1+cu118、RTX 2060。原始 yolo26n.pt SHA256：
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`。

- 修改前先保存了失败版本源码指纹、同源输入和模型。首次 CPU/CUDA、B1、128×160、theta=-0.7 检查均通过旧判据。
- 有界 CUDA 输入 seed 100–131 搜索在 seed 105 首次复现：两个排名交换，300 个原候选集合相同。
  全部 420 个 raw/decode 候选符合原容差，最大分数扰动为 `1.6298145055770874e-9`。
  原候选 ID 24、112 在第 256/257 名交换，旧逐行最大坐标误差为 `78.04142761230469`，超界 8 个元素。
  两者融合前分数同为 `0.000939345802180469`，融合后 ID 112 分数升至 `0.0009393462096340954`。
  同输入的原生和非零 v1 控制均通过旧行判据，v2 通过新的严格候选证明；这不能说明所有 v1 输入都不受排列影响。
  完整数值和源码指纹见[公开 JSON 摘要](evidence/dcs_sppf_v2_fuse_validation.json)。
- CPU/CUDA FP32 的 native、v1、v2_zero、v2_updated、v2_diagnostic 共 10 项融合检查通过。
  残差预算和 v2 绑定覆盖各设备的三个 v2 状态；EMA 和跨进程 fused reload 分别检查各设备的 updated 状态。
  没有把最终行位置变动直接当作 PASS。
- AMP 扩展 fuse 等价检查 **8/8 FAILED，全部仍被审计阻断**：CPU BF16 的 native/v1/v2_zero
  raw boxes 最大误差 0.34375，v2_updated 为 0.5；CUDA FP16 四项均为 0.0390625，首差均为 `model.0`。
  这些检查先在 FP32 融合模型，仅对前向使用 autocast。更早一次把融合也放入 autocast 的不正确测试
  已单独保留，不能用于 AMP 前向结论。8 项对应负例测试通过表示“正确拒绝”，不表示 AMP 融合等价通过。
- 原服务器报错检查的是 FP32；其入口继续使用 FP32 fuse gate，并保留原有 AMP 模块、loss/梯度、
  MuSGD/BN gamma 审计。没有新设 AMP 失败豁免，也没有修改正式训练精度策略。

最终回归为 **88 passed**，保留原有 45 项并增加 43 项，其中 8 项明确验证拒绝 AMP raw 超界结果。
完整本地 verifier 的 `passed=true, local_only=true`，FP32/MuSGD B2/640 两次尝试、AMP 六次尝试，
四个 BN gamma 保留 `verified_sub_ulp_task_step` 判据，未退回“参数逐位变化才算更新”的误判。
Ruff、Python syntax、`bash -n`、归档 dry-run 均通过。测试执行源码由 JSON 中的实际哈希绑定，
基线 commit 字段不冒充尚未产生的交付 commit；交付 SHA 以 Git 实际查询及最终回执为准。
服务器原失败输入：**未获得输入或访问**。Native B32 preflight：**NOT RUN**。正式训练：**NOT STARTED**。
本地结果不构成 AutoDL PASS，也不证明精度涨跌。

## 复用与部署

复用 LBI commit `bc090ac0a01dc9c406b40fd45e9b9dcdae0764fe` 的 snapshot、raw/decode、候选并集和边界扰动
框架，核对过其本地测试与报告；其报告未提供服务器修复通过证据。补齐真实两阶段捕获、完整 top-score 多重集、
RNG、DCS 残差量、部分失败收据和 AMP 拒绝测试。没有合并 LBI 分支或模块，也没有编辑 LBI worktree。
LBI 可供后续复用的调用点为该分支 `verify_b19_lbi_fusion.py::_lifecycle_checks`（约第 317 行）
及 `lbi_fuse_audit.py::capture/audit_candidates`；不宣称这些额外缺口已替它修复。

服务器入口保持：

```bash
bash tools/experiments/server_b19_dcs_sppf_v2.sh train
```

本轮未执行该命令。它仍先启动新的独立 preflight 子进程，要求本次收据 commit、recipe、source 完全匹配；
失败非零退出。仅全部既定审计通过后，正式 trainer 才从原始 yolo26n.pt 新建模型、optimizer、EMA、GradScaler
和训练随机状态，不使用诊断/预检权重继续训练，不读取旧 passed.json 放行。

Deleted: 旧最终表行对齐的通过判据、复用代码中仅看 kth 的弱判断与重复排序、canonical `.pt` 的一刀切过滤。
Reused: 固定生产 decode/top-k/fuse、common 数值审计、原 preflight/runner/归档及既有 staged MuSGD 验证。
净增代码用于原流程缺失的候选身份证明、失败可重建证据和负例；只删除旧断言不能满足这些要求。

原理参考：[PyTorch topk 的并列索引说明](https://docs.pytorch.org/docs/2.8/generated/torch.topk.html)及
[数值精度说明](https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html)。这些一般性说明不能替代上述固定源码、
真实候选证据或放宽数值门。
