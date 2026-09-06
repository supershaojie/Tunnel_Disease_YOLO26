# SIR-SPPF v1 保存重载预检修复

本轮依据用户明确授权的 `Codex_YOLO26n_SIRSPPF_reload_fix.md` 和服务器报错截图执行。
部署旧提交为 `28c4bc9c85b924bddddaecb094211b8f333772a7`；修复仍在同一 SIR v1 分支。
没有访问服务器失败目录，没有执行服务器 batch=32 预检或启动正式训练。

## 原因与实测证据

截图中失败发生于 `fuse()` 之前的完整 raw 输出零容差比较，后续 `CalledProcessError` 是同一次失败传播。
旧父进程沿用入口的 CPU 线程设置，重载子进程单独固定为 4；旧测试的 autouse fixture 也设为 4，掩盖了入口差异。
本地直接使用原始 `yolo26n.pt`、原生 Trainer 构造 SIR，父进程设为 1 后，未修改的旧重载函数复现同一失败链。
这确认了代码中可复现的计算条件缺陷；截图本身未提供服务器误差，不能将下表数字当成服务器实测。

本地环境为 Windows / Python 3.11.15 / torch 2.7.1+cu118 / Ultralytics 8.4.98。
保存口径为同一 FP16 EMA 快照恢复到 CPU FP32；原生 `load_checkpoint` 确实调用 `.float().eval()`，
并更新 args、pt_path、task、inplace 属性，未重新构造 SIR router。参数与缓冲区全部 714 项键/形状/dtype/数值精确一致。

| 比较条件                             | 解码输出最大绝对差 | 解码输出平均绝对差 | 不等元素数 | 完整 raw 结果    |
| ------------------------------------ | ------------------ | ------------------ | ---------- | ---------------- |
| 旧 1 线程参考 vs 同一加载模型 4 线程 | 5.340576171875e-5  | 3.491471587463e-6  | 337        | 严格比较失败     |
| 同一加载模型统一 1 线程              | 0                  | 0                  | 0          | 所有张量严格相等 |
| 修复后真正的新 Python 子进程         | 0                  | 0                  | 0          | 所有张量严格相等 |

旧线程不一致时，解码参考最大幅值为 203.2974853515625，最大误差/整体幅值为 2.626976e-7；
但 `atol=1e-6, rtol=1e-5` 仍有 18 个解码元素超限。因此没有采用这一候选容差，更没有继续放宽直到通过。
统一计算条件后继续要求 `atol=rtol=0`。输入由父进程保存一次，两端使用同一个 `[1,3,64,96]` CPU FP32 张量；
本地该次输入字节 SHA-256 为 `67e7a5c183638e099a318aebc063df25e750cd75120fe379172538eec42c6349`。

[PyTorch 2.8 数值精度说明](https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html)指出等价浮点计算不保证逐位一致，
本次判断依据仍是实际状态与受控比较证据。服务器版本的
[MKLDNN context 源码](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/backends/mkldnn/__init__.py)也已核对。

## 修正及检查边界

- 删除仅子进程固定 4 线程的设置和重复内联重载程序，改由同一个 `reload_context` 管理两端计算。
  仅在保存重载范围内统一 CPU FP32、eval/no_grad、CPU autocast 关闭、1 线程、确定性、MKLDNN 和最高 FP32 矩阵精度；
  finally 恢复调用方线程、后端、确定性设置，context 恢复 RNG、autocast 和梯度模式。正式训练条件不变。
- 序列化一个明确的 FP16 EMA 快照，再将该内存快照转回 FP32 形成独立参考；没有用重载同一文件两次替代原模型对照。
  完整 state_dict 保存在 `reload_reference.pt`，包括 router、cv1/cv2 和所有 BN 状态。
  重载前向前后均精确比较状态，并比较模块类型、n/add、end2end/inplace、BN 属性及 Detect 缓存的预期变化。
  SIR 修正系数仍由原模块代码中的 0.5 定义，模块文件和配方快照没有改动。
- `assert_close_tree` 保持原调用点的容差，补充具体路径、shape/dtype/device、有限性、最大/平均绝对差、
  整体参考尺度及超限计数；不依赖接近零元素的相对误差判断。
  未融合完整 raw 仍严格相等；融合保留的 one2one 和解码输出继续使用原先 `1e-4/1e-4`，不改原生 one2many 移除规则。

4 项针对性测试通过（其余 9 项本轮未重复运行）：
原始权重的新进程保存/重载/融合/预测；2 张真实训练图的 3 次可丢弃 AMP 更新后 EMA 重载；
异常后调用方条件恢复；超限与非有限 raw 差异拒绝。
第一项同时验证故意破坏 router 权重或 BN 的 `num_batches_tracked` 时精确状态检查必须失败，并检查原模型与 RNG 未被改变。
两条真实模型路径各有 714 项状态精确一致，所有未融合 raw 张量最大误差均为 0；
融合解码最大绝对差为 9.1552734375e-5，原融合容差下超限元素为 0。
Ruff 与交付 Bash 命令语法检查通过。未训练 epoch，未使用 held-out test 图，也未重跑 b19 或 DCR v2。

本地证据位于 `runs/sir_development/reload_fix_before/diagnosis.json` 和 `reload_fix_verified/`。
正式服务器预检会在其自身目录写入 `reload_reference_conditions.json`、`reload_check.json`、`reload_process.log`；
失败仍返回非零状态，不写 `passed.json`。源码指纹发生变化，旧通过记录不能复用；旧失败目录保持原样。

## 服务器已有 detached worktree

使用交付回复中的固定提交更新命令。它从原仓库 fetch，只在指定 SIR worktree 干净、没有活动 SIR Python/包装器进程且独占锁可用时，
执行 `git switch --detach <已验证的新提交>`。不在 detached HEAD 上盲目 pull，不切换 DCR v2，不删除或覆盖预检历史。
旧 console 日志和退出状态先复制到新的 history 目录，再由本次真实进程更新状态。

如果同名 tmux 存在，命令检查所有 pane 为无子进程的空闲 shell，并在该 session 内创建新的预检/训练窗口；不 kill-session。
会话不存在时创建 `y26_sir_v1`。先显式 preflight，通过才进入 train，正式模型仍从原预训练和 seed=42 重新初始化。
正式输出目录若已存在则停止并列出内容，需根据实际记录查明是否已有训练内容，不能据截图猜测后删除或 resume。
创建窗口不代表训练启动；以新的预检通过记录、epoch/batch/loss 日志及真实退出状态为准。
