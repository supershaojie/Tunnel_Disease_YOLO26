# BDI-P3 v1 实施与核验证据

本实验只实现带通细节注入。分支为 `codex/exp-yolo26n-b19-bdi-p3-v1`，基线为
`4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6`。当前部署版本由 `git rev-parse HEAD` 获取；提交后交付的部署脚本另填入真实完整 SHA，避免文档自引用提交哈希。

本地验证已完成（包括开发权重的 16 图诊断计算/可视化 smoke；因没有真实服务器预检来源，正确拒绝生成正式 diagnose 成功记录）；未启动服务器正式训练，也未执行服务器 batch32/640 预检、完整 val/test、训练后诊断或最终结果打包。没有精度提升结论。

## 原始依据

- b19 原始结果包：`E:/ditieyolo26跑结果/8.12离线在线/b19 200e/b19_yolo26n_e200_train_val_test_20260823_224153`。
  读取了 `train_run/args.yaml`、`results.csv`、`environment/git_state.txt`、训练启动日志和数据配置。
- 源码提交与归档 git_state 一致。args SHA256 为
  `b08b915756bf85c91d3356586a651a37156d71867b84b6e8493d75c9568642b9`。
  真实启动日志显示 Python 3.12.3、torch 2.8.0+cu128、Ultralytics 8.4.98、RTX 4090。
- 原始 `yolo26n.pt` 本地实测 SHA256 为
  `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`。
  新训练只从此原始 COCO 权重初始化。b19 best.pt 仅允许后续补评。
- 历史启动命令来自用户原先提供的 `Codex_YOLO26n_b19_SIRSPPF_v1.md` 第 9 节，保存在
  `tools/experiments/b19_launcher_expanded.txt`；已逐项重建并比对归档 args。原 shell 脚本不在归档中，不能声称找到了该脚本。
- 完整配置逐字段报告：[b19_sources_and_config.json](evidence/b19_sources_and_config.json)。本地核实
  train/val/test 图像数 8414/2404/1202，目标数 10243/2985/1477。服务器入口重新验证数据配置与所有图片/标签内容哈希。

阅读并保持了 baseline `DetectionTrainer`、`BaseTrainer` 的初始化/预训练加载、warmup/累积/裁剪、MuSGD、Validator、Detect 的 one-to-many/one-to-one 和 detach、EMA、fuse 路径。没有改动这些原生实现。

## 模块与注册

只替换实验 YAML 的第 15 层：`Concat([-1,4])` → `Concat_BDI_P3([14,4,2])`。
`-1` 原本即第 14 层。第 2 层进入 save 列表；所有后续索引保持不变，Detect 仍为 `[16,19,22]`、stride `[8,16,32]`。

| 节点 | 640 输入下形状 | 用途                         |
| ---- | -------------- | ---------------------------- |
| 2    | 64×160×160     | F2                           |
| 4    | 128×80×80      | L                            |
| 14   | 128×80×80      | S                            |
| 15   | 256×80×80      | cat(S,L+R)                   |
| 16   | 64×80×80       | 原生 C3k2，输入仍为 256 通道 |

源码 `ultralytics/nn/modules/bdi_p3.py`：`pin`、`dw`、`po` 分别为 Pin、DW3、Pout。
固定 r=16，三个卷积均无 bias、BN；DW3 采用普通 zero padding 和 SiLU。只有 `po.weight` 清零。
新增卷积构造使用 `fork_rng(devices=[])`，不消耗后续原生 CPU 参数初始化的随机序列。

`k3/k5` 为非训练 buffers，来自 `[1,2,1]/4`、`[1,4,6,4,1]/16` 的二维外积。
`D=4*(B3(U)-B5(U))`，滤波和相减在显式关闭 autocast 的 FP32 区域完成，之后恢复特征 dtype；
down2 同样 FP32 累加。三个固定滤波均 replicate padding；down2 中心为 0、2、4…，奇数输入按 ceil 下采样。
没有插值对齐；真实尺寸不符抛出带 P2/L/R 形状的错误。

注册只添加模块导出、tasks 导入及单独 parse_model 分支，解析实际 cS/cL/cP2，输出 cS+cL；r 不经过 width multiplier。
保存权重在普通 `YOLO(path)`、AutoBackend、Validator 中直接可用，不依赖 runner 临时 patch。

## 本地验证结果与成本

本机 Python 3.11.15、torch 2.7.1+cu118、RTX 2060，与服务器环境不同。

- 7 项 pytest 回归通过：独立标量边界参考、频带峰值与错误网格、完整图/RNG、与 native 一致的检测匹配、拒绝不完整打包、失败子进程退出码/attempt 保留、CUDA 初始化等价。
- 单算子验证包含核和为 1、常量差分为零、棋盘格内部为零及边界单独比对、31×47→16×24。
  CPU BF16/CUDA FP16 autocast 下通过 dispatcher 记录三个固定卷积实际输入及核均为 FP32。
- 完整 640×640 与 640×960 前向、训练原始输出、推理输出通过。仅对 stride 对齐尺寸承诺完整原生图支持；实测原生及 BDI 的 65×97 均因原生上采样/Concat 网格不兼容而报错，不为任意奇数输入改造主干。
- [weights.json](evidence/weights.json)：708 项共有状态逐键一致；606 项原始权重匹配；candidate 共 713 状态项（增加 3 个参数张量、2 个 buffers）。
- 未融合参数：baseline **2,504,190**，BDI **2,507,406**，新增 **3,216** = 64×16 + 16×9 + 16×128。
- 真标注本地小批测试为 batch2/96、CPU、固定开发 LR，非服务器预检。首步 Pin/DW3 任务梯度为零，第二步三个参数均有非零任务梯度和实际更新。
  首步上游权重衰减造成的移动没有算作学习通过；服务器预检另用独立零当前梯度的 MuSGD 副本排除 decay/旧动量影响。
- 非零分支 EMA、FP16 保存→新 Python 进程重载 FP32、原生融合、AutoBackend 和 one-to-one 分支影响均通过。
  真实 Validator 对两张验证图走过新模块；这不是全量精度评估。
- RTX 2060 batch1 下 640×640、640×960，FP32/AMP 真任务损失和反向有限。640 方形观测峰值 allocated
  325,624,832 / 241,395,712 bytes；含开发脚本当时驻留的模型，不能换算为 batch32 峰值。
- 初始化比较 FP32 atol=1e-6、rtol=1e-5；CUDA AMP atol=1e-3、rtol=1e-3。实际各张量误差记录在
  [structural.json](evidence/structural.json)、[cuda_initial_equivalence.json](evidence/cuda_initial_equivalence.json)。非零融合使用 1e-4/1e-4，并以 native 融合作对照。
- 640、MAC 计 2：学习卷积新增 0.086016 GFLOPs；固定滤波新增 0.029696 GFLOPs。
  PyTorch profiler 实际整网记录 baseline 5.74197708、BDI 5.85891788 GFLOPs，差 0.1169408。
  profiler 含其支持的卷积及加乘，未完整计 SiLU、padding、dtype 转换、访存等，不能称为完整硬件成本。
  CPU 4 线程、batch1、未融合 FP32，3 次预热/10 次计时均值 104.51→122.78 ms。服务器完整延迟、显存和正式性能待测。

复现本地检查：

```bash
python -m pytest tests/test_bdi_p3_v1.py -q -o addopts=
python -m tools.experiments.verify_b19_bdi_p3_v1 --local \
    --data ../../datasets/Tunnel_Crack_AugFirst_Diverse5x_RandomSplit_7_2_1_seed42/data.yaml \
    --weight ../../yolo26n.pt --output artifacts/new_local_check
```

第二条为本地嵌套 worktree 布局，输出目录必须不存在。验证器会复制两张图及标签到临时验证目录，避免写入原始数据缓存。
模型检查与完整本地记录见 [local_validation.json](evidence/local_validation.json)、[reload_check.json](evidence/reload_check.json)。

## 配方与服务器生命周期

服务器实际配方由原 b19 args 读取，仅变更 model/project/name、由同一原始文件解析出的 pretrained 路径以及重建 save_dir；cfg 已展开。
保持 epochs=200、patience=60、batch=32、imgsz=640、MuSGD、seed42、AMP、workers8、nbs64、全部损失/在线增强。
包括 cos_lr=True、warmup_epochs=3、warmup_bias_lr=0.1、mosaic=1、mixup=.135、cutmix=.03，不能默默关闭。
训练中验证仍使用原生 batch 倍增规则；独立 val/test 使用 batch32。

复用 BCI worktree 提交 `8a56b9d4d42d0a855a37f4149e3e3d1a3f175cc2` 的通用 b19 生命周期（来自既有 AMP 预检修复），并对照 PKC 修复后的预检实现。
实际 runner 为 `run_b19_bdi_p3_v1.py`，收尾为 `finish_b19_bdi_p3_v1.py`，共用 `b19_common.py`。

`train` 自动运行独立预检进程；最多 128 个真实 batch，采用原生任务损失、warmup、累积、GradScaler、全模型裁剪和 MuSGD。
观测每个新增参数的 pre/post-clip 梯度、真实 LR、任务引起的实际更新、AMP 跳步和 P2 可达性。
清除仅用于副本的观测 hook，防止把 FP32 对照的梯度误写成正式 AMP 证据。
停止条件还要求 EMA 经原生 FP16 量化后仍有非零投影且影响实际 one-to-one 输出；若早期更新被量化抹去，继续在原 128 batch 窗口内观察，而不提前结束或改初始化。
预检完成后父进程再启动全新 Python 进程，seed42/原始权重/optimizer/EMA 从零重建；不继承预检参数或 RNG。
阻止原生 OOM 自动减 batch 的触发点是实验 Trainer 的重试计数写入口；不修改公共训练循环，不提供跳过预检/降低 batch 的路径。

`test` 在 val-only best.pt 上进行全量 FP32 val/test，导出逐 IoU AP、PR/F1/P/R 曲线、混淆矩阵、预测、置信度/TP 匹配统计及完整 args。
当前源码用 `quantize=None` 表达 half=False。P/R 默认是 native 最佳 F1 操作点；另在 val 上选 Precision≥.85 的最高 Recall 置信度边界并冻结到 test，记录实际达到的 Precision/FPPI。
若 val 上无可行阈值，记录 unavailable，不在 test 选阈值。已有 b19 best 可用时自动按同口径补评，无需重训。

`diagnose` 使用固定前 16 张验证图、同权重 FP32/no_grad、conf=.25、原生 end2end 后处理，临时 hook 关闭 Pout 输出。
记录 U/D/V3/R 范数、有限比例、D/U、R/L（epsilon=1e-12），IoU .5/.75 的 TP/FP/FN、补回/丢失 GT、FP 变化和共同匹配目标的 IoU 变化。
保存原图上 GT/预测编号的 off/on 对照；不是重训消融，D 不是裂缝概率。

`package` 核对训练 SHA、原始权重/数据/源码指纹、同一来源的成功预检及绑定阶段记录后才打包；包含 best/last、配置、完整结果、源代码快照和依赖版本。
不能用失败 attempt 的文件补齐。生成 tar.gz、manifest、外部 sha256，读取归档逐文件校验并执行 `sha256sum -c`。
当前无最终训练结果包；只在服务器阶段实际成功后输出完整路径、大小和哈希。

## 理论来源与适用边界

参考包实际路径 `D:/7.21yolo26改/YOLO26缝合.zip`，SHA256
`a0c197f38e1510291a9a762fc7dc0a8c13694742fff9e3132c50a3ee32ddadcf`，与要求一致。
读取包内 `YOLO26缝合/ultralytics/nn/newsAddmodules/HFP_SDP_2025AAAI.py` 的
DctSpatialInteraction、DctChannelInteraction、HFP；包内文件与本地解压源逐字节一致，源文件摘要见 `tools/experiments/bdi_sources.json`。

[HS-FPN 官方论文](https://ojs.aaai.org/index.php/AAAI/article/view/32740)与
[原始方法](https://arxiv.org/html/2412.10116v2)使用频率信息构造空间/通道注意力。本实现只借鉴频率选择动机，采用局部带通数值直接注入，未引入 torch_dct、全局高通掩码、GroupNorm、sigmoid 或 SDP。
[ICML 2019](https://proceedings.mlr.press/v97/zhang19a.html)提供平滑后降采样的思想来源；本模块不能宣称严格抗混叠。

忽略边界，q=cos(wx/2)^2 cos(wy/2)^2，H=4q(1−q)。因此 DC 和最高频端为零、峰值为 1；不是增强全部高频，也不显式预测方向。
经过 DW3/SiLU 后整条非线性支路不再有同一个线性响应。可能利用 P2 留存而深层削弱的细节，无法恢复输入缩放丢失的信息，亦可能压掉有用纹理。
细裂缝纹理宽度不是检测框大小。效果须由正式实验验证。

MPDF-P3 修正上采样语义相位，BDI 增加 P2 输入并修正 L；SGK-P3 在第 16 层生成语义引导动态核，BDI 在第 15 层使用固定滤波；DCR-Strip 利用方向条形/法向差异，BDI 不预测方向。
本实现未增加 P2 检测头；梯度经 P2 影响后续骨干及多尺度，不能声称其他分支完全不受影响。

Deleted: 替换实验图原第 15 层 Concat 定义；复用脚本时删除 BCI 注意力诊断、另一实验模块注册、对全部旧 attempt 的打包枚举和仅收集 best 的权重过滤。
本功能须新增固定公式及独立审计入口；基线没有可删除的对应 BDI 实现。没有复制其他实验的模型或已训练权重。

参考文档生成已运行；Windows 默认 GBK 与生成器反斜杠导航问题在本次产物中修正，仅保留新 API 页及一个导航项，没有扩大修改生成器。新增 Python 与脚本按 Ruff/Prettier 固定版本及 bash 语法检查验证。
