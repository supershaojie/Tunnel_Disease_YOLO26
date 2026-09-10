# CCA-Fusion v1 实现与验证

本实验从 `4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6` 创建独立工作树，分支为
`codex/exp-yolo26n-b19-cca-fusion-v1`。仅正常推送 `supershaojie/Tunnel_Disease_YOLO26`，依实施规格不创建 PR。
本文是开发验证记录，尚无正式 CCA 训练、val/test 或涨点结论。

## 模型与公式

原生单类 nano YAML 仅替换第12层为 `[[11,6,10],1,Concat_CCA_Fusion,[]]`。
输入 `[U,L,H]` 通道为 `[256,128,256]`，输出 `cat(U+R,L)` 为384通道。
第11层 nearest、第13层 C3k2、Detect([16,19,22])、stride 8/16/32、reg_max=1 均保留。
第12层位于多个检测尺度路径的上游，不能理解为只影响 P4 检测输出。

四个无偏置1×1投影，r固定16。Wq/Wk/Wv 使用默认 Conv2d 初始化，Wo全零；新增层在 CPU 的
`fork_rng(devices=[])` 内初始化，不改变原生后续层的随机序列。参数先在CPU构建，再随原模型迁移设备。
没有额外门控、归一化层、激活、可学习温度、损失或第三方算子依赖。

Q/K投影在原生 autocast 中执行，L2归一化 eps=1e-6、4倍余弦相似度、softmax与显式
`sum(a*(V_neighbor-V_parent))` 在FP32中计算；D转回V的dtype后送入Wo。
粗尺度 unfold 的九邻域按行优先排列，中心4，父位置为 floor(y/2), floor(x/2)。
无效边界在softmax前置负无穷，因此权重严格为零。四相位查询与粗邻域广播组合，不构造N×N矩阵，
也没有按batch或像素循环、尺寸缓存或forward中新建层。空间常量V产生零差值；这不是对任意背景的保证。
Softmax约束权重，Wo后的特征幅度并无天然上界。

## 实际读取的参考材料

模块包实际路径：`D:/7.21yolo26改/YOLO26缝合.zip`。
SHA256：`a0c197f38e1510291a9a762fc7dc0a8c13694742fff9e3132c50a3ee32ddadcf`，与实施规格一致。
两份文件从ZIP直接完整读取，没有整包覆盖或导入其依赖。

| 包内文件（均在 ultralytics/nn/newsAddmodules） | SHA256                                                             | 实际阅读的类                                                 |
| ---------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------ |
| DPCF_INFFUS2025.py                             | `f4a0606dc51ecb6f80feb65febb68f70f9d097c449144664acc00f0d7da9dd10` | AdaptiveCombiner、conv_block、DPCF                           |
| RLAB_fusion_2025CVPR.py                        | `9e166e61f96cdfe6588679092de06960ad427d127a80c066c0d538b4a07a6d62` | ConvBlock、DSUB、EUB、ResidualBlock、ResidualRB、RLAB_fusion |

DPCF用于参考双输入与通道匹配的组织，不采用标量混合门控、方形插值假设和额外BN主干。
RLAB用于参考跨尺度投影与关系计算组织；实际 `q @ k.transpose` 是N×N，注释“线性注意力”不足以证明线性复杂度。
其 `_proj_if_needed` 在forward路径创建可训练层，本实现不采用。文件会议名不是出版证据。

现场读取 [SAPA](https://arxiv.org/html/2209.12866v2) 的父坐标、局部相似度和聚合公式，及
[On Point Affiliation](https://arxiv.org/html/2307.08198v1) 的点选择扩展讨论。
后者是前者扩展；本实验固定九邻域没有动态核形状的全部能力。
[FreqFusion](https://arxiv.org/abs/2408.12879) 原始论文页面已核对，HTML全文接口此次返回错误；
仅采用实施规格给出的融合一致性与边界偏移动机，不声称完整阅读全文或实现其ALPF/AHPF/offset组合。
这些研究在其他任务的结果不构成本实验效果承诺。

读取并沿用原生 parse_model、预训练load、Conv/C3k2/SPPF/C2PSA/Detect、MuSGD、Trainer warmup与累积、
GradScaler/unscale/clip(10)/step/EMA顺序、Validator one-to-one及fuse源码。
runner/评估/归档基础设施从已有 SGK 分支 `54b87cb` 的实现迁移并审计，只保留配置与生命周期基础设施，
删除其结构判定和机制诊断，替换为CCA专属接口。没有继承SGK模型或权重。

## b19、数据和初始化证据

实际完整读取本地 `E:/ditieyolo26跑结果/8.12离线在线/b19 200e/` 下原始归档展开目录的
`train_run/args.yaml`，全部112项与记录一致。其摘要和results/best摘要存于
[来源证据](evidence/cca_reference_sources.json)。原始启动记录与112项展开配置也已核对一致；记录来源说明保留在
`tools/experiments/b19_launcher_expanded.txt`，不伪称该启动脚本来自原始归档。

| 证据                    | SHA256                                                             |
| ----------------------- | ------------------------------------------------------------------ |
| b19 args.yaml           | `b08b915756bf85c91d3356586a651a37156d71867b84b6e8493d75c9568642b9` |
| b19 results.csv         | `44795068d0ebc687c29ff7b7b8cbf337469546b2d4dd515cb0616c270bdf9507` |
| 原始 yolo26n.pt         | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| b19 best.pt（只供评估） | `d0b2ca5a5d30de9ed002c64c9238b182ddeccce644c055ae5f2dba566878bd5e` |
| 本地数据ZIP内data.yaml  | `1f18760508e9dbf2332cd7102ee9c15e11e08f3d15fc4202c8ee0ed1bb785b12` |

本地从主工作区 `artifacts/datasets/Tunnel_Crack_AugFirst_Diverse5x_RandomSplit_7_2_1_seed42.zip`
逐张读取图像和标签内容，图片数量8414/2404/1202，实例10243/2985/1477。
排序相对路径、图片长度、图片SHA256、标签SHA256后生成的分割指纹保存于
`tools/experiments/b19_dataset_manifest.json`，服务器必须重新计算并相等。完整配置差异见
[逐字段比较](evidence/cca_configuration_comparison.json)，只允许实验身份与同内容路径表达变化。
服务器实际args、数据路径与内容指纹、初始权重、完整配置仍会重新核对，不能用本地预览替代。

## 开发验证结果和边界

环境为 Python 3.11.15 / torch 2.7.1+cu118 / RTX2060 6GiB；与b19的服务器环境不同。
[本地生命周期证据](evidence/cca_local_validation.json) 保存完整检查结果。

| 项目                       | 实测                                                                            |
| -------------------------- | ------------------------------------------------------------------------------- |
| 原生/新增/总参数（未融合） | 2,504,190 / 14,336 / 2,518,526                                                  |
| 原生共有状态               | 708项全部逐tensor相等，包括未加载头部与BN缓冲                                   |
| 原始预训练匹配             | 原生606/708；CCA606/712，额外4项均在model.12                                    |
| 初始完整原始输出           | one-to-many/one-to-one均严格相等，最大误差0                                     |
| CPU参考                    | 非零Wo前向与梯度、角/边/内部、矩形与1×1、零/极小范数、常量差值均通过            |
| 合成检测任务更新           | CPU FP32、CUDA FP32、CUDA AMP；Wo后Wq/Wk/Wv均观察到有限梯度和更新               |
| optimizer观测实现          | 与原生clip/step/EMA结果逐tensor相等；零任务回放排除weight decay伪更新           |
| 生命周期                   | 已更新非零EMA→FP16快照→全新进程FP32重载→fuse；真实Validator合成图前向有非零残差 |
| eval状态                   | 相同模式与矩形切换不修改BN/state_dict                                           |
| 单模块性能                 | RTX2060、FP32、合成batch32、40/20特征，前后向约11.29ms，峰值分配332,548,096字节 |
| 16图诊断与FP32评估入口     | 合成16图实际运行通过；不是裂缝数据评估                                          |

单模块显存包含当时存活张量，不能作为完整训练显存或服务器延迟；THOP也不完整计数自定义归约，因此不将其GFLOPs输出当作完整成本。
本地还执行CUDA batch1的640×640与640×960 FP32/AMP loss检查；证据随最终交付记录。

最终定向测试：11 passed；低层归档烟测实际核验文件大小、SHA256、归档内容和gzip CRC通过（合成fixture，不是正式结果包）。

实际命令（在CCA独立工作树运行）：

```bash
python -m pytest -o addopts='' tests/test_cca_fusion.py -q
python -m tools.experiments.verify_b19_cca_fusion --weights /path/to/original/yolo26n.pt --output artifacts/local_verify
ruff check ultralytics/nn/modules/cca_fusion.py ultralytics/nn/modules/__init__.py ultralytics/nn/tasks.py tools/experiments/run_b19_cca_fusion.py tools/experiments/finish_b19_cca_fusion.py tools/experiments/verify_b19_cca_fusion.py tests/test_cca_fusion.py
bash -n tools/experiments/server_b19_cca_fusion_v1.sh
bash -n tools/experiments/deploy_b19_cca_fusion_v1.sh
```

Ruff固定0.12.12，安装在本工作树临时目录；pip的跨盘临时文件清理报错，但复制出的Ruff可执行文件已实际运行通过检查。
不改变用户原有Python环境。

**待服务器执行**：真实训练数据batch32/640 FP32与AMP完整预检、最多128批真实任务更新、完整val Validator、
正式200轮/原patience训练、best.pt同条件FP32 val/test、固定验证16图诊断、正式归档。
本地Linux flock/tmux行为未运行（Windows Git Bash缺少这些程序）；bash语法已检查。
无涨点结论、不做额外训练/阈值搜索/test选择。

## 实现审计

Deleted: 从迁移基础设施删除SGK的第16层结构判定、SGK参数清单及原机制诊断；新模型配置替换原第12层Concat。
新增代码承担固定CCA公式、可追溯服务器验证和独立实验入口，原生实现中没有可直接替换的CCA算子。
复用原生优化器、Trainer、Validator、EMA、fuse及已存在的b19公共审计流程，没有增加症状跳过开关。
固定batch错误直接失败，任何验证失败均保留attempt；不能降batch、改r、关AMP或删mask放行。
