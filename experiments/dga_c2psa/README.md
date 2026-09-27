# b19 + DGA-C2PSA

正式训练状态：**NOT_STARTED**。这是尚未验证检测精度收益的独立实验。

代码基线：`4a1b167dee2b0c7353d9d7ead2911383d1ebf1d6`，分支：`exp/dga-c2psa-b19`。
只将 YOLO26n 第 10 层换为 `C2PSA_DGA`；原生 `C2PSA`、neck、SPPF、Detect、损失、分配器和 O2M/O2O
detach/调度均未修改。实际构建：1 个 PSABlock，Attention 输入 128 通道、2 头、Q/K 每头 32 维、V 每头
64 维；640 输入对应 20×20 网格。

## 实现与初始化

`ultralytics/nn/modules/dga.py` 继承原生 Attention 的 qkv/pe/proj 构造以及 PSABlock/C2PSA 的 forward。
各原生组件只构造一次；新增预测器在 CPU RNG 隔离范围内初始化，末层 weight/bias 全零，无额外 gate。
原生参数路径不变。独立 YAML 和 parser 注册支持 Trainer 重建及新进程 checkpoint 加载。

预测器读取 Attention 输入，按 `[u0,v0,u1,v1,...]` 输出。固定径向映射为
`r=sqrt(u²+v²+1e-6)`，`(a,b)=0.8*tanh(r)/r*(u,v)`。query 系数决定
`bias[i,j]=-a[i]*(dx²-dy²)-2*b[i]*dx*dy`，x 为列、y 为行，位移统一除以 `max(H-1,W-1,1)`。
仅径向映射、位移及 bias 在局部 FP32 中计算；预测器遵循 AMP，bias 转回 logits dtype 后相加。

锁定源码实际使用 `(q*self.scale).transpose(-2,-1) @ k`，本实现保留该计算顺序。
PE 仍作用于原始 V。D 是迹为零的方向校正矩阵，通常不是正定矩阵；本实现没有添加 I 项或各向同性距离惩罚，
不能解释成完整高斯核、正定距离或对所有远距离位置的抑制。原 Attention 已有卷积 PE。

## 配置来源与入口

`b19_recipe.yaml` 来自本地历史归档
`b19_yolo26n_e200_train_val_test_20260823_224153/train_run/args.yaml`，仅移除需替换的运行字段。
原始 args 文件 SHA256：`b08b915756bf85c91d3356586a651a37156d71867b84b6e8493d75c9568642b9`；
已对照同归档的 b19 启动日志。其余默认值来自锁定源码，包含 `nbs=64`、`cls_remap=True`、`cls_pw=0`。
原生 O2M 权重从 0.8 线性衰减至 0.1，O2O 为其补值，沿用原生 epoch 调度。

模型：`experiments/dga_c2psa/yolo26n-dga.yaml`。
入口：`experiments/dga_c2psa/train.py`；支持 `--data --weights --project --name --device --baseline-args --dry-run`。
可选 `--baseline-args` 严格核对完整历史配方。正式训练固定 200 epochs、batch 32、640、seed 42、MuSGD、AMP，
`resume=False`。原始权重 SHA256 必须为
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`。

正式 Trainer 的 `get_model` 在同一起始 RNG 下加载原始权重，对全部共同参数/buffers 逐项核验，包含未迁移的
nc=1 检测头。102 个形状不兼容张量属于原生 nc80→nc1 分类头适配，意外原生缺口为 0。
`--dry-run` 仅使用该构建路径检查配置/权重，不创建正式输出目录，不训练。
正式入口记录真实依赖，核对已确认服务器核心版本及 Albumentations 缺席状态；不自动安装或修改共享环境。
原生 AMP 检查仍执行；其失败或 OOM 导致的 batch 缩减会在创建训练管线时终止实验，避免继续使用不同配方。

## 有限验证

`validate.py` 使用独立合成输入及非空标签，所有更新/BN/EMA 状态与正式模型隔离。
2026-09-27 本地 Python 3.11.15、torch 2.7.1+cu118、CUDA runtime 11.8、RTX 2060 上完成：

- 独立标量/矩阵参考检查 2×3、1×4、4×1、1×1 网格；bias 最大绝对误差 2.98e-8。
- Trainer 原生 `setup_model` 从 YAML/原始权重重建；708 个共同状态张量逐项相等，构建后 RNG 相等。
- 640×640 和 96×160 单张原始输出与 native 初始最大误差均为 0。
- 原生损失和 MuSGD 完成 3 次实际 optimizer.step，合成 batch=2、96×128、积累=1，AMP 跳步=0。
  梯度在裁剪/weight decay 前测量：末层首步非零；前两层首步为零，随后两步均非零。
  3 个新增卷积 weight 按原生规则进入 Muon 组，末层 bias 进入 bias 组，各恰好一次、学习率 0.01。
- 独立非零副本 bias 最大绝对值 0.1292，原始输出变化 0.03547；deepcopy 误差 0。
  EMA 更新后原始输出误差 2.44e-5 以内；原生格式 FP16 EMA checkpoint 重载与对应量化参考误差 0，
  新进程加载通过。融合后原始 O2O 误差 2.13e-5 以内，预测器参数保留且实际执行，原生 BN 正常融合。

| 模型        | 融合前参数 | 原生融合后参数 |
| ----------- | ---------: | -------------: |
| native nc=1 |  2,504,190 |      2,375,031 |
| DGA nc=1    |  2,506,450 |      2,377,291 |

新增 2260 参数，640 输入时预测器卷积约 0.9024M MACs；这不含几何 bias 的
`batch*heads*N²` 级逐元素计算和存储。常规卷积统计不代表完整开销，也不代表线性注意力。
原生 end-to-end 融合会删除 O2M 推理分支，因此融合一致性比较使用原始 O2O 输出。

未进行：正式训练、完整 val/test、真实检测效果评估、多 seed、参数搜索、batch32 显存压力测试、服务器执行或
全部导出后端验证。本地核心环境与服务器不同，且本地装有 Albumentations；上述合成测试未使用该增强，
不据此宣称完整环境已经复刻或模型超过 b19。原数据划分保持 train/val/test=8414/2404/1202，未用 test 调参。
