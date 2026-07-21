# YOLO26n隧道裂缝正式基线——200轮

## 实验定位

本实验是后续YOLO26n结构改进实验的正式baseline。
150轮实验仅作为前期参考，不作为模块消融的正式对照。

## 统一设置

- 初始化：官方COCO预训练yolo26n.pt
- 数据：Tunnel_Crack_AugFirst_5x
- 轮数：200
- 图像尺寸：640
- Batch size：32
- 随机种子：42
- YOLO在线增强：关闭
- 默认Albumentations：通过Python API传入空列表关闭
- 模型选择：验证集best.pt
- 测试集：完成模型选择后只评估一次

后续模块实验必须保持数据、优化器、学习率、轮数、图像尺寸、
batch、随机种子和增强策略一致，只改变模型结构。
