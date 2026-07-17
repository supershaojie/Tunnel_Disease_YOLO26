# YOLO26n 隧道裂缝检测项目

本目录用于管理基于 Ultralytics YOLO26n 的单类别隧道裂缝检测实验。基线模型确定为 **YOLO26n**，
使用 COCO 预训练权重 `yolo26n.pt`。

## 数据原则

- 原始数据必须始终保持只读，不得在原目录中提取、复制、移动、重命名、删除或修改图片和标签。
- 原始裂缝类别 ID 为 `2`；未来构建的新单类别数据集会将其映射为类别 ID `0`，类别名为 `crack`。
- 数据处理顺序固定为：**审查 → 筛选与映射 → 防泄漏划分 → 可视化验证 → 冒烟训练 → 正式 baseline**。
- 数据集、训练结果以及 `.pt` 权重文件不进入 Git。

## 目录

- `scripts/`：只读审查及后续数据处理脚本。
- `reports/`：审查脚本生成的 JSON、Markdown 和 CSV 报告。
- `configs/`：后续项目专用配置；当前审查阶段保持为空。

## 原始数据审查

必须显式使用 `yolo26` 环境的 Python 解释器：

```powershell
& "D:\miniconda3\envs\yolo26\python.exe" tunnel_project/scripts/01_audit_raw_crack_dataset.py `
    --dataset-root "E:\ditie_dataset\隧道数据集" `
    --target-class-id 2
```

脚本只读取 `<dataset-root>/images`、`<dataset-root>/labels` 和 `data.yaml` 的状态，报告固定写入本项目的
`tunnel_project/reports/`，不会写入原始数据集。
