# 有限开发验证

全部产物为 **SMOKE**，位于开发工作区 `.dev/validation-delivery/`，未提交权重、数据或运行产物。
正式服务器训练 **NOT_STARTED**，未连接服务器、未执行真实数据全量 test、未开展参数扫描。

| 检查           | 真实观察                                                                                                                                |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| 官方源码与模型 | v7.0 / `915bbf294bb74c859f0b41f1c23bc395014ea679`；原始 nc=1 C3/SPPF/anchor Detect                                                      |
| 官方权重       | 实际下载4,062,133 bytes；SHA256 `4f180cf23ba0717ada0badd6c685026d73d48f184d00fc159c2641284b2ac0a3`；自己计算，GitHub未公布digest        |
| 配置/入口      | 所有 --help、指定 AdamW/lr0/batch/mixup/cutmix/close_mosaic 覆盖通过；未知键/概率/冲突/重复 YAML 拒绝；dry-run 无正式目录               |
| 预训练迁移     | 343/349 状态项；输出卷积六个 shape 不同的键未迁移；未 fuse 初始化                                                                       |
| CPU FP32       | 两次原生 ComputeLoss + 有效 optimizer.step；box/objectness loss有限且非零；保存完整checkpoint                                           |
| CUDA AMP       | RTX2060；原生 check_amp通过；两次有效 optimizer.step，以 optimizer post-hook计数；无AMP溢出步伪计数                                     |
| 增强           | 固定非空框强制 Mosaic/MixUp/CutMix；CutMix像素及标签确实改变；框有效；worker spawn通过                                                  |
| 关闭日程       | epoch15/20关闭 Mosaic/CopyPaste/MixUp/CutMix；其他变换保留；none确定性且不消耗Python随机状态                                            |
| 冻结指标       | 正确/错误/重复预测、空GT/空预测；TP2/FP3/FN1、P0.4/R2/3/F10.5；与B19参考全部返回数组精确一致；部分重叠样本区分AP50/AP75；阈值后重新匹配 |
| 训练后独立重载 | 正式evaluate入口新进程重载SMOKE best；模型/输入均torch.float32；fuse/NMS/完整NPZ/GT/曲线/图/val阈值/test冻结阈值通过                    |
| 参数/计算量    | nc=1 unfused1,765,270 / fused1,760,518；640×640 batch1 THOP2.0.20：2,064,793,600MACs / 4,129,587,200FLOPs，未覆盖操作已列出             |
| 独立测速       | 仅CPU synthetic 64输入warmup1/iterations2检查接口；SMOKE且不具论文测速资格                                                              |
| 精简包         | 实际157,035 bytes、79成员；ZIP CRC、SHA及完整float32导出检查通过；无模型权重                                                            |
| 显式恢复       | 同一个SMOKE run恢复一个额外epoch，model/optimizer/EMA/scheduler/scaler/早停状态完整重载；有效更新计数2→3                                |
| 运行保护       | 已有run拒绝覆盖；无GPU等待/kill/自动降batch路径；tmux脚本bash语法通过，真实服务器tmux未验证                                             |
| 公共协议       | 改模型身份/优化器/输出路径保持公共hash；共享实现SHA256 `0ba1f5198cb24dd404212a25ffceeed14aa01b5468a79beabbf7e1828f299b6b`               |

本机：Python3.11.15、Torch2.7.1+cu118、Torchvision0.22.1+cu118、RTX2060。实际导入为本实验
`vendor/yolov5/models/yolo.py` 与 `vendor/yolov5/utils/general.py`，没有使用其他worktree的models包。
本地原始数据YAML字节SHA恰与历史值一致，但本次未读取全部真实图像进行清单核验。

服务器Python3.12/Torch2.8/RTX4090、真实完整数据清单、B19 args实际内容、可选Albumentations历史启用情况、
原四组test P/R阈值、组合模型服务器源码提交均为 **UNVERIFIED**。
正式preflight会核查可读取的数据/args，另需在服务器只读核对历史环境和旧test入口。不能据本机SMOKE推断精度或超过Ours。

兼容修补及删除在backend.lock.json中；原生优化公式、原生autoanchor默认和早停机制保留。
原生AMP接口的弃用warning不改变本次执行结果。冻结源码使用LF及.gitattributes，防止Windows/Linux检验值不同。

逐文件复用YOLOv8n共享源码；与其提交cd9e69720b5c0d6b3fa297ec588431099ded9e58的公共协议在同一合成数据及默认配置下完全一致，证据见shared_protocol_proof.json。
