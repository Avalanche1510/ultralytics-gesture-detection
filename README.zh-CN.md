# 交警指挥手势识别

简体中文 | [English](README.md)

本项目基于 [Ultralytics](https://github.com/ultralytics/ultralytics)，用于学习和探索“交警检测 + 手势分类”的两阶段视觉系统：先定位穿着反光执勤服的目标人员，再识别八类交警指挥手势。长期目标是在树莓派 5 上实现本地推理与硬件交互。

本仓库是 Ultralytics 的定制开发分支，并非官方上游仓库。

## 技术路线

计划中的推理流程为：输入图像 → YOLOv8n 交警检测 → 扩边裁切 → 等比例缩放并补边 → YOLOv8n-cls 手势分类。

八类手势包括停止、直行、左转弯、右转弯、变道、减速慢行、靠边停车、左转弯待转。目前分类器没有独立的背景类别。

数据准备阶段由检测器筛选原始帧分组，再根据原始手势标注框裁切；推理阶段将使用检测器的预测框。两种裁切来源的差异仍需通过端到端实验验证。

## 当前进度

- 已完成交警检测模型训练和数据集人工筛查。
- 已实现按原始帧分组筛选、扩边裁切及 640 × 640 正方形补边。
- 已检查训练、验证、测试集之间的原始帧 key 重叠。
- 已实现分类训练、JSON 实验配置保存与读取、断点恢复。
- 已完成 YOLOv8n-cls 全量微调基线：输入为 224 × 224，关闭在线增强。记录的最后一轮在 559 张验证图像上取得 98.927% Top-1 准确率。

该指标属于验证集结果，不代表独立测试或真实部署效果。原始帧 key 不重叠，也不等于人物或视频在不同集合之间相互独立。

端到端推理管线正在建设中；树莓派部署与硬件控制尚未完成。

## 项目文件

- [prepare_gesture_classifier_dataset.py](examples/traffic_gesture/train/prepare_gesture_classifier_dataset.py)：数据筛选、裁切与补边预处理。
- [train_gesture_classifier.py](examples/traffic_gesture/train/train_gesture_classifier.py)：数据检查、实验配置、分类训练与恢复。

- [inference.py](examples/traffic_gesture/inference/inference.py)：正在建设的推理入口，尚未实现完整两阶段推理。

运行前需要检查数据集、模型权重和输出目录配置。数据准备模块中仍保留未实现的 `InferencePipeline` 占位类。

## 实验结果与权重

- [交警检测实验](examples/traffic_gesture/results/police_detector/filteredV1_traffic_police_200epc/README.md) · [best.pt](examples/traffic_gesture/results/police_detector/filteredV1_traffic_police_200epc/weights/best.pt)
- [手势分类实验](examples/traffic_gesture/results/gesture_classifier/baseline_100epc_no_online_augmentation_20261006_032754/README.md) · [best.pt](examples/traffic_gesture/results/gesture_classifier/baseline_100epc_no_online_augmentation_20261006_032754/weights/best.pt)

结果目录包含配置、指标、曲线、样例图及 best.pt/last.pt 权重。推理优先使用 best.pt。名为 last.pt 不保证能够续训：训练正常结束后可能已移除优化器状态。完整数据集未包含在仓库中。配置中的 Windows 路径记录原训练环境，复现时需要调整。

## 后续计划

串联交警检测与手势分类，建立统一推理入口；分析验证错误，评估新人物、新摄像头下的泛化能力，在模型选择结束后进行独立测试。随后导出模型，在树莓派 5 上测试速度，并探索 LED 等硬件交互。

## 资源链接

- [本项目仓库](https://github.com/Avalanche1510/ultralytics-gesture-detection)
- [Ultralytics 上游仓库](https://github.com/ultralytics/ultralytics)
- [分类任务文档](https://docs.ultralytics.com/tasks/classify/)
- [LabelImg 标注工具](https://github.com/HumanSignal/labelImg)
- [核心工作区](./examples/traffic_gesture/)

开发过程中使用的数据集本地名称包括 `TrafficPolice` 和 `gesture(DST1556)`。原始下载地址和再分发条款尚待补充，目前未在此提供数据集下载。

## 许可证

本仓库沿用上游的 [GNU AGPL-3.0 许可证](LICENSE)。第三方数据集与依赖分别遵循其自身许可证。
