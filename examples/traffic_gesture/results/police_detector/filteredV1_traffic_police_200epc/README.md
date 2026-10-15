<details>
<summary>中文｜交警检测模型实验记录</summary>

# 交警检测：筛选后数据集基线

## 任务与配置

使用 YOLOv8n 定位目标交警，作为两阶段手势识别系统的第一阶段。训练数据为人工筛选后的 TrafficPoliceV1，单类标签为 traffic_police。

- 实验：filteredV1_traffic_police_200epc。
- 预训练模型：yolov8n.pt；输入：640；batch：8。
- 计划 200 轮，CSV 实际记录 108 轮；patience=30。
- freeze=null（未指定层冻结）；optimizer=auto；seed=0；deterministic=true。
- 检测增强：HSV=(0.015, 0.7, 0.4)、平移 0.1、缩放 0.5、水平翻转概率 0.5、mosaic=1.0、close_mosaic=10。
- args.yaml 包含跨任务通用参数；其中 auto_augment、erasing 等分类配置不能据此认定用于本次检测训练。

## 验证结果

下表直接取自 results.csv。最佳记录按 mAP50–95 选取，不等同于另一次 best.pt 独立验证。

| 记录 | Epoch | Precision | Recall | mAP50 | mAP50–95 |
| --- | --- | --- | --- | --- | --- |
| 最高 mAP50–95 | 78 | 0.88130 | 0.78156 | 0.88089 | 0.57287 |
| 最后一轮 | 108 | 0.82589 | 0.81734 | 0.84869 | 0.54286 |

mAP50 与更严格的 mAP50–95 存在差距，定位精度仍有改进空间。上述结果来自验证集，不能代表真实摄像头、树莓派或完整两阶段系统的表现。

## 文件与复现

- [args.yaml](args.yaml)：实际运行参数。
- [results.csv](results.csv)：逐轮指标。
- [训练曲线](results.png)、[混淆矩阵](confusion_matrix.png)、[归一化混淆矩阵](confusion_matrix_normalized.png)。
- [PR 曲线](BoxPR_curve.png)、[F1 曲线](BoxF1_curve.png)。
- [标注分布](labels.jpg)、[验证标注示例](val_batch0_labels.jpg)、[验证预测示例](val_batch0_pred.jpg)。

本目录未包含模型权重或数据集。配置中的 Windows 绝对路径需按本机修改；数据集版本与对应代码提交号尚未以不可变标识完整归档。

</details>

<details>
<summary>English | Traffic officer detector experiment</summary>

# Traffic officer detection: filtered dataset baseline

## Task and configuration

A pretrained YOLOv8n model localizes target traffic officers as the first stage of a gesture-recognition system. Training uses the manually filtered TrafficPoliceV1 dataset with one class, traffic_police.

- Run: filteredV1_traffic_police_200epc.
- Model: yolov8n.pt; image size: 640; batch: 8.
- Planned epochs: 200; recorded epochs: 108; patience: 30.
- freeze=null (no requested layer freezing); optimizer=auto; seed=0; deterministic=true.
- Detection augmentation: HSV=(0.015, 0.7, 0.4), translation=0.1, scale=0.5, horizontal flip probability=0.5, mosaic=1.0, close_mosaic=10.
- args.yaml includes shared settings for multiple tasks. Classification options such as auto_augment and erasing are not evidence that those transforms ran during detector training.

## Validation results

These rows come directly from results.csv. The best recorded row is selected by mAP50–95; it is not a separate evaluation of best.pt.

| Record | Epoch | Precision | Recall | mAP50 | mAP50–95 |
| --- | --- | --- | --- | --- | --- |
| Highest mAP50–95 | 78 | 0.88130 | 0.78156 | 0.88089 | 0.57287 |
| Final epoch | 108 | 0.82589 | 0.81734 | 0.84869 | 0.54286 |

The gap between mAP50 and stricter mAP50–95 leaves room for better localization. These validation metrics do not establish camera, Raspberry Pi, or end-to-end system performance.

## Artifacts and reproducibility

- [args.yaml](args.yaml): runtime settings.
- [results.csv](results.csv): per-epoch metrics.
- [Learning curves](results.png), [confusion matrix](confusion_matrix.png), [normalized matrix](confusion_matrix_normalized.png).
- [PR curve](BoxPR_curve.png), [F1 curve](BoxF1_curve.png).
- [Label distribution](labels.jpg), [validation labels](val_batch0_labels.jpg), [validation predictions](val_batch0_pred.jpg).

Weights and datasets are not included. Adjust the Windows paths for your machine. Immutable dataset identifiers and the exact training code commit have not been fully archived.

</details>
