<details>
<summary>中文｜八类交警手势分类实验记录</summary>

# 手势分类：无在线增强基线

## 任务与配置

实际任务为图像分类（task=classify），不输出检测框。八类为 LANE_CHANGING、LEFT_TURN、LEFT_TURN_WAITING、MOVE_STRAIGHT、PULL_OVER、RIGHT_TURN、SLOW_DOWN、STOP；没有独立背景类。

- 实验：baseline_100epc_no_online_augmentation_20261006_032754。
- 初始模型：yolov8n-cls.pt；数据：gesture_reflective_classifier_v4_adjusted。
- 图片先经裁切与 640×640 等比例补边，模型输入为 224×224。
- 计划 100 轮，CSV 实际记录 26 轮；patience=20。
- freeze=null，未指定层冻结；seed=42；deterministic=true。
- fliplr、flipud、scale、HSV、erasing 均为 0，auto_augment=null。“无在线增强”不表示原数据未经过离线增强。
- 初始配置 batch=0.6，表示自动显存预算；恢复后的 args.yaml 记录 batch=703。
- args.yaml 的 model 和 resume 指向 last.pt，表明该实验使用过断点恢复；初始设置保留在 experiment_config.json。

## 验证结果

| CSV 记录 | Epoch | Top-1 | Top-5 | 验证 loss |
| --- | --- | --- | --- | --- |
| 最早达到最高 Top-1 的一轮 | 6 | 99.106% | 100% | 0.02475 |
| 最后一轮 | 26 | 98.927% | 100% | 0.02529 |

本次验证集为 559 张。最佳逐轮成绩约对应 5 张错误，末轮约 6 张错误。最终混淆矩阵属于训练结束后的评估产物，不应直接当成末轮 CSV 的逐样本记录。

这些是验证成绩，不能代表独立测试集或部署准确率。原始帧 key 无交集只能排除同 key 重叠，不能证明不同视频、人物或相邻帧之间独立。训练阶段使用标注框裁切，实际推理使用检测框，仍需评估该差异。

## 文件与复现

- [experiment_config.json](experiment_config.json)：原始实验配置。
- [args.yaml](args.yaml)：记录的实际运行及恢复参数。
- [results.csv](results.csv)：逐轮指标。
- [训练曲线](results.png)、[混淆矩阵](confusion_matrix.png)、[归一化混淆矩阵](confusion_matrix_normalized.png)。
- [训练样例](train_batch0.jpg)、[验证标签](val_batch0_labels.jpg)、[验证预测](val_batch0_pred.jpg)。

目录未包含模型权重或数据集。使用配置前需修改本地路径。恢复训练还需要未清除优化器状态的 last.pt；仅有 JSON 和 CSV 无法恢复训练。数据集版本及训练代码提交号尚未完整归档。

</details>

<details>
<summary>English | 8-class traffic gesture classification</summary>

# Gesture classification: baseline without online augmentation

## Task and configuration

This is an image classification experiment (task=classify), not a bounding-box detector. Classes are LANE_CHANGING, LEFT_TURN, LEFT_TURN_WAITING, MOVE_STRAIGHT, PULL_OVER, RIGHT_TURN, SLOW_DOWN, and STOP. There is no background class.

- Run: baseline_100epc_no_online_augmentation_20261006_032754.
- Initial model: yolov8n-cls.pt; dataset: gesture_reflective_classifier_v4_adjusted.
- Crops are letterboxed to 640×640 while preserving aspect ratio; model input is 224×224.
- Planned epochs: 100; recorded epochs: 26; patience: 20.
- freeze=null (no requested layer freezing); seed=42; deterministic=true.
- fliplr, flipud, scale, HSV, and erasing are zero; auto_augment=null. Disabling online augmentation does not mean the source dataset lacks offline augmentation.
- Initial batch=0.6 requests automatic memory budgeting; resumed args.yaml records batch=703.
- The model and resume entries in args.yaml reference last.pt, documenting checkpoint resumption. experiment_config.json retains the initial settings.

## Validation results

| CSV record | Epoch | Top-1 | Top-5 | Validation loss |
| --- | --- | --- | --- | --- |
| First epoch attaining highest Top-1 | 6 | 99.106% | 100% | 0.02475 |
| Final epoch | 26 | 98.927% | 100% | 0.02529 |

The validation set contains 559 images, corresponding to approximately five errors at peak recorded accuracy and six in the final epoch. The final confusion matrix is an end-of-training evaluation artifact; it should not be assumed to represent the final CSV row sample by sample.

These are validation scores, not held-out test or deployment accuracy. Disjoint source-frame keys do not establish independence between videos, subjects, or neighboring frames. Training crops use annotations, while inference will use detector predictions; this difference requires evaluation.

## Artifacts and reproducibility

- [experiment_config.json](experiment_config.json): initial experiment settings.
- [args.yaml](args.yaml): recorded runtime and resume arguments.
- [results.csv](results.csv): per-epoch metrics.
- [Learning curves](results.png), [confusion matrix](confusion_matrix.png), [normalized matrix](confusion_matrix_normalized.png).
- [Training samples](train_batch0.jpg), [validation labels](val_batch0_labels.jpg), [validation predictions](val_batch0_pred.jpg).

Weights and datasets are not included. Update local paths before reuse. Resuming also requires a last.pt checkpoint retaining optimizer state; JSON and CSV alone are insufficient. The dataset version and training code commit have not been fully archived.

</details>
