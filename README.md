# Traffic Police Gesture Recognition

[简体中文](README.zh-CN.md) | English

A learning and prototyping project based on [Ultralytics](https://github.com/ultralytics/ultralytics). It explores a two-stage system that detects traffic officers wearing reflective uniforms and classifies eight traffic-control gestures, with Raspberry Pi 5 deployment as a future goal.

This is a customized Ultralytics fork, not the official upstream repository.

## Approach

The planned inference pipeline is: image → YOLOv8n officer detection → expanded crop → aspect-ratio-preserving letterbox → YOLOv8n-cls gesture classification.

The eight classes are stop, move straight, left turn, right turn, lane changing, slow down, pull over, and left-turn waiting. The classifier currently has no separate background class.

During dataset preparation, the detector selects source-frame groups while original gesture annotations define the crops. Inference will use predicted detection boxes; this difference still needs end-to-end evaluation.

## Current progress

- Completed officer detector training and manual dataset review.
- Implemented source-group filtering, padded cropping, and 640 × 640 letterbox preparation.
- Checked source-frame key overlap across training, validation, and test splits.
- Implemented classification training, JSON experiment configuration, and checkpoint resumption.
- Completed an unfrozen YOLOv8n-cls baseline with 224 × 224 input and online augmentation disabled. The recorded final epoch reached 98.927% validation Top-1 accuracy on 559 images.

This is a validation result, not an independent test or deployment result. Separating source-frame keys does not by itself ensure separation by person or video.

End-to-end inference is under development. Raspberry Pi deployment and hardware control have not yet been completed.

## Project files

- [prepare_gesture_classifier_dataset.py](examples/traffic_gesture/train/prepare_gesture_classifier_dataset.py): dataset filtering, cropping, and letterbox preprocessing.
- [train_gesture_classifier.py](examples/traffic_gesture/train/train_gesture_classifier.py): dataset checks, experiment configuration, training, and resumption.

- [inference.py](examples/traffic_gesture/inference/inference.py): inference entry point under construction; an end-to-end pipeline is not yet implemented.

Review dataset, checkpoint, and output settings before running the scripts. The dataset preparation module still contains an unfinished `InferencePipeline` placeholder.

## Experiment artifacts and weights

- [Officer detector results](examples/traffic_gesture/results/police_detector/filteredV1_traffic_police_200epc/README.md) · [best.pt](examples/traffic_gesture/results/police_detector/filteredV1_traffic_police_200epc/weights/best.pt)
- [Gesture classifier results](examples/traffic_gesture/results/gesture_classifier/baseline_100epc_no_online_augmentation_20261006_032754/README.md) · [best.pt](examples/traffic_gesture/results/gesture_classifier/baseline_100epc_no_online_augmentation_20261006_032754/weights/best.pt)

The result folders include configurations, metrics, plots, sample images, and best.pt/last.pt weights. Use best.pt for inference. A file named last.pt is not necessarily resumable: completed runs may have optimizer state removed. Full datasets are not included. Recorded Windows paths describe the original training environment and must be adapted for reproduction.

## Next steps

Connect the detector and classifier into one inference pipeline, inspect validation errors, evaluate generalization to new people and cameras, and perform held-out testing after model selection. Export and benchmark models on Raspberry Pi 5 before exploring LED or other hardware interactions.

## Resources

- [Project repository](https://github.com/Avalanche1510/ultralytics-gesture-detection)
- [Upstream Ultralytics](https://github.com/ultralytics/ultralytics)
- [Classification documentation](https://docs.ultralytics.com/tasks/classify/)
- [LabelImg annotation tool](https://github.com/HumanSignal/labelImg)

Local dataset names include `TrafficPolice` and `gesture(DST1556)`. Original download URLs and redistribution terms remain to be documented. Full datasets are not provided; archived model weights are linked above.

## License

This fork retains the upstream [GNU AGPL-3.0 license](LICENSE). Third-party datasets and dependencies retain their respective licenses.
