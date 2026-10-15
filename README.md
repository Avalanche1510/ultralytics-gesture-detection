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

- [prepare_gesture_classifier.py](examples/traffic_gesture/prepare_gesture_classifier.py): dataset filtering, cropping, and letterbox preprocessing.
- [train_gesture_classifier.py](examples/traffic_gesture/train_gesture_classifier.py): dataset checks, experiment configuration, training, and resumption.

The preparation script currently contains an unfinished `InferencePipeline` placeholder. Review local Windows dataset, checkpoint, and output paths before running either script. Images and trained weights are stored separately from the source code.

## Next steps

Connect the detector and classifier into one inference pipeline, inspect validation errors, evaluate generalization to new people and cameras, and perform held-out testing after model selection. Export and benchmark models on Raspberry Pi 5 before exploring LED or other hardware interactions.

## Resources

- [Project repository](https://github.com/Avalanche1510/ultralytics-gesture-detection)
- [Upstream Ultralytics](https://github.com/ultralytics/ultralytics)
- [Classification documentation](https://docs.ultralytics.com/tasks/classify/)
- [LabelImg annotation tool](https://github.com/HumanSignal/labelImg)

Local dataset names include `TrafficPolice` and `gesture(DST1556)`. Original download URLs and redistribution terms remain to be documented. No dataset or trained-weight download is currently provided here.

## License

This fork retains the upstream [GNU AGPL-3.0 license](LICENSE). Third-party datasets and dependencies retain their respective licenses.
