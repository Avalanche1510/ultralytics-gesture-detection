from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO


# 导入自定义的交警定位器，标注框裁剪器，标注框（正方形）塑形器
from examples.traffic_gesture.train.prepare_gesture_classifier_dataset import (
    TrafficPoliceDetector,
    GestureCropTransformer,
    LetterboxTransformer,
)
# 导入ModelUtil工具类
from examples.traffic_gesture.inference.model_util import (
    ModelUtility
)

from examples.traffic_gesture.inference.defaults import *


class InferencePipeline:
    """组合交警检测、裁切预处理与手势分类。"""
    def __init__(self, detector_path: Path, classifier_path: Path):
        """加载两个模型及裁剪、letterbox 预处理器。

        Args:
            detector_path (Path): 交警检测模型权重路径。
            classifier_path (Path): 手势分类模型权重路径。
        """
        self.detector = TrafficPoliceDetector(
            model_path=detector_path,
            confidence=0.68
        )
        self.crop_transformer = GestureCropTransformer(
            pad_x=0.15,
            pad_y=0.1
        )
        self.letterbox_transformer = LetterboxTransformer(
            size=640,
            pad_color=(114, 114, 114)
        )
        self.classifier = YOLO(model=classifier_path)

    def identify_police(self, image: Image.Image, display=False, tracking=False) -> list[dict]:
        """检测或跟踪交警，返回每个目标的位置和检测分数。

        Args:
            image (Image.Image): 待检测的 Pillow 图片。
            display (bool): 是否打印目标信息。
            tracking (bool): 是否保留连续帧跟踪状态；新视频应先重置跟踪器。

        Returns:
            (list[dict]): 每个目标对应一个字典，结构如下::

                [
                    {
                        "track_id": int,               # 轨迹 ID 或帧内序号
                        "box": tuple,                  # 原图像素坐标 (x1, y1, x2, y2)
                        "detector_confidence": float,  # 交警检测分数
                    },
                    ...
                ]

            无目标或无可用跟踪 ID 时返回 []。跟踪模式使用轨迹 ID；
            普通检测使用从 0 开始的帧内序号，不能跨帧关联。
        """
        if tracking:
            detections = self.detector.track(image)
        else:
            track_id = -1
            detections = []
            for xyxy, confidence in self.detector.detect(image):
                track_id += 1
                detections.append({
                        "track_id": track_id,
                        "box": xyxy,
                        "detector_confidence": confidence
                    })

        if display:
            print(f"检测到 {len(detections)} 个目标")

            for detection in detections:
                print(
                    f"ID={detection['track_id']}，"  # track_id是int类型
                    f"框={detection['box']}，"
                    f"检测置信度={detection['detector_confidence']:.3f}"
                )

        return detections

    def classify_gesture(self, image: Image.Image, detection=None, display=False) -> dict:
        """裁剪并塑形后分类；没有检测结果时对整张图片分类。

        Args:
            image (Image.Image): 原始 Pillow 图片。
            detection (dict, optional): 检测字典，需含 "box" 和 "detector_confidence"；None 表示整图分类。
            display (bool): 是否显示预处理图片并打印分类概率。

        Returns:
            (dict): 分类概率及类别映射。返回结构如下::

                {
                    "gesture_confidences": np.ndarray,  # 所有类别概率，形状 (8,)
                    "result_names": dict[int, str],     # 类别 ID → 名称
                }

            有 detection 时，额外包含以下字段::

                {
                    "box": tuple,                  # 原检测框 (x1, y1, x2, y2)
                    "crop_box": tuple,             # 扩边裁剪框 (left, top, right, bottom)
                    "detector_confidence": float,  # 检测置信度
                }

            返回值不包含 track_id；目标 ID 由外层字典的键保存。
        """
        if detection is not None:
            xyxy, detector_confidence = detection["box"], detection["detector_confidence"]
            # 根据检测框扩边裁切，不再读取数据集标签框。
            crop_box = self.crop_transformer.expand_clip_xyxy(xyxy, image.size)
            crop = image.crop(crop_box)
        else:
            crop = image
        prepared_image = self.letterbox_transformer.apply(crop)

        if display:
            prepared_image.show()
        # 每次只分类一张裁切图，因此取第一个 Results。
        result = self.classifier(
            prepared_image,
            imgsz=224,
            verbose=False,
        )[0]

        if display:
            print("=======================================================")
            for class_id, prob in enumerate(result.probs.data):
                gesture_name = result.names[class_id]
                gesture_top_confidence = prob.item()
                print(f"类别: {gesture_name}, 置信度: {gesture_top_confidence}")
            print("=======================================================")
        # class_id = result.probs.top1
        # gesture_name = result.names[class_id]
        # gesture_top_confidence = result.probs.top1conf.item()
        gesture_confidences = result.probs.data.cpu().numpy().copy()

        if detection is not None:
            prediction = {
                "box": xyxy,
                "crop_box": crop_box,
                "detector_confidence": detector_confidence,
                "gesture_confidences": gesture_confidences,  # 当前帧全部8个手势分类的confidences，用于平滑的视频识别
                "result_names": result.names,
            }
        else:
            prediction = {
                "gesture_confidences": gesture_confidences,
                "result_names": result.names
            }

        if display:
            for key in prediction:
                print(prediction[key])

        return prediction

    def get_image_predictions(self, image: Image.Image, use_detector=True, display=False, tracking=False) -> dict[int, dict]:
        """处理一张图片，将各目标分类结果按 ID 组织成字典。

        Args:
            image (Image.Image): 输入图片，内部转换为 RGB。
            use_detector (bool): 是否检测人员；False 时直接整图分类。
            display (bool): 是否打印诊断信息并显示分类输入图片。
            tracking (bool): 是否跟踪；仅在 use_detector=True 时生效。

        Returns:
            (dict[int, dict]): 按目标 ID 组织的预测，结构如下::

                {
                    target_id: {
                        "gesture_confidences": np.ndarray,  # 类别概率，形状 (8,)
                        "result_names": dict[int, str],     # 类别 ID → 名称
                        "box": tuple,                      # 仅检测模式：原检测框
                        "crop_box": tuple,                 # 仅检测模式：扩边裁剪框
                        "detector_confidence": float,      # 仅检测模式：检测分数
                    },
                    ...
                }

            跟踪时 target_id 为轨迹 ID，普通检测时为帧内序号；
            整图分类固定使用键 -1，无目标时为 {}。框坐标格式见 classify_gesture。
        """
        image = image.convert("RGB")
        predictions = dict()

        if not use_detector:
            predictions[-1] = self.classify_gesture(image, display=display)
            return predictions

        detections = self.identify_police(image, display=display, tracking=tracking)

        for detection in detections:
            predictions[detection["track_id"]] = self.classify_gesture(image, detection=detection, display=display)

        return predictions

    @staticmethod
    def update_confidences(current_confidences: np.ndarray, history_confidences: np.ndarray, smoothing_params: dict = None):
        """用 EMA 融合当前概率和历史平滑概率，不原地修改输入数组。

        Args:
            current_confidences (np.ndarray): 当前类别概率，一维数组，当前形状为 (8,)。
            history_confidences (np.ndarray): 相同类别顺序、相同形状的历史平滑概率。
            smoothing_params (dict, optional): {"method": "EMA", "alpha": 0.5} 为默认值。
                "method" (str) 目前只支持 "EMA"；"alpha" (float) 为当前帧权重，
                调用者应保证 0 < alpha <= 1，当前实现不校验范围。

        Returns:
            (np.ndarray): alpha * 当前概率 + (1 - alpha) * 历史概率，形状与输入相同。
        """
        if smoothing_params is None:
            smoothing_params = {"method": "EMA", "alpha": 0.2}

        # EMA: Exponential Moving Average，指数移动平均
        if smoothing_params["method"] == "EMA":
            return smoothing_params["alpha"] * current_confidences + (1 - smoothing_params["alpha"]) * history_confidences

        # Power Sharpened EMA: 对current_confidences取平方锐化，然后除以总值做归一化，随后用普通EMA的方式融合历史
        elif smoothing_params["method"] == "PSEMA":
            # 锐化current_confidences并归一化处理
            current_confidences = (current_confidences ** 2) / (current_confidences ** 2).sum()
            # 常规EMA
            return smoothing_params["alpha"] * current_confidences + (1 - smoothing_params["alpha"]) * history_confidences
        else:
            raise RuntimeError(f"不支持的平滑方式: {smoothing_params['method']}")

    def smoothing(self, current_predictions: dict[int, dict], history_predictions: dict[int, dict], smoothing_params: dict = None):
        """按目标 ID 原地更新当前预测的概率，只平滑具有历史的目标。

        Args:
            current_predictions (dict[int, dict]): 当前帧 {ID: prediction}，每项需含 "gesture_confidences"。
            history_predictions (dict[int, dict]): 历史 {ID: prediction}，每项需含 "gesture_confidences"。
            smoothing_params (dict, optional): "method" 和 "alpha" 的含义、默认值见 update_confidences。

        Returns:
            (dict[int, dict]): 原 current_predictions 对象，结构如下::

                {
                    target_id: {
                        "gesture_confidences": np.ndarray,  # 匹配 ID 时替换为 EMA 概率
                        "result_names": dict[int, str],     # 保持不变
                        "box": tuple,                      # 已有时保留，不额外添加
                        "crop_box": tuple,                 # 已有时保留，不额外添加
                        "detector_confidence": float,      # 已有时保留，不额外添加
                    },
                    ...
                }

            新 ID 在 EMA 模式下保留原始概率，在 PSEMA 模式下使用平方归一化后的概率初始化。

        Notes:
            本方法不保存历史、不决定历史有效期；这些策略由视频调用方负责。
        """
        if smoothing_params is None:
            smoothing_params = {"method": "EMA", "alpha": 0.2}

        # EMA: Exponential Moving Average，指数移动平均
        if smoothing_params["method"] == "EMA":
            for track_id in current_predictions.keys():
                current_prediction = current_predictions[track_id]

                # 查询当前预测目标是否具有历史
                if track_id in history_predictions.keys():
                    current_confidences = current_prediction["gesture_confidences"]
                    history_confidences = history_predictions[track_id]["gesture_confidences"]
                    # 平滑更新current_prediction
                    current_predictions[track_id]["gesture_confidences"] = self.update_confidences(current_confidences, history_confidences, smoothing_params=smoothing_params)
                else:
                    # 第一帧不做平滑，此处按格式显式处理第一帧
                    pass
            return current_predictions
        elif smoothing_params["method"] == "PSEMA":
            for track_id in current_predictions.keys():
                current_prediction = current_predictions[track_id]
                current_confidences = current_predictions[track_id]["gesture_confidences"]

                # 查询当前预测目标是否具有历史
                if track_id in history_predictions.keys():
                    current_confidences = current_prediction["gesture_confidences"]
                    history_confidences = history_predictions[track_id]["gesture_confidences"]
                    # 平滑更新current_prediction
                    current_predictions[track_id]["gesture_confidences"] = self.update_confidences(current_confidences, history_confidences, smoothing_params=smoothing_params)
                else:
                    # 第一帧只做锐化
                    current_predictions[track_id]["gesture_confidences"] = (current_confidences ** 2) / (current_confidences ** 2).sum()

            return current_predictions
        else:
            raise RuntimeError(f"不支持的平滑方式: {smoothing_params['method']}")

    @staticmethod
    def determine_prediction_result(prediction: dict) -> dict:
        """从原始或平滑后的概率中提取最高分的类别。

        Args:
            prediction (dict): 必须含 "gesture_confidences" (一维 np.ndarray) 和
                "result_names" (dict[int, str])，两者类别编号应一致。

        Returns:
            (dict): 最高分分类结果，结构如下::

                {
                    "gesture_result_id": int,     # 最大概率的类别索引
                    "gesture_result": str,        # 类别名称
                    "gesture_confidence": float,  # 对应的原始或平滑概率
                }

            最大概率并列时取第一个索引；输入字典不被修改。
        """
        class_id = int(np.argmax(prediction["gesture_confidences"]))
        confidences = prediction["gesture_confidences"]
        prediction_result = {
            "gesture_result_id": class_id,
            "gesture_result": prediction["result_names"][class_id],
            "gesture_confidence": float(confidences[class_id])
        }
        return prediction_result

    def draw_predictions(self, image: Image.Image, predictions: dict[int, dict], save: bool = False, save_dir: Path = IMAGE_OUTPUT_DIR) -> Image.Image:
        """在原图副本上绘制当前概率对应的类别，可选保存为 JPEG。

        Args:
            image (Image.Image): 原始图片，不会被原地修改。
            predictions (dict[int, dict]): {目标 ID: prediction}；每项必须含 "gesture_confidences"、
                "result_names"。可选 "box" 绘制蓝色检测框、"crop_box" 绘制橙色裁剪框；
                两种框均为原图像素坐标 (x1, y1, x2, y2)。无检测框时在左上角绘制文本。
            save (bool): True 时保存 JPEG 并打印路径；False 时只返回画布，不检查或创建输出目录。
            save_dir (Path): JPEG 的实际输出目录，默认 IMAGE_OUTPUT_DIR；仅在保存时创建缺失的目录及父目录。

        Returns:
            (Image.Image): 绘制后的 RGB 图片副本，不返回保存路径。
        """

        canvas = image.convert("RGB").copy()
        draw = ImageDraw.Draw(canvas)

        for track_id in predictions.keys():
            prediction = predictions[track_id]
            # 实时解析分类结果
            prediction_result = self.determine_prediction_result(prediction)
            label = (
                f"Entity <{track_id}> "
                f"{prediction_result['gesture_result']} - "
                f"{prediction_result['gesture_confidence']:.2f}"
            )
            if "crop_box" in prediction:
                draw.rectangle(
                    prediction["crop_box"],
                    outline="orange",
                    width=2,
                )
            if "box" in prediction:
                x1, y1, x2, y2 = prediction["box"]
                draw.rectangle(
                    (x1, y1, x2, y2),
                    outline="blue",
                    width=3
                )
                position = (x1, max(0, y1-44))
            else:
                # 整图分类没有定位框，只在左上角显示类别。
                position = (10, 10)

            draw.text(
                position,
                label,
                font_size=FONT_SIZE,
                fill="red"
            )

        if save:
            ModelUtility.make_outputs_dir(image_output_dir=save_dir, scope=["image"])
            save_path = ModelUtility.set_output_path(save_dir)["path"]
            canvas.save(save_path, quality=95)
            print(f"图片已保存至：{save_path}")
        return canvas


def main():
    pipeline = InferencePipeline(DETECTOR_PATH, CLASSIFIER_PATH)
    print("分类模型的类别映射：", pipeline.classifier.names)

    util = ModelUtility()

    video_path = Path(VIDEO_INPUT_DIR / "test_video.mp4")
    util.predict_video(pipeline, video_path, smoothing_params={"method": "PSEMA", "alpha": 0.2})

    # image_path = Path(r"C:\Users\AvaAtlas\Pictures\Screenshots\Screenshot 2026-10-25 125547.png")
    # with Image.open(image_path) as image:
    #     predictions = pipeline.get_image_predictions(image)
    #     pipeline.draw_predictions(image, predictions, save=True)


if __name__ == "__main__":
    main()
