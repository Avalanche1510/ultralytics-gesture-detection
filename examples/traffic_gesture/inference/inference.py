from pathlib import Path
from PIL import Image, ImageDraw, ImageGrab
from ultralytics import YOLO
from datetime import datetime
import cv2

# 导入自定义的交警定位器，标注框裁剪器，标注框（正方形）塑形器
from examples.traffic_gesture.train.prepare_gesture_classifier_dataset import (
    TrafficPoliceDetector,
    GestureCropTransformer,
    LetterboxTransformer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
IMAGE_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "images"

DETECTOR_PATH = (
    RESULTS_ROOT
    / "police_detector"
    / "filteredV1_traffic_police_200epc"
    / "weights"
    / "best.pt"
)

CLASSIFIER_PATH = (
    RESULTS_ROOT
    / "gesture_classifier"
    / "baseline_100epc_no_online_augmentation_20261006_032754"
    / "weights"
    / "best.pt"
)


class InferencePipeline:
    """组合交警检测、裁切预处理与手势分类。"""
    def __init__(self, detector_path: Path, classifier_path: Path):
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

    def identify_police(self, image: Image.Image, display=False):
        """输入Image.Image类型，只检测一张图片中的目标交警，并返回一个检测框列表"""
        detections = self.detector.detect(image)

        if display:
            print(f"检测到 {len(detections)} 个目标")

            for index, (xyxy, confidence) in enumerate(detections):
                print(
                    f"目标 {index}: "
                    f"框={xyxy}, 检测置信度={confidence:.3f}"
                )

        return detections

    def classify_gesture(self, image: Image.Image, detection=None, display=False):
        """输入Image.Image类型，裁剪塑形，并分类一张图片中人员的交警手势"""
        if detection is not None:
            xyxy, detector_confidence = detection[0], detection[1]
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
                gesture_confidence = prob.item()
                print(f"类别: {gesture_name}, 置信度: {gesture_confidence}")
            print("=======================================================")
        class_id = result.probs.top1
        gesture_name = result.names[class_id]
        gesture_confidence = result.probs.top1conf.item()

        if detection is not None:
            prediction = {
                "box": xyxy,
                "crop_box": crop_box,
                "detector_confidence": detector_confidence,
                "gesture": gesture_name,
                "gesture_confidence": gesture_confidence,
            }
        else:
            prediction = {
                "gesture": gesture_name,
                "gesture_confidence": gesture_confidence,
            }

        if display:
            for key in prediction:
                print(prediction[key])

        return prediction

    def get_image_predictions(self, image: Image.Image, use_detector=True, display=False) -> list:
        """处理一张图片，返回图片内所有目标的手势预测。"""
        image = image.convert("RGB")

        if not use_detector:
            prediction = self.classify_gesture(image, display=display)
            return [prediction]

        detections = self.identify_police(image, display=display)

        predictions = []
        for detection in detections:
            prediction = self.classify_gesture(image, detection=detection, display=display)
            predictions.append(prediction)

        return predictions

    @staticmethod
    def draw_predictions(image: Image.Image, predictions: list[dict]) -> Image.Image:
        """在1张原图副本上绘制预测结果，返回1张可视化图片。"""
        canvas = image.convert("RGB").copy()
        draw = ImageDraw.Draw(canvas)

        for prediction in predictions:
            label = (
                f"{prediction['gesture']} - "
                f"{prediction['gesture_confidence']:.2f}"
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
                position = (x1, max(0, y1-16))
            else:
                # 整图分类没有定位框，只在左上角显示类别。
                position = (10, 10)

            draw.text(
                position,
                label,
                fill="red",
                stroke_width=0,
                stroke_fill="black"
            )
        return canvas

    @staticmethod
    def make_outputs_dir(image_output_dir: Path):
        if image_output_dir.exists():
            print(f"输出目录已存在于{image_output_dir}")
        else:
            print(f"成功创建输出目录于{image_output_dir}")
            image_output_dir.mkdir(parents=True)


class ModelUtility:
    def __init__(self):
        pass

    @staticmethod
    def read_clipboard_image() -> Image.Image:
        """读取剪贴板中的截图，返回 RGB 图片。"""
        content = ImageGrab.grabclipboard()

        if not isinstance(content, Image.Image):
            raise ValueError(
                "剪贴板中没有图片。请先用 Win+Shift+S 截图，再运行程序。"
            )

        return content.convert("RGB")

    def read_video(self):
        pass


def main():
    pipeline = InferencePipeline(DETECTOR_PATH, CLASSIFIER_PATH)
    pipeline.make_outputs_dir(IMAGE_OUTPUT_DIR)
    print("分类模型的类别映射：", pipeline.classifier.names)

    # test_image_path = Path(r"C:\Users\AvaAtlas\Pictures\Screenshots\Screenshot 2026-10-25 125822.png")
    # with Image.open(test_image_path) as image:
    #     predictions = pipeline.get_image_predictions(image, use_detector=True)
    #     annotated_image = pipeline.draw_predictions(image, predictions)
    # output_path = IMAGE_OUTPUT_DIR / f"{test_image_path.stem}_prediction.jpg"
    util = ModelUtility()
    image = util.read_clipboard_image()

    predictions = pipeline.get_image_predictions(
        image,
        use_detector=True,
        display=False,
    )
    annotated_image = pipeline.draw_predictions(image, predictions)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = IMAGE_OUTPUT_DIR / f"clipboard_{timestamp}_prediction.jpg"

    annotated_image.save(output_path, quality=95)
    print(f"预测图片已保存：{output_path}")
    annotated_image.show()

    for prediction in predictions:
        print(prediction)


if __name__ == "__main__":
    main()
