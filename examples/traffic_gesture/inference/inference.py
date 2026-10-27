from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageGrab, ImageFont
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
VIDEO_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "videos"
FONT = ImageFont.truetype(
    r"C:\Windows\Fonts\arial.ttf",
    size=36,
)

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
                position = (x1, max(0, y1-44))
            else:
                # 整图分类没有定位框，只在左上角显示类别。
                position = (10, 10)

            draw.text(
                position,
                label,
                font=FONT,
                fill="red"
            )
        return canvas


class ModelUtility:
    def __init__(self):
        pass

    @staticmethod
    def set_output_path(output_dir, prefix: str = "IMAGE", suffix: str = "jpg") -> dict:
        # 生成默认的时序文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = Path(output_dir / f"{prefix}_{timestamp}.{suffix}")
        return {"timestamp": timestamp, "path": output_path}

    @staticmethod
    def make_outputs_dir(image_output_dir: Path, video_output_dir: Path):
        # 检查图片输出路径是否存在并创建
        if image_output_dir.exists():
            print(f"图片输出目录已存在于{image_output_dir}")
        else:
            print(f"成功创建图片输出目录于{image_output_dir}")
            image_output_dir.mkdir(parents=True)
        # 检查视频输出路径是否存在并创建
        if video_output_dir.exists():
            print(f"视频输出目录已存在于{video_output_dir}")
        else:
            print(f"成功创建视频输出目录于{video_output_dir}")
            video_output_dir.mkdir(parents=True)

    def read_clipboard_image(self, output_dir) -> tuple[Image.Image, Path]:
        """读取剪贴板中的截图，返回 RGB 图片。"""
        content = ImageGrab.grabclipboard()

        if not isinstance(content, Image.Image):
            raise ValueError(
                "剪贴板中没有图片。请先用 Win+Shift+S 截图，再运行程序。"
            )

        output_path = self.set_output_path(output_dir)["path"]
        return content.convert("RGB"), output_path

    @staticmethod
    def read_video(video_path: Path, start_frame: int = 0) -> tuple[cv2.VideoCapture, dict]:
        """打开本地视频；调用者负责在使用结束后 release。"""
        if not video_path.is_file():
            raise FileNotFoundError(f"视频不存在：{video_path}")

        if start_frame < 0:
            raise ValueError("start_frame 不能小于 0。")

        # VideoCapture是视频读取器，并不是全部视频帧。
        # 后续每调用一次capture.read()，才读取下一帧。
        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"无法打开视频：{video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"尺寸：{width} × {height}")
        print(f"帧率：{fps}")
        print(f"声明帧数：{frame_count}")
        capture_info = {"fps": fps, "width": width, "height": height, "frame_count": frame_count}

        # 尝试从一个特定帧开始读取
        if 0 < frame_count <= start_frame:
            capture.release()
            raise ValueError(
                f"起始帧 {start_frame} 超出范围，视频声明共有 {frame_count} 帧。"
            )
        else:
            positioned = capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            if not positioned:
                capture.release()
                raise RuntimeError(f"无法定位到第 {start_frame} 帧。")
            return capture, capture_info

    def get_video_writer(self, output_dir: Path, capture_info: dict) -> cv2.VideoWriter:
        # 把读取的单独的视频帧写入为一个完整的循环，合成视频
        fps = capture_info["fps"]
        width = capture_info["width"]
        height = capture_info["height"]

        output_path = self.set_output_path(output_dir, "VIDEO", "mp4")["path"]

        if not (0 < fps < float("inf")) or width <= 0 or height <= 0:
            raise ValueError("视频的帧率或尺寸无效，无法创建输出视频。")

        codec = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path),
            codec,
            fps,
            (width, height)
        )

        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"无法创建输出视频: {output_path}")

        return writer

    def predict_video(self, pipeline: InferencePipeline, video_path: Path, start_frame=0, end_frame=-1, use_detector=True, progress=True):
        # 只预测视频并输出到目录，不返回任何值。输出视频范围：[start_frame, end_frame)
        if end_frame != -1 and end_frame <= start_frame:
            raise ValueError("end_frame 必须大于 start_frame，或为 -1。")

        capture, capture_info = self.read_video(video_path, start_frame=start_frame)  # 视频帧同样是从0开始，到n-1帧结束
        writer = None
        try:
            writer = self.get_video_writer(VIDEO_OUTPUT_DIR, capture_info)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if end_frame == -1:
                end_frame = frame_count
            frame_index = start_frame

            while end_frame == -1 or frame_index < end_frame:
                success, frame = capture.read()
                if not success:
                    break
                if progress:
                    print(f"当前处于{frame_index}/{end_frame}帧，进度{round(((frame_index-start_frame+2)/end_frame-start_frame)*100, 1)}%")

                # OpenCV 的 BGR 数组 → Pillow 的 RGB 图片
                image = Image.fromarray(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )

                predictions = pipeline.get_image_predictions(
                    image,
                    use_detector=use_detector,
                    display=False
                )

                annotated_image = pipeline.draw_predictions(image, predictions)
                # annotated_image.show()
                # 输入 Pillow 的 RGB 图片，输出 OpenCV 的 BGR
                output_frame = cv2.cvtColor(
                    np.asarray(annotated_image),
                    cv2.COLOR_RGB2BGR
                )
                writer.write(output_frame)
                frame_index += 1
        finally:
            capture.release()
            if writer is not None:
                writer.release()


def main():
    pipeline = InferencePipeline(DETECTOR_PATH, CLASSIFIER_PATH)
    print("分类模型的类别映射：", pipeline.classifier.names)

    util = ModelUtility()
    util.make_outputs_dir(IMAGE_OUTPUT_DIR, VIDEO_OUTPUT_DIR)

    video_path = Path(r"C:\Users\AvaAtlas\Videos\Captures\test_video.mp4")
    util.predict_video(pipeline, video_path)


if __name__ == "__main__":
    main()
