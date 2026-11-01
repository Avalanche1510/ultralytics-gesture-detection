from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageGrab, ImageFont
from ultralytics import YOLO
from datetime import datetime
import cv2
from typing import Union

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

    def identify_police(self, image: Image.Image, display=False, tracking=False) -> list[dict]:
        """检测或跟踪交警；视频模式下同时返回跟踪 ID。"""
        if tracking:
            detections = self.detector.track(image)
        else:
            detections = [
                {
                    "track_id": None,
                    "box": xyxy,
                    "detector_confidence": confidence
                }
                for xyxy, confidence in self.detector.detect(image)
            ]

        if display:
            print(f"检测到 {len(detections)} 个目标")

            for detection in enumerate(detections):
                print(
                    f"ID={detection['track_id']}，"  # track_id是int类型
                    f"框={detection['box']}，"
                    f"检测置信度={detection['detector_confidence']:.3f}"
                )

        return detections

    def classify_gesture(self, image: Image.Image, detection=None, display=False) -> dict:
        """输入Image.Image类型，裁剪塑形，并分类一张图片中人员的交警手势"""
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
        class_id = result.probs.top1
        gesture_name = result.names[class_id]
        gesture_top_confidence = result.probs.top1conf.item()
        gesture_confidences = result.probs.data.cpu().numpy().copy()

        if detection is not None:
            prediction = {
                "box": xyxy,
                "crop_box": crop_box,
                "detector_confidence": detector_confidence,
                "gesture": gesture_name,
                "gesture_top_confidence": gesture_top_confidence,
                "gesture_confidences": gesture_confidences,  # 当前帧全部8个手势分类的confidences，用于平滑的视频识别
                "result_names": result.names,
            }
        else:
            prediction = {
                "gesture": gesture_name,
                "gesture_top_confidence": gesture_top_confidence,
                "gesture_confidences": gesture_confidences,
                "result_names": result.names
            }

        if display:
            for key in prediction:
                print(prediction[key])

        return prediction

    def get_image_predictions(self, image: Image.Image, use_detector=True, display=False, tracking=False) -> dict[int, dict]:
        """处理一张图片，返回图片内所有目标的手势预测。"""
        image = image.convert("RGB")
        predictions = dict()

        if not use_detector:
            predictions[-1] = self.classify_gesture(image, display=display)
            return predictions

        detections = self.identify_police(image, display=display, tracking=tracking)

        for detection in detections:
            predictions[detection["track_id"]] = self.classify_gesture(image, detection=detection, display=display)

        return predictions

    # @staticmethod
    # def update_confidences(current_confidences: np.ndarray, history_confidences: np.ndarray, smoothing_params: dict = None):
    #     # 更新平滑后的分类置信度
    #     if smoothing_params is None:
    #         smoothing_params = {"method": "EMA", "alpha": 0.5}
    #
    #     # EMA: Exponential Moving Average，指数移动平均
    #     if smoothing_params["method"] == "EMA":
    #         return smoothing_params["alpha"] * current_confidences + (1 - smoothing_params["alpha"]) * history_confidences
    #     else:
    #         raise RuntimeError(f"不支持的平滑方式: {smoothing_params['method']}")
    #
    # def smoothing(self, current_confidences_dict: dict[np.ndarray], history_confidences_dict: dict[np.ndarray], smoothing_params: dict = None):
    #     # 更新平滑后的分类结果
    #     if smoothing_params is None:
    #         smoothing_params = {"method": "EMA", "alpha": 0.5}
    #
    #     # EMA: Exponential Moving Average，指数移动平均
    #     if smoothing_params["method"] == "EMA":
    #         for track_id in current_confidences_dict.keys():
    #             # 查询当前预测目标是否具有历史
    #             if track_id in history_confidences_dict.keys():
    #                 smoothed_confidences = self.update_confidences(current_confidences_dict[track_id], history_confidences_dict[track_id], smoothing_params=smoothing_params)
    #                 current_confidences_dict[track_id] = smoothed_confidences
    #         return current_confidences_dict
    #     else:
    #         raise RuntimeError(f"不支持的平滑方式: {smoothing_params['method']}")

    @staticmethod
    def draw_predictions(image: Image.Image, predictions: dict[int, dict]) -> Image.Image:
        """在1张原图副本上绘制预测结果，返回1张可视化图片。"""
        canvas = image.convert("RGB").copy()
        draw = ImageDraw.Draw(canvas)

        for track_id in predictions.keys():
            prediction = predictions[track_id]
            label = (
                f"Entity <{track_id}> "
                f"{prediction['gesture']} - "
                f"{prediction['gesture_top_confidence']:.2f}"
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

    # def predict_video(self, pipeline: InferencePipeline, video_path: Path,  smoothing_params: dict = None, start_frame=0, end_frame=-1, use_detector=True, progress=True):
    #     # 只预测视频并输出到目录，不返回任何值。默认开启跟踪模式，但是可选择是否使用平滑模式。输出视频范围：[start_frame, end_frame)
    #     if end_frame != -1 and end_frame <= start_frame:
    #         raise ValueError("end_frame 必须大于 start_frame，或为 -1。")
    #
    #     capture, capture_info = self.read_video(video_path, start_frame=start_frame)  # 视频帧同样是从0开始，到n-1帧结束
    #     writer = None
    #     try:
    #         writer = self.get_video_writer(VIDEO_OUTPUT_DIR, capture_info)
    #         frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    #         if end_frame == -1:
    #             end_frame = frame_count
    #         frame_index = start_frame
    #
    #         history_confidences_dict = []
    #
    #         while end_frame == -1 or frame_index < end_frame:
    #             success, frame = capture.read()
    #             if not success:
    #                 break
    #             if progress:
    #                 print(f"当前处于第{frame_index+1}/{end_frame}帧，进度{round((frame_index - start_frame + 1) / (end_frame - start_frame) * 100, 1)}%")
    #
    #             # OpenCV 的 BGR 数组 → Pillow 的 RGB 图片
    #             image = Image.fromarray(
    #                 cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #             )
    #             predictions = pipeline.get_image_predictions(
    #                 image,
    #                 use_detector=use_detector,
    #                 display=False,
    #                 tracking=True,
    #             )
    #
    #             # 将confidences从predictions中提取出来，组合成一个dict，方便查找和对比track_id
    #             current_confidences_dict = {}
    #             for prediction in predictions:
    #                 current_confidences_dict[prediction["track_id"]] = prediction["gesture_confidences"]
    #
    #             if smoothing_params is not None:
    #                 current_confidences_dict = pipeline.smoothing(current_confidences_dict, history_confidences_dict, smoothing_params)
    #                 for track_id in current_confidences_dict.keys():
    #
    #
    #             history_confidences_dict = current_confidences_dict
    #
    #             annotated_image = pipeline.draw_predictions(image, predictions)
    #             # annotated_image.show()
    #             # 输入 Pillow 的 RGB 图片，输出 OpenCV 的 BGR
    #             output_frame = cv2.cvtColor(
    #                 np.asarray(annotated_image),
    #                 cv2.COLOR_RGB2BGR
    #             )
    #             writer.write(output_frame)
    #             frame_index += 1
    #     finally:
    #         capture.release()
    #         if writer is not None:
    #             writer.release()


def main():
    pipeline = InferencePipeline(DETECTOR_PATH, CLASSIFIER_PATH)
    print("分类模型的类别映射：", pipeline.classifier.names)

    util = ModelUtility()
    util.make_outputs_dir(IMAGE_OUTPUT_DIR, VIDEO_OUTPUT_DIR)

    # video_path = Path(r"C:\Users\AvaAtlas\Videos\Captures\test_video.mp4")
    # util.predict_video(pipeline, video_path)

    image_path = Path(r"C:\Users\AvaAtlas\Pictures\Screenshots\Screenshot 2026-10-25 125547.png")
    with Image.open(image_path) as image:
        print(pipeline.get_image_predictions(image))


if __name__ == "__main__":
    main()
