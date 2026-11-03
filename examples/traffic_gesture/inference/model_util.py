from datetime import datetime
import cv2
from pathlib import Path
import numpy as np
from PIL import Image, ImageGrab
from typing import TYPE_CHECKING
from examples.traffic_gesture.inference.defaults import VIDEO_INPUT_DIR, VIDEO_OUTPUT_DIR, IMAGE_INPUT_DIR, IMAGE_OUTPUT_DIR

if TYPE_CHECKING:
    from examples.traffic_gesture.inference.inference import InferencePipeline


class ModelUtility:
    """提供输出命名、剪贴板读取、视频读写及逐帧推理编排。"""
    def __init__(self):
        pass

    @staticmethod
    def set_output_path(output_dir, prefix: str = "IMAGE", suffix: str = "jpg") -> dict:
        """生成带时间戳的输出路径，不创建目录或文件。

        Args:
            output_dir (Path): 输出目录。
            prefix (str): 文件名前缀，默认为 "IMAGE"。
            suffix (str): 不带点的扩展名，默认为 "jpg"；视频使用 "mp4"。

        Returns:
            (dict): 时间戳和建议路径，结构如下::

                {
                    "timestamp": str,  # 格式 YYYYMMDD_HHMMSS_ffffff
                    "path": Path,      # output_dir / 前缀_时间戳.扩展名
                }

            保存文件时应取返回值的 ["path"]，不能把整个字典传给 save()。
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = Path(output_dir / f"{prefix}_{timestamp}.{suffix}")
        return {"timestamp": timestamp, "path": output_path}

    @staticmethod
    def make_outputs_dir(image_output_dir: Path = IMAGE_OUTPUT_DIR, video_output_dir: Path = VIDEO_OUTPUT_DIR, scope=None):
        """创建图片、视频输出目录及缺失的父目录，并打印状态。

        Args:
            image_output_dir (Path): 图片目录。
            video_output_dir (Path): 视频目录。
            scope (list[str], optional): ["image"] 仅检查 image_output_dir，["video"] 仅检查
                video_output_dir；None 表示两者都检查。

        Returns:
            (None): 不返回路径或文件。
        """
        # 检查图片输出路径是否存在并创建
        if scope is None:
            scope = ["image", "video"]

        if "image" in scope:
            if image_output_dir.exists():
                print(f"图片输出目录已存在于{image_output_dir}")
            else:
                print(f"成功创建图片输出目录于{image_output_dir}")
                image_output_dir.mkdir(parents=True)

        # 检查视频输出路径是否存在并创建
        if "video" in scope:
            if video_output_dir.exists():
                print(f"视频输出目录已存在于{video_output_dir}")
            else:
                print(f"成功创建视频输出目录于{video_output_dir}")
                video_output_dir.mkdir(parents=True)

    @staticmethod
    def make_inputs_dir(image_input_dir: Path = IMAGE_INPUT_DIR, video_input_dir: Path = VIDEO_INPUT_DIR, scope=None):
        """创建图片、视频输入目录及缺失的父目录，并打印状态。

        Args:
            image_input_dir (Path): 图片目录。
            video_input_dir (Path): 视频目录。
            scope (list[str]): 当前检查补全的范围，可以为"image"或"video"。

        Returns:
            (None): 不返回路径或文件。
        """
        # 检查图片输出路径是否存在并创建
        if scope is None:
            scope = ["image", "video"]

        if "image" in scope:
            if image_input_dir.exists():
                print(f"图片输入目录已存在于{image_input_dir}")
            else:
                print(f"成功创建图片输入目录于{image_input_dir}")
                image_input_dir.mkdir(parents=True)

        # 检查视频输出路径是否存在并创建
        if "video" in scope:
            if video_input_dir.exists():
                print(f"视频输入目录已存在于{video_input_dir}")
            else:
                print(f"成功创建视频输入目录于{video_input_dir}")
                video_input_dir.mkdir(parents=True)

    def read_clipboard_image(self, output_dir: Path = IMAGE_OUTPUT_DIR) -> tuple[Image.Image, Path]:
        """读取剪贴板图片并生成建议保存路径，不执行保存。

        Args:
            output_dir (Path): 建议 JPEG 路径的父目录，默认 IMAGE_OUTPUT_DIR。此方法不创建目录或文件。

        Returns:
            (tuple[Image.Image, Path]): 共两个值，按顺序返回::

                (
                    image,        # Image.Image：RGB 图片
                    output_path,  # Path：建议输出路径，图片尚未保存
                )

        Raises:
            ValueError: 剪贴板内容不是 Pillow 图片，包括仅复制了文件路径的情况。
        """

        content = ImageGrab.grabclipboard()

        if not isinstance(content, Image.Image):
            raise ValueError(
                "剪贴板中没有图片。请先用 Win+Shift+S 截图，再运行程序。"
            )

        output_path = self.set_output_path(output_dir)["path"]
        return content.convert("RGB"), output_path

    @staticmethod
    def read_video(video_path: Path, start_frame: int = 0) -> tuple[cv2.VideoCapture, dict]:
        """打开本地视频并定位下一次读取的位置，不读取或返回图像帧。

        Args:
            video_path (Path): 输入视频文件。
            start_frame (int): 从 0 开始的帧索引；500 表示下一次 read() 读取第 501 帧。

        Returns:
            (tuple[cv2.VideoCapture, dict]): 读取器及元数据，结构如下::

                (
                    capture,  # cv2.VideoCapture：已打开并定位的读取器
                    {
                        "fps": float,        # 视频报告的帧率
                        "width": int,        # 帧宽度，像素
                        "height": int,       # 帧高度，像素
                        "frame_count": int,  # 声明帧数，可能未知或不准确
                    },
                )

            调用者取得读取器后负责调用 release()。

        Raises:
            FileNotFoundError: 视频路径不存在。
            ValueError: 起始索引为负，或超出已知声明帧数。
            RuntimeError: 无法打开视频或无法定位。
        """
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
        """创建 MP4 写入器，不在此处循环写入视频帧。

        Args:
            output_dir (Path): 实际写入目录；若不存在则连同父目录一起创建。文件名为 VIDEO_时间戳.mp4。
            capture_info (dict): 需含 "fps" (float)、"width" (int)、"height" (int)，
                通常来自 read_video；本方法不使用 "frame_count"。

        Returns:
            (cv2.VideoWriter): 已打开的 mp4v 写入器；调用者需按配置尺寸写入 BGR 数组，
                并在结束时调用 release()。不返回输出路径，不处理音轨。

        Raises:
            ValueError: 帧率或尺寸无效。
            RuntimeError: 无法创建写入器。
        """
        self.make_outputs_dir(video_output_dir=output_dir, scope=["video"])

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

    def predict_video(self, pipeline: "InferencePipeline", video_path: Path, save_dir: Path = VIDEO_OUTPUT_DIR, smoothing_params: dict = None, start_frame=0, end_frame=-1, use_detector=True, progress=True):
        """逐帧识别并保存标注视频，处理索引范围 [start_frame, end_frame)。

        Args:
            pipeline (InferencePipeline): 复用的模型管线；启用检测时会在本次视频开始时重置跟踪器。
            video_path (Path): 输入视频路径。
            save_dir (Path): 标注视频的实际输出目录，默认 VIDEO_OUTPUT_DIR；创建写入器时按需建目录。
            smoothing_params (dict, optional): None 表示不平滑；启用时传
                {"method": "EMA", "alpha": 0.5}，键含义见 update_confidences。
            start_frame (int): 首个处理帧的零基索引，包含该帧。
            end_frame (int): 结束索引，不包含该帧；-1 表示持续读取到 read() 失败。
            use_detector (bool): True 为人员跟踪后逐人分类；False 为整图分类，使用固定键 -1。
            progress (bool): 是否打印处理进度。

        Returns:
            (None): 不返回预测或路径；视频写入 save_dir。
                输出使用原视频报告的帧率和尺寸，不保留音轨。

        Notes:
            内部 predictions/history_predictions 为 {ID: prediction}，字段见 get_image_predictions。
            EMA 仅保留上一帧历史：目标缺席一帧后重新出现会重新初始化，不跨漏检继承历史。
            无目标的帧仍写入；读取失败即停止，当前不区分正常结尾和解码失败。
            方法通过 finally 释放读取器和成功创建的写入器。
        """
        if end_frame != -1 and end_frame <= start_frame:
            raise ValueError("end_frame 必须大于 start_frame，或为 -1。")

        capture, capture_info = self.read_video(video_path, start_frame=start_frame)  # 视频帧同样是从0开始，到n-1帧结束
        writer = None
        try:
            if use_detector:
                pipeline.detector.reset_tracking()

            writer = self.get_video_writer(save_dir, capture_info)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_index = start_frame

            history_predictions = {}

            while end_frame == -1 or frame_index < end_frame:
                success, frame = capture.read()
                if not success:
                    break
                if progress:
                    if end_frame == -1:
                        print(f"已处理 {frame_index - start_frame + 1} 帧，预期需要{frame_count}帧")
                    else:
                        print(f"当前处于第{frame_index+1}/{frame_count}帧，进度{round((frame_index - start_frame + 1) / (end_frame - start_frame) * 100, 1)}%")

                # OpenCV 的 BGR 数组 → Pillow 的 RGB 图片
                image = Image.fromarray(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )
                predictions = pipeline.get_image_predictions(
                    image,
                    use_detector=use_detector,
                    display=False,
                    tracking=True,
                )

                # 平滑predictions
                if smoothing_params is not None:
                    predictions = pipeline.smoothing(predictions, history_predictions, smoothing_params)
                    history_predictions = predictions

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


