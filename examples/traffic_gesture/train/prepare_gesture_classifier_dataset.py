from pathlib import Path
from PIL import Image
from math import ceil, floor
from ultralytics import YOLO
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional
from typing import Union

# 常量规范：写为大写字母，单词间用下划线分隔
# 基础数据集路径
SOURCE_ROOT = Path(r"E:\datasets\gesture(DST1556)")
# 交警识别器权重路径
MODEL_PATH = Path(r"E:\models\traffic_police\filteredV1_traffic_police_200epc\weights\best.pt")

# 裁剪图像的边缘扩大比例，此处由于通常不存在比帽子高特别多的手势，因此需要在横轴X上多扩张一些，确保不会截断手部。
PAD_X = 0.30
PAD_Y = 0.10
# 测试时只处理前多少张图片，-1表示全部处理
HEAD = -1

OUTPUT_SPLITS = {
    "train": "train",
    "valid": "val",
    "test": "test",
}

LETTERBOX_SIZE = 640
PAD_COLOR = (114, 114, 114)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
# candidate版本 表示经过交警检测器筛选出的候选图片，后续还需要人工审核，审理完成的数据集为 adjusted版本
CANDIDATE_ROOT = Path(
    r"E:\datasets\gesture_reflective_classifier_v4_candidate"
)
DETECTOR_CONFIDENCE = 0.68


@dataclass(frozen=True)
class TrainPipelineConfig:
    """记录从原始 YOLO 数据生成待审核 v4 候选数据集的全部固定参数。"""

    source_root: Path
    output_root: Path
    model_path: Path
    detector_confidence: float
    crop_pad_x: float
    crop_pad_y: float
    letterbox_size: int
    pad_color: tuple[int, int, int]


# 定义手势类别枚举
class GestureClass(IntEnum):
    """手势类别枚举"""
    STOP = 0
    MOVE_STRAIGHT = 1
    LEFT_TURN = 2
    RIGHT_TURN = 3
    LANE_CHANGING = 4
    SLOW_DOWN = 5
    PULL_OVER = 6
    LEFT_TURN_WAITING = 7


@dataclass(frozen=True)
class LetterboxTransformer:
    """将任意图片等比例缩放并补边为正方形，不拉伸人物比例。"""

    size: int
    pad_color: tuple[int, int, int]

    def apply(self, image: Image.Image) -> Image.Image:
        """把长边缩放到目标尺寸，再将短边的空缺部分以底色居中补齐。"""
        image = image.convert("RGB")
        source_width, source_height = image.size
        # 取两个缩放比例中较小的一个，保证完整原图都能放入正方形画布。
        scale = min(self.size / source_width, self.size / source_height)
        resized_width = round(source_width * scale)
        resized_height = round(source_height * scale)
        resized_image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

        # 先创建具有底色的目标尺寸正方形画布，再将图像居中放置。
        canvas = Image.new("RGB", (self.size, self.size), self.pad_color)
        offset_x = (self.size - resized_width) // 2
        offset_y = (self.size - resized_height) // 2
        canvas.paste(resized_image, (offset_x, offset_y))
        return canvas


@dataclass(frozen=True)
class GestureCropTransformer:
    """将标注框或检测框扩大并裁切为手势分类所需的图像。"""

    pad_x: float
    pad_y: float

    def yolo_to_xyxy(self, box: tuple[int, float, float, float, float], image_size: tuple[int, int]):
        """将 YOLO 的“类别、中心比例、宽高比例”转换为像素 xyxy 坐标。"""
        class_id, xc, yc, box_width, box_height = box
        image_width, image_height = image_size
        return (
            class_id,
            (xc - box_width / 2) * image_width,
            (yc - box_height / 2) * image_height,
            (xc + box_width / 2) * image_width,
            (yc + box_height / 2) * image_height,
        )

    def expand_clip_xyxy(self, xyxy: tuple[float, float, float, float], image_size: tuple[int, int]):
        """按横纵比例扩大 xyxy 框，并将结果裁剪在原始图像边界内。"""
        x1, y1, x2, y2 = xyxy
        image_width, image_height = image_size
        box_width = x2 - x1
        box_height = y2 - y1
        return (
            max(0, floor(x1 - box_width * self.pad_x)),
            max(0, floor(y1 - box_height * self.pad_y)),
            min(image_width, ceil(x2 + box_width * self.pad_x)),
            min(image_height, ceil(y2 + box_height * self.pad_y)),
        )

    def crop(self, image: Image.Image, xyxy: tuple[float, float, float, float]) -> Image.Image:
        """对检测器框或标注框执行扩大与裁切，返回一张新的 Pillow 图片。"""
        return image.crop(self.expand_clip_xyxy(xyxy, image.size))


class TrafficPoliceDetector:
    """加载反光服交警检测器，并输出可供离线筛选或实时推理复用的检测框。"""

    def __init__(self, model_path: Path, confidence: float = 0.68):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, source: Union[Path, Image.Image]) -> list[tuple[tuple[float, float, float, float], float]]:
        """对一张图片检测，返回通过置信度阈值的 xyxy 框及其分数。"""
        result = self.model(source, conf=self.confidence, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        return [(tuple(box), confidence) for box, confidence in zip(boxes, confidences)]


class TrainPipeline:
    """将人工审核后的分类数据处理为可供训练使用的最终数据集。"""
    # 流程：
    # 0. 原始训练数据集中，包含了诸多具有同一个source key的增强样本组。
    # 1. 使用交警检测器直接筛选 source key，剔除不含交警的样本组。

    def __init__(self, config: TrainPipelineConfig):
        self.config = config
        self.crop_transformer = GestureCropTransformer(
            pad_x=config.crop_pad_x,
            pad_y=config.crop_pad_y,
        )
        self.letterbox_transformer = LetterboxTransformer(
            size=config.letterbox_size,
            pad_color=config.pad_color
        )
        self._detector: Optional[TrafficPoliceDetector] = None

    @property
    def detector(self) -> TrafficPoliceDetector:
        if self._detector is None:
            self._detector = TrafficPoliceDetector(
                model_path=self.config.model_path,
                confidence=self.config.detector_confidence,
            )
        return self._detector

    @staticmethod
    def get_source_key(label_path: Path) -> str:
        """返回共享视频帧 key，使同一原图的增强兄弟样本始终一起保留或剔除。"""
        return label_path.stem.split("_jpg.rf.")[0]

    def get_label_paths(self, source_split: str) -> list[Path]:
        """返回一个原始 YOLO 切分的标注文件，排除 classes.txt。"""
        labels_dir = self.config.source_root / "labels" / source_split

        if not labels_dir.is_dir():
            raise FileNotFoundError(f"Missing label directory: {labels_dir}")

        return sorted(
            path
            for path in labels_dir.glob("*.txt")
            if path.name != "classes.txt"
        )

    @staticmethod
    def read_yolo_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
        """读取 YOLO 标签：类别 ID、中心 x/y 比例、宽/高比例。"""
        return [
            (int(class_id), float(xc), float(yc), float(width), float(height))
            for class_id, xc, yc, width, height in (
                line.split() for line in label_path.read_text(encoding="utf-8").splitlines()
            )
        ]

    def select_label_paths(self, source_split: str, head: int = HEAD) -> list[Path]:
        """返回本次要处理的标签文件；-1 表示处理全部。"""
        label_paths = self.get_label_paths(source_split)

        if head == -1:
            return label_paths
        elif head <= 0:
            raise ValueError(f"head must be -1 or positive, got {head}")
        else:
            # 从后往前处理，前面的样本通常不是交警。
            return label_paths[-head:]

    def get_source_image_path(self, source_split: str, label_path: Path) -> Path:
        """返回与标注文件对应的原始图像路径。"""
        image_path = (
            self.config.source_root
            / "images"
            / source_split
            / f"{label_path.stem}.jpg"
        )

        if not image_path.is_file():
            raise FileNotFoundError(f"Missing source image: {image_path}")
        return image_path

    def get_accepted_source_keys(self, source_split: str, head: int = HEAD) -> set[str]:
        """用交警检测器筛选 source key；裁切位置仍不使用检测器框。"""
        accepted_keys = set()

        for label_path in self.select_label_paths(source_split, head):
            image_path = self.get_source_image_path(source_split, label_path)
            detections = self.detector.detect(image_path)

            for box, confidence in detections:
                print("detector:", box, label_path.name, f"conf={confidence:.2f}")

            if detections:
                accepted_keys.add(self.get_source_key(label_path))

        return accepted_keys

    def get_reshaped_example(self, source_split: str, label_path: Path, mode: str = "preview") -> int:
        """对一个原始标注文件生成 letterbox 后的裁切预览。"""
        image_path = self.get_source_image_path(source_split, label_path)
        saved_count = 0

        with Image.open(image_path) as image:
            for box_index, box in enumerate(self.read_yolo_boxes(label_path)):
                class_id, *xyxy = self.crop_transformer.yolo_to_xyxy(box, image.size)
                # 裁剪边框
                crop = self.crop_transformer.crop(image, tuple(xyxy))
                # letterbox 正方形处理
                prepared_image = self.letterbox_transformer.apply(crop)
                # 输出路径并创建目录
                if mode == "preview":
                    output_path = self.config.output_root / "_preview" / source_split / OUTPUT_SPLITS[str(GestureClass(class_id).name)] / f"{label_path.stem}_box{box_index}_class{class_id}.jpg"
                else:
                    output_path = self.config.output_root / source_split / GestureClass(class_id).name / f"{label_path.stem}_box{box_index}_class{class_id}.jpg"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                # 保存图片
                prepared_image.save(output_path, quality=95)
                saved_count += 1
        print(f"{saved_count} {mode} reshaped images are saved to: {output_path}")
        return saved_count
    
    def get_reshaped_batch_by_split(self, source_split: str):
        accepted_keys = self.get_accepted_source_keys(source_split)
        for label_path in self.get_label_paths(source_split):
            if self.get_source_key(label_path) in accepted_keys:
                self.get_reshaped_example(source_split, label_path, mode="batch")

    def get_sibling_label_paths(self, source_split: str, source_key: str) -> list[Path]:
        """返回同一原始帧对应的全部增强兄弟标签。"""
        # 性能开销大，仅在测试时调用
        return [
            label_path
            for label_path in self.get_label_paths(source_split)
            if self.get_source_key(label_path) == source_key
        ]


def main():
    config = TrainPipelineConfig(
        source_root=SOURCE_ROOT,
        output_root=CANDIDATE_ROOT,
        model_path=MODEL_PATH,
        detector_confidence=DETECTOR_CONFIDENCE,
        crop_pad_x=PAD_X,
        crop_pad_y=PAD_Y,
        letterbox_size=LETTERBOX_SIZE,
        pad_color=PAD_COLOR
    )

    pipeline = TrainPipeline(config)

    print(pipeline.config)
    print(pipeline.crop_transformer)
    print(pipeline.letterbox_transformer)

    for source_split in OUTPUT_SPLITS:
        print(source_split, len(pipeline.get_label_paths(source_split)))

    for source_split in OUTPUT_SPLITS:
        pipeline.get_reshaped_batch_by_split(source_split)


if __name__ == "__main__":
    main()
