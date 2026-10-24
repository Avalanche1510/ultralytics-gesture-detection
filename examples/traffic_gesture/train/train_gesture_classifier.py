from pathlib import Path
from ultralytics import YOLO
from dataclasses import dataclass, asdict, replace
from typing import Optional
import torch
import json
import csv
from datetime import datetime
DATA_ROOT = Path(r"E:\datasets\gesture_reflective_classifier_v4_adjusted")
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MODEL_PATH = "yolov8n-cls.pt"
OUTPUT_ROOT = Path(r"E:\models\traffic_gesture_classifier")
CONFIG_FILENAME = "experiment_config.json"


@dataclass(frozen=True)  # 加上 @dataclass 后，Python 自动生成 __init__、易读的打印形式等。frozen=True 的含义是：一次创建后的配置不能再被代码改写。
class ClassificationTrainConfig:  # 记录实验配置：用什么数据、什么模型、输出到哪里、冻结多少层等等
    data_root: Path
    model_path: str
    output_root: Path
    experiment_name: str
    run_id: Optional[str] = None  # 不手动填写时，直接使用本次运行的时间戳
    image_size: int = 224
    freeze: Optional[int] = None  # Optional[int] 表示“这个值要么是整数，要么是 None”。
    config_filename = CONFIG_FILENAME  # 配置文件名

    # 默认的 baseline 图像增强配置
    fliplr: float = 0.0  # 水平翻转概率。
    # 默认不左右镜像。对 LEFT_TURN / RIGHT_TURN 必须关闭。

    flipud: float = 0.0  # 垂直翻转概率。
    # 默认不上下颠倒。真实交警画面不会倒置。

    scale: float = 0.0  # 随机裁切、缩放强度。
    # 默认不随机放大并截掉部分画面，避免裁掉手势。

    auto_augment: Optional[str] = None  # 自动组合多种增强策略。
    # 默认关闭 RandAugment 等自动变换。

    erasing: float = 0.0  # 随机遮挡一块区域的概率。
    # 默认不随机抹掉手、反光服或手势。

    hsv_h: float = 0.0  # 色相随机变化幅度。
    # 默认不改变颜色种类。

    hsv_s: float = 0.0  # 饱和度随机变化幅度。
    # 默认不改变颜色鲜艳程度。

    hsv_v: float = 0.0  # 明暗随机变化幅度。
    # 默认不随机变亮或变暗。

    # 默认的 baseline 训练配置
    epochs: int = 100
    patience: int = 20
    batch: float = 0.6  # 自动使用约 60% GPU 显存决定 batch 大小
    device: int = 0
    workers: int = 0  # Windows 下最稳定的加载方式
    seed: int = 42  # 某些速度更快、但每次结果可能略有不同的 GPU 算法不受该随机种子控制，故需要deterministic
    deterministic: bool = True  # 尽可能保证不同实验可复现
    exist_ok: bool = True

    def to_dict(self) -> dict:  # 目标是把 dataclass 的所有字段自动收集成字典，并把不能直接写进 JSON 的 Path 转成字符串。
        values = asdict(self)
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in values.items()
        }

    def save_to_json(self, config_path: Path) -> None:  # 将当前配置输出为json格式的配置文件
        config_path.parent.mkdir(parents=True, exist_ok=True)  # 若模型输出目录尚不存在，就创建它

        config_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    @classmethod
    def load_from_json(cls, config_path: Path) -> "ClassificationTrainConfig":  # 之所以类名注解用了双引号，是因为这个方法写在 ClassificationTrainConfig 类的内部；Python 正在定义这个类时，类名还没有完全创建好。
        values = json.loads(config_path.read_text(encoding="utf-8"))  # 返回dict字典

        # 只有json中的路径需要单独处理
        values["data_root"] = Path(values["data_root"])
        values["output_root"] = Path(values["output_root"])

        return cls(**values)

    @property  # 让方法的输出为一个属性
    def run_name(self) -> str:
        if self.run_id is None:
            return self.experiment_name
        return f"{self.experiment_name}_{self.run_id}"

    def for_new_run(self) -> "ClassificationTrainConfig":  # 由于@dataclass(frozen=True)禁止在创建后修改配置，直接返回新的配置对象
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return replace(self, run_id=timestamp)  # 返回新对象

    def to_ultralytics_kwargs(self) -> dict:  # 把配置转换成 Ultralytics 所需的参数字典
        return {
            "data": str(self.data_root),
            "epochs": self.epochs,
            "patience": self.patience,
            "batch": self.batch,
            "imgsz": self.image_size,
            "device": self.device,
            "workers": self.workers,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "freeze": self.freeze,
            "fliplr": self.fliplr,
            "flipud": self.flipud,
            "scale": self.scale,
            "auto_augment": self.auto_augment,
            "erasing": self.erasing,
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
            "project": str(self.output_root),
            "name": self.run_name,
            "exist_ok": self.exist_ok
        }


class GestureClassificationDataset:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def split_dir(self, split: str) -> Path:  # “ -> Path”是返回类型注解，表示这个方法预计返回一个 Path 对象
        if split in SPLITS:
            return self.data_dir / split
        else:
            raise ValueError(f'"{split}" is not a valid split type!')

    def class_dir(self, split: str, class_name: str) -> Path:
        if class_name in self.get_class_names(split):
            return self.split_dir(split) / class_name
        else:
            raise ValueError(f'"{class_name} is not a valid class name!"')

    def get_class_names(self, split: str) -> list:  # 返回指定数据集划分下的类别名称列表
        split_dir = self.split_dir(split)
        return sorted(
            path.name
            for path in split_dir.iterdir()
            if path.is_dir()
        )

    def count_images(self, split: str, class_name: str) -> int:  # 返回指定类别目录下的图像数量
        class_dir = self.class_dir(split, class_name)
        return sum(
            1
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )

    def get_source_keys(self, split: str) -> set:  # 返回指定数据集划分下的所有图像文件的唯一标识符集合
        return {
            image_path.stem.split("_jpg.rf.")[0]  # 返回图像id前缀
            for class_name in self.get_class_names(split)
            for image_path in self.class_dir(split, class_name).iterdir()
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES
        }

    def get_overlap_keys(self, split_a: str, split_b: str) -> set:
        return self.get_source_keys(split_a) & self.get_source_keys(split_b)


class TrainPipeline:
    def __init__(self, config: ClassificationTrainConfig, resume_run_dir: Optional[Path] = None):
        self.config = config
        self.resume_run_dir = resume_run_dir
        self.dataset = GestureClassificationDataset(self.config.data_root)
        print("Prepared run:", self.config.run_name)
        print("config in dictionary: ", self.config.to_dict())
        print("train_kwargs: ", self.config.to_ultralytics_kwargs())

        self.run_dir = None
        self.model = None
        self.config_path = None
        self.last_checkpoint_path = None
        self.completed_epoch = None

    @staticmethod
    def read_last_completed_epoch(results_path: Path) -> int:
        """读取 results.csv 中最后一个已完整记录的 epoch。"""
        with results_path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))

            if not rows:
                raise ValueError(f"no epoch record in results.csv!")
        return int(rows[-1]["epoch"])

    @classmethod
    def from_resume_run(cls, run_dir: Path) -> "TrainPipeline":
        """从一个未完成实验目录读取配置和断点信息。"""
        config_path = run_dir / CONFIG_FILENAME
        last_checkpoint_path = run_dir / "weights" / "last.pt"
        results_path = run_dir / "results.csv"

        for path in (config_path, last_checkpoint_path, results_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required file not found: {path}")

        config = ClassificationTrainConfig.load_from_json(config_path)
        pipeline = cls(config, resume_run_dir=run_dir)
        pipeline.run_dir = run_dir
        pipeline.config_path = config_path
        pipeline.last_checkpoint_path = last_checkpoint_path
        pipeline.completed_epoch = cls.read_last_completed_epoch(results_path)

        print(f"resume run from: {run_dir}")
        print(f"results.csv last completed epoch: {pipeline.completed_epoch}")
        print(f"the checkpoint path to be used: {pipeline.last_checkpoint_path}")
        return pipeline

    def validate_dataset(self) -> None:
        """确认 train、val、test 之间不存在同源增强样本泄漏。"""
        for split_a, split_b in (
                ("train", "val"),
                ("test", "val"),
                ("train", "test"),
        ):
            overlap_keys = self.dataset.get_overlap_keys(split_a, split_b)
            print(f"{split_a} ∩ {split_b}: {len(overlap_keys)}")
            if overlap_keys:
                raise RuntimeError(
                    f"Source key leakage detected between {split_a} and {split_b}!"
                    f"{len(overlap_keys)} frames overlap!"
                )

    def validate_runtime(self) -> None:
        """检查运行环境是否满足要求。"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable, but this experiment requires device=0.")

        if self.config.device >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {self.config.device} does not exist.")

        print("GPU:", torch.cuda.get_device_name(self.config.device))

    def prepare_run_directory(self) -> Path:
        """创建输出目录，并保存配置文件。"""
        run_dir = self.config.output_root / self.config.run_name

        if run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {run_dir}")

        # 在检查输出结果目录时，顺便提前输出配置文件
        self.config_path = run_dir / self.config.config_filename
        self.config.save_to_json(self.config_path)
        print("Configuration saved to:", self.config_path)
        self.run_dir = run_dir
        return run_dir

    def load_model(self, model_path: Path=None) -> YOLO:
        """加载模型并检查"""
        if model_path is None:
            model_path = self.config.model_path

        model = YOLO(model_path)

        for index, module in enumerate(model.model.model):
            print(index, module.__class__.__name__)
        self.model = model
        return model

    def train_classifier(self, resume: bool = False) -> None:
        """开始训练，或从 last.pt 恢复未完成的训练。"""
        if self.model is None:
            raise RuntimeError("Model is not loaded. Please call load_model() first.")

        if resume:
            return self.model.train(resume=True)

        return self.model.train(**self.config.to_ultralytics_kwargs())  # ** 会把配置字典展开成：model.train(data=..., epochs=..., imgsz=..., ...)

    def validate_loaded_config(self, config_path: Path = None) -> bool:
        """验证加载的配置文件是否与当前配置一致"""
        if config_path is None:
            if self.config_path is None:
                raise ValueError(
                    "Config path is not set. Please call prepare_run_directory() first."
                )
            config_path = self.config_path

        loaded_config = ClassificationTrainConfig.load_from_json(config_path)
        result = loaded_config == self.config

        print("Loaded config equals saved config:", result)
        return result

    def run(self) -> None:
        """执行完整的训练流程，或恢复一个未完成的训练。"""
        is_resuming = self.resume_run_dir is not None

        # 检查数据集划分是否存在泄漏
        self.validate_dataset()

        # 运行环境检查
        self.validate_runtime()

        # 输出配置文件并创建运行目录
        if not is_resuming:
            self.prepare_run_directory()

        # 测试读取配置
        if not self.validate_loaded_config():
            raise RuntimeError("Loaded config does not match the current config.")

        # 加载模型并检查
        if is_resuming:
            if self.last_checkpoint_path is None:
                raise RuntimeError("Resume checkpoint path is not set.")
            self.load_model(self.last_checkpoint_path)
        else:
            self.load_model()

        # 开始训练
        return self.train_classifier(resume=is_resuming)


def main():
    # 断点路径
    RESUME_DIR = Path(r"E:\models\traffic_gesture_classifier\baseline_100epc_no_online_augmentation_20261006_032754 ")

    # 加载实验配置
    # ctrl + / 以取消长段注释
    base_config = ClassificationTrainConfig(
        data_root=DATA_ROOT,
        model_path=MODEL_PATH,
        output_root=OUTPUT_ROOT,
        experiment_name="100epc_",
        epochs=100
    )
    print(base_config)

    # # 设置为新的运行配置，生成唯一的 run_id
    # config = base_config.for_new_run()
    #
    # # 创建训练流
    # pipeline = TrainPipeline(config)
    #
    # # 开始训练
    # pipeline.run()

    # 从断点恢复训练
    pipeline = TrainPipeline.from_resume_run(RESUME_DIR)
    pipeline.run()


if __name__ == "__main__":
    main()
