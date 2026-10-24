from pathlib import Path
from PIL import Image
from math import ceil, floor
from ultralytics import YOLO
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"

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

print(DETECTOR_PATH, DETECTOR_PATH.is_file())
print(CLASSIFIER_PATH, CLASSIFIER_PATH.is_file())
