from pathlib import Path

INFERENCE_DIR = Path(__file__).resolve().parent   # inference/
PROJECT_ROOT = INFERENCE_DIR.parent              # traffic_gesture/
RESULTS_ROOT = PROJECT_ROOT / "results"

IMAGE_INPUT_DIR = INFERENCE_DIR / "inputs" / "images"
VIDEO_INPUT_DIR = INFERENCE_DIR / "inputs" / "videos"
IMAGE_OUTPUT_DIR = INFERENCE_DIR / "outputs" / "images"
VIDEO_OUTPUT_DIR = INFERENCE_DIR / "outputs" / "videos"

FONT_SIZE = 36

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
