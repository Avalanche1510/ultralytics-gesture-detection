# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Run YOLOv8 object detection on a webcam feed.

Usage:
    python examples/webcam_yolov8.py
    python examples/webcam_yolov8.py --camera 1 --conf 0.4
"""

import argparse

import cv2

from ultralytics import YOLO


def main():
    """Open a webcam, annotate each frame with YOLO detections, and show the result."""
    parser = argparse.ArgumentParser(description="Run YOLOv8 detection on a webcam.")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to a YOLO detection model.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index, usually 0 for the default camera.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum detection confidence.")
    parser.add_argument("--device", default="0", help="Inference device, for example 0 or cpu.")
    args = parser.parse_args()

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}. Try another --camera index.")

    model = YOLO(args.model)
    print("Camera opened. Press q or Esc in the video window to quit.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame from the camera.")

            result = model(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
            cv2.imshow("YOLOv8 webcam detection", result.plot())

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
