FROM python:3.11-slim-bookworm

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_AUTOINSTALL=false

# 当前 fork 声明的是 opencv-python，准备它在无桌面 Linux 上导入时需要的运行库。
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY ultralytics/ ./ultralytics/

# 安装这个 fork；lap 是当前 ByteTrack 代码需要的依赖，不能等首次推理时再下载。
RUN python -m pip install . "lap>=0.5.12" \
    && python -m pip check \
    && python -c "import cv2, torch, torchvision, lap; from PIL import ImageDraw"

COPY examples/traffic_gesture/ ./examples/traffic_gesture/

CMD ["python", "-m", "examples.traffic_gesture.inference.inference"]