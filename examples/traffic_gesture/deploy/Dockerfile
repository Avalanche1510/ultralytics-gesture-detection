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

# 先从官方 CPU 索引安装固定版本；ARM64 的 torchvision 版本名不带 +cpu。
RUN python -m pip install --only-binary=:all: \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.7.1+cpu" "torchvision==0.22.1"

# 安装这个 fork；lap 是当前 ByteTrack 代码需要的依赖，不能等首次推理时再下载。
RUN python -m pip install . "lap>=0.5.12" "torch==2.7.1+cpu" "torchvision==0.22.1" \
    && python -m pip check \
    && python -c "import cv2, torch, torchvision, lap; from PIL import ImageDraw; assert torch.version.cuda is None, 'CPU-only PyTorch required'; assert torchvision.ops.nms(torch.tensor([[0., 0., 10., 10.]]), torch.tensor([0.9]), 0.5).tolist() == [0]; print('CPU torch:', torch.__version__, 'torchvision:', torchvision.__version__)"

COPY examples/traffic_gesture/ ./examples/traffic_gesture/

CMD ["python", "-m", "examples.traffic_gesture.inference.inference"]
