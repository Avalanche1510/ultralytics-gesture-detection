<details>
<summary>中文｜Docker 交警手势视频推理部署指南</summary>

# Docker：交警手势视频推理

目标设备为 Raspberry Pi 5（64 位 Linux / ARM64）。本指南处理已有 MP4，不读取 CSI 摄像头，不需要训练集或设备特权。Windows 本地管线已能输出视频；Pi 上的容器构建、推理速度和实时性仍待验证。

请使用包含此部署目录的 `codex/gesture-pipeline` 分支。部署文件尚未合并到 `main`。以下 Linux 命令读取宿主机上的 MP4，并将标注视频写回宿主机；镜像构建成功不等于推理验证通过。

### 准备

1. 在 Pi 安装 [Docker Engine 和 Compose 插件](https://docs.docker.com/engine/install/debian/)，确认下面三条命令都能成功。
2. 从包含此部署目录的仓库根目录进入 `examples/traffic_gesture/deploy`。两份推理用的 `best.pt` 位于同一工作树的 [results](../results/) 下，会随镜像复制，无需另外下载。
3. 准备 5–10 秒、可解码且有交警清楚出镜的测试 MP4，以便同时验证检测、分类和绘图。

```bash
docker --version
docker compose version
docker info --format '{{.Architecture}}'
cd examples/traffic_gesture/deploy
```

### 构建并运行

将输入路径换成已存在的真实文件。主机文件名不必叫 `test_video.mp4`：Compose 将它只读挂载为容器入口期待的文件名。输出目录是另一个可写挂载，删除容器不会删除结果。

```bash
export INPUT_VIDEO="$HOME/videos/demo.mp4"
export OUTPUT_DIR="$HOME/traffic-gesture-output"
test -f "$INPUT_VIDEO" || { echo "输入视频不存在：$INPUT_VIDEO"; exit 1; }
mkdir -p "$OUTPUT_DIR"

docker compose config
docker compose build
docker compose run --rm inference
ls -lh "$OUTPUT_DIR"/VIDEO_*.mp4
```

**构建路径检查：** 在 `docker compose config` 的输出中，`build.context` 必须指向包含 `pyproject.toml` 和 `ultralytics/` 的仓库根目录。相对于本目录是 `../../..`；`dockerfile` 则相对于上下文，为 `examples/traffic_gesture/deploy/Dockerfile`。上下文不对时先修正，不要在 Pi 上开始下载依赖。

### 验收

- 在宿主机打开本次新生成的 `VIDEO_*.mp4`，检查可播放、时长和尺寸与输入相符。此写入方式不保留音轨。
- 在有检测目标的帧上检查检测框、扩边框、轨迹 ID、手势名及分数。当前入口使用 PSEMA，alpha 为 0.2。无目标时应保留视频帧；只有空场景不能验证分类器。
- 记录设备架构、输入分辨率、帧数和实际处理耗时；输出视频的播放帧率不是模型吞吐量。
- [Dockerfile](Dockerfile) 使用 Python 3.11 并在构建时安装此 fork 及跟踪依赖。当前基础镜像和依赖版本未完全锁定：若要精确复现实测环境，请保留验证过的镜像和依赖清单。未验证的 CPU 架构、Pi Camera 和硬件控制不在本次验收范围。

</details>

<details>
<summary>English | Docker deployment for traffic gesture video inference</summary>

# Docker: Traffic gesture video inference

This guide targets a Raspberry Pi 5 running 64-bit Linux (ARM64). It takes an existing MP4 on the host and writes an annotated copy back to the host. It does not require a training dataset, camera access, or a privileged container. The pipeline has produced videos on Windows; the container build, inference speed, and real-time performance on the Pi have not yet been verified.

### Prepare

1. Use a checkout of `codex/gesture-pipeline` that contains this deployment directory. The deployment files have not yet been merged into `main`.
2. Install [Docker Engine and the Compose plugin](https://docs.docker.com/engine/install/debian/) on the Pi. From the repository root, run the commands below. The two `best.pt` checkpoints under [results](../results/) are copied into the image during the build; no separate model download is needed.
3. Choose a playable 5–10 second MP4 with a traffic officer clearly visible. An empty scene can test video decoding and writing, but not gesture classification.

```bash
docker --version
docker compose version
docker info --format '{{.Architecture}}'
cd examples/traffic_gesture/deploy
```

### Build and run

Set `INPUT_VIDEO` to the absolute path of your actual video. Its name on the host does not matter: Compose mounts it read-only as `test_video.mp4` inside the container. Mount a separate writable output directory so the result remains on the host after the container exits.

```bash
export INPUT_VIDEO="$HOME/videos/demo.mp4"
export OUTPUT_DIR="$HOME/traffic-gesture-output"
test -f "$INPUT_VIDEO" || { echo "Input video not found: $INPUT_VIDEO"; exit 1; }
mkdir -p "$OUTPUT_DIR"
docker compose config
```

Before building, check the resolved configuration: `build.context` must point to the repository root containing `pyproject.toml` and `ultralytics/` (`../../..` from this directory). The Dockerfile path is relative to that root: `examples/traffic_gesture/deploy/Dockerfile`. Check both host mount paths as well. Once they are correct, run:

```bash
docker compose build
docker compose run --rm inference
ls -lh "$OUTPUT_DIR"/VIDEO_*.mp4
```

### Verify the result

- Open the newly created `VIDEO_*.mp4` on the host. It should play at the input dimensions and approximately the input duration. This writer does not preserve audio.
- On frames with a detected officer, check the detection box, expanded crop box, track ID, gesture name, and score. The entry point uses PSEMA with `alpha=0.2`. Frames with no detections should still appear in the output.
- Record the device architecture, input resolution, frame count, and elapsed processing time. The playback frame rate stored in the output is not a measurement of inference throughput.
- The [Dockerfile](Dockerfile) uses Python 3.11 and installs this fork and its tracking dependencies at build time. The base image and dependencies are not fully pinned. Keep the tested image and dependency list to reproduce a measured run. Other architectures, live Pi Camera input, and hardware control are outside this test.

</details>
