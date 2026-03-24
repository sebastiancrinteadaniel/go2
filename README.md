# Go2 Dashboard (c2-go2)

A FastAPI + WebRTC control interface for the Unitree Go2 quadruped robot. Streams live video, displays system telemetry, and supports hand gesture control via MediaPipe + ONNX inference.

Designed to run on a **Jetson Orin Nano (8 GB)** mounted on the robot, accessible from any browser over the local network.

---

## Target Hardware

### Jetson Orin Nano 8GB (compute module)

| Spec | Value |
|------|-------|
| AI performance | 40 TOPS |
| GPU | 1024-core NVIDIA Ampere + Tensor Cores @ 625 MHz |
| CPU | 6-core ARM Cortex-A78AE v8.2 64-bit @ 1.5 GHz |
| Memory | 8 GB 128-bit LPDDR5 — 68 GB/s |
| DL Accelerator | — (no DLA on Nano variant) |
| Video encode | 1080p30 (1–2 CPU cores) |
| Video decode | 1× 4K60, 5× 1080p60, 11× 1080p30 (H.265 HW) |
| Power | 7 W – 15 W |
| Storage | External NVMe |

> No DLA on the Nano — ONNX inference runs on the Ampere GPU via CUDA or falls back to CPU.

### Unitree Go2 head camera

| Spec | Value |
|------|-------|
| Resolution | 1080p or 720p |
| Framerate | **15 fps** (both modes) |
| Aperture | F2.2 |
| Field of view | 120° |
| Transmission | 720p 15fps in clear/unobstructed conditions |

> The Go2 camera is capped at **15 fps**. `CAMERA_FPS = 30` in config only applies to an external webcam (`CameraStreamTrack`). The `Go2CameraStreamTrack` pulls frames as fast as `GetImageSample()` delivers them, which tops out at 15 fps.

---

## Architecture

```
Browser
  └─ POST /offer (SDP) → FastAPI
       └─ WebRTC peer connection
            ├─ Video track → CameraStreamTrack / Go2CameraStreamTrack
            │    ├─ Thread 1: frame capture (blocking I/O)
            │    ├─ Thread 2: inference (MediaPipe → ONNX classifier)
            │    └─ GestureDispatcher → Unitree SportClient commands
            └─ Data channel → telemetry loop
                 ├─ psutil  (CPU, RAM, uptime)
                 └─ Unitree DDS (battery, motor temps)
```

| Path | Purpose |
|------|---------|
| `app/api/routes.py` | WebRTC `/offer` endpoint and data channel loop |
| `app/services/video.py` | `CameraStreamTrack` (webcam) and `Go2CameraStreamTrack` (SDK) |
| `app/services/gesture_processor.py` | MediaPipe hand detection + ONNX keypoint classifier |
| `app/services/gesture_dispatcher.py` | Maps gestures to robot actions with cooldown throttling |
| `app/services/telemetry.py` | Robot state monitoring via Unitree DDS |
| `app/services/yolo_processor.py` | YOLOv8n object detection (disabled by default) |
| `app/core/config.py` | Pydantic Settings — camera, gesture thresholds, cooldowns |
| `app/models/` | ONNX gesture classifier + YOLOv8n weights |
| `app/static/` | Frontend: `index.html`, `main.js`, `style.css` |
| `example/` | Standalone SDK examples and gesture training scripts |

---

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- `unitree_sdk2py` — optional, required for Go2 camera and robot control (graceful fallback to webcam mode without it)
- CUDA / ONNX Runtime GPU — optional, falls back to CPU inference automatically

### Installation

```bash
git clone <repo-url>
cd c2_go2
uv sync
```

### Run

```bash
uv run python app/main.py
```

Open `http://<device-ip>:8000` in your browser.

---

## Configuration

Settings are loaded from a `.env` file (or environment variables) via Pydantic Settings. Defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `CAMERA_WIDTH` | `1920` | Capture width (px) |
| `CAMERA_HEIGHT` | `1080` | Capture height (px) |
| `CAMERA_FPS` | `30` | Target capture framerate |
| `GESTURE_DISPATCH_MIN_CONFIDENCE` | `0.75` | Minimum classifier confidence to consider a gesture |
| `GESTURE_DISPATCH_MIN_STABLE_FRAMES` | `3` | Consecutive matching frames before dispatch |
| `GESTURE_DISPATCH_COOLDOWN` | `2.0` | Per-gesture cooldown (seconds) |
| `GESTURE_DISPATCH_GLOBAL_COOLDOWN` | `2.0` | Global cooldown between any two dispatches (seconds) |

---

## Gesture Control

### Recognized gestures

The ONNX keypoint classifier recognizes 10 hand poses. MediaPipe first detects hand landmarks, then the classifier runs on the normalized 21-point skeleton — so lighting and skin tone don't matter, only the shape of your hand.

| Label | Shape | How to form it |
|-------|-------|----------------|
| `Open` | 🖐️ | All five fingers spread wide open |
| `Closed` | ✊ | Full fist — all fingers curled in, thumb across |
| `Like` | 👍 | Thumbs up — fist with thumb pointing straight up |
| `Dislike` | 👎 | Thumbs down — fist with thumb pointing straight down |
| `PeaceSign` | ✌️ | Index + middle fingers up, other fingers folded, thumb tucked |
| `One` | ☝️ | Only index finger pointing up, others curled |
| `Three` | --- | Thumb + pinky folded |
| `Four` | --- | All four fingers up, thumb folded in across palm |
| `FingerHeart` | 🫰 | Thumb and index finger crossed to form a small heart |
| `HeartHalf` | 🫶 | One hand curved inward — combine both hands' HeartHalf to make a full ❤️ |

> The model is trained on right-hand poses. Left hands are automatically mirrored before classification so the same gestures work on both hands.

### Robot actions

When `unitree_sdk2py` is available, five of the recognized gestures are wired to robot commands:

| Gesture | Emoji | Robot action |
|---------|-------|-------------|
| `Like` | 👍 | Stand up + free walk |
| `Dislike` | 👎 | Stop + stand down |
| `PeaceSign` | ✌️ | Hello wave |
| `FingerHeart` | 🫰 | Heart pose |
| `HeartHalf` × 2 | 🫶🫶 | Heart pose (both hands detected simultaneously) |

`Open`, `Closed`, `One`, `Three`, and `Four` are classified and shown on screen but do not currently trigger any robot action — they are available for extension.

A gesture must be held stable for `GESTURE_DISPATCH_MIN_STABLE_FRAMES` frames at or above `GESTURE_DISPATCH_MIN_CONFIDENCE` before it fires. Both a per-gesture and a global cooldown prevent repeated triggers.

---

## Connecting to the Robot

### Via Ethernet
```bash
ssh unitree@192.168.123.18
```

### Via WiFi
```bash
ssh unitree@<robot-wifi-ip>
```
