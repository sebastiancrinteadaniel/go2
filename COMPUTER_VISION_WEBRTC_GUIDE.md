# Computer Vision and WebRTC Overview

## Core Architecture

The system integrates multiple computer vision models with WebRTC to enable robot control both locally on the Jetson Nano and remotely via laptop. A modular design allows detection, gesture recognition, and depth sensing to work seamlessly across different sources—whether capturing from the robot's camera, a WebRTC stream, or a local USB camera.

The camera abstraction layer decouples vision modules from their input source. YOLO object detection, MediaPipe hand gesture recognition, and RealSense depth processing all work identically regardless of whether frames come from the robot's VideoClient, a WebRTC peer connection, or an OpenCV stream.

```python
# Camera abstraction - same interface for all sources
class BaseCamera:
    def read(self) -> Frame:
        raise NotImplementedError

# Works with OpenCV, video files, Go2 camera, or WebRTC
camera = create_camera(source="go2", width=640, height=480)
frame = camera.read()
```

## Computer Vision Modules

### YOLO Object Detection

Uses YOLOv8 nano for real-time detection on resource-constrained hardware. Key design features:

- **Threaded pipeline**: Separates capture, inference, and rendering into parallel threads
- **Bounded queues**: Frames between stages; newer frames take priority when processing falls behind
- **Configurable parameters**: Confidence threshold, input resolution, worker threads

The threaded approach maintains responsiveness—while one thread waits for GPU inference, another captures the next frame. Bounded queues mean frame drops under load rather than latency increases.

```python
CONFIG = {
    "yolo": {
        "model": "src/yolo/model/yolov8n.pt",
        "conf": 0.25,
        "imgsz": 320,
        "enable_threads": True,
        "workers": 1,
        "queue_size": 3,
        "drop_if_full": True,  # Prioritizes recent frames
    },
}
```

### Hand Gesture Recognition

Combines MediaPipe for hand pose estimation with a secondary classification network that interprets keypoints as specific gestures. Key features:

- **Two-stage approach**: Hand detection → keypoint extraction → gesture classification
- **Multi-hand support**: Up to 10 hands detected simultaneously
- **Gesture dispatch**: Automatically translates recognized gestures into robot commands
- **Cooldown mechanism**: Prevents rapid re-triggering from natural hand jitter

The gesture dispatcher bridges perception to robot control. When a gesture is recognized, it can trigger SportClient commands like stand, sit, or dance.

```python
CONFIG = {
    "mediapipe": {
        "max_num_hands": 10,
        "min_detection_confidence": 0.7,
        "min_tracking_confidence": 0.5,
    },
    "gestures": {
        "enable_dispatch": True,
        "cooldown": 2.0,  # Seconds between same gesture repeats
    },
}
```

### Depth Sensing

Applies object detection to RGB frames while using depth data to estimate distance to detected objects.

- **Combined modalities**: YOLO detections + depth estimates = spatial understanding
- **Noise filtering**: Contour detection and minimum area thresholds
- **Configurable thresholds**: Depth distance and object size for region of interest

```python
CONFIG = {
    "depth_camera": {
        "width": 640,
        "height": 480,
        "fps": 30,
        "model_path": "src/depth_camera/model/yolov8n.pt",
        "depth_threshold_m": 0.5,  # Objects closer than 0.5m
        "min_contour_area": 1000,
    }
}
```

## WebRTC for Remote Control

WebRTC establishes peer-to-peer video and data channels between the robot and remote machines, providing low-latency bidirectional communication suitable for real-time control.

### Connection Methods

- **LocalSTA**: Both devices on same WiFi network (requires robot IP)
  - Development and testing with powerful laptop GPU
  - Local network only (no cloud required)

- **LocalAP**: Robot creates its own WiFi access point
  - Standalone operation without existing WiFi
  - No infrastructure dependencies

- **Remote**: Cloud-brokered connection for different networks
  - Robot registers with cloud using serial number + credentials
  - Enables robot fleets and remote troubleshooting

```python
# Configure connection method in webrtc/config.py
CONFIG = {
    "webrtc": {
        "source": "DDS",  # or "WebRTC"
        "method": "LocalSTA",  # LocalSTA, LocalAP, or Remote
        "ip": "192.168.123.18",
        "width": 640,
        "height": 480,
    }
}
```

Unlike traditional HTTP streaming, WebRTC automatically adapts compression and bitrate to network conditions, maintaining consistent frame delivery with latency typically in hundreds of milliseconds.

## Deployment Models

### Onboard Deployment (Jetson Nano)

Run the entire system directly on the robot using DDS (native Unitree protocol).

```bash
# Ultra-low latency communication via DDS
python3 -m src.dashboard.app eth0
```

- All vision and control on robot hardware
- DDS provides millisecond-level latency
- Limited by Jetson's computational power
- Best for autonomous operation

### Laptop Remote Control

Connect a laptop to the robot via WebRTC and run perception models locally.

```bash
# Laptop client connects to robot at specified IP
python3 -m src.laptop_client 192.168.123.18
```

- Vision modules run on laptop GPU (not robot)
- Robot sends video via WebRTC data channel
- Laptop sends back control commands
- Enables sophisticated analysis with powerful hardware

### Hybrid Deployment

Lightweight perception on robot (immediate response) + intensive analysis on laptop (asynchronous).

- Robot runs gesture detection → instant motor commands
- Laptop receives full sensor data → runs heavy models
- Both operate simultaneously without blocking each other
- Optimal for complex autonomy tasks

## Communication Patterns

### Frame Pipeline with Queuing

The modular design uses bounded queues between processing stages. When throughput is high, the system naturally drops old frames instead of increasing latency.

```
Capture → [Queue] → Inference → [Queue] → Display
    ↓ (full)         ↓ (full)
  Drop frames     Drop frames
```

A human controlling via gestures cares about responsive recent frames, not whether frame #43 was processed. This queuing strategy prioritizes responsiveness.

### Data Flow

1. **Sensor input**: Robot camera (VideoClient) → WebRTC stream → USB camera
2. **Processing**: Vision modules extract features (bounding boxes, keypoints, depth)
3. **Decision**: Gesture recognized → control command generated
4. **Execution**: SportClient translates command → motor commands

Each component doesn't care about the others' implementation:
- YOLO detector: Just processes frames (from any source)
- Gesture dispatcher: Just interprets keypoints (from any detector)
- Motion controller: Just executes commands (from any perception module)

### Telemetry Stream

Separate channel for robot state (battery, motor temps, pose) independent from video, preventing data competition for bandwidth.

## Key Design Principle

The core strength is the **camera abstraction**: the same vision code works across DDS, WebRTC, USB cameras, and video files without modification. This enables flexible deployment—add new detectors, change connection methods, or switch hardware with minimal code changes.

```python
# One create_camera call supports multiple sources
camera = create_camera(
    source="go2",        # or "opencv", "file", "webrtc"
    width=640,
    height=480,
    device=0,
    video_path="video.mp4",
)
```
