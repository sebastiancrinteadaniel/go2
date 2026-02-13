# Unified Go2 Dashboard - Async Architecture

A complete rewrite of the Go2 dashboard using a **single-process, fully-async architecture** with zero subprocess overhead.

## Architecture Overview

### Key Improvements Over Previous Design

| Aspect | Previous | New |
|--------|----------|-----|
| **Processes** | 4 separate Python processes (YOLO, Hand, Depth, Simple) | 1 unified process + async tasks |
| **Memory** | ~320MB base (4x interpreter) | ~80MB base |
| **SDK Init** | Replicated per process | Single init at startup |
| **Model Loading** | Duplicated across processes | Shared via ModelPool |
| **Frame Source** | 4 independent camera instances | 1 shared FrameRouter |
| **Concurrency** | Thread-based process management | Pure async (asyncio) |
| **Startup** | ~3-5 seconds | ~1-2 seconds |

### Design Philosophy

```
Single Event Loop (FastAPI/Uvicorn)
    ↓
Manages all I/O asynchronously
    ├─ HTTP endpoints (FastAPI routes)
    ├─ WebSocket connections
    └─ Frame distribution via asyncio.Queue
    
Inference Workers (Thread Pool)
    ├─ YOLO model detection
    ├─ MediaPipe hand detection
    ├─ RealSense depth capture
    └─ Simple pass-through
    
Frame Pipeline
    Camera → FrameRouter → Processor Queues → MJPEG Streaming
```

## Components

### 1. **config.py** - Unified Configuration
- All settings in one place
- Per-processor configuration
- Easy to modify for different cameras/models

### 2. **model_pool.py** - Shared Model Management
- Ensures models loaded only once
- Lazy loading with async locks
- Singleton pattern for thread-safe access

### 3. **processor.py** - Base Processor Class
- Abstract base class for all vision processors
- Handles frame loop, queue management, statistics
- Easy to extend for new vision tasks

### 4. **processors/** - Individual Processors
- `yolo_processor.py` - YOLOv8 object detection
- `hand_detection_processor.py` - MediaPipe hand tracking
- `depth_camera_processor.py` - RealSense depth + YOLO fusion
- `simple_camera_processor.py` - Low-latency raw frames

### 5. **frame_router.py** - Central Frame Orchestration
- Single camera instance
- Distributes frames to all active processors
- Handles backpressure (drops old frames if queues full)
- Statistics tracking

### 6. **video_streaming.py** - MJPEG HTTP Streaming
- Async JPEG encoding
- Frame generators for FastAPI streaming responses
- Multipart MJPEG protocol

### 7. **app.py** - FastAPI Application
- Lifespan management (startup/shutdown)
- HTTP endpoints for processor control
- Video streaming routes
- Statistics and monitoring

### 8. **templates/index.html** - Web Dashboard
- Real-time video feeds (all MJPEG streams)
- Processor on/off toggles
- System statistics
- System log with real-time updates
- Responsive design

## Running the Application

### Basic Usage

```bash
# Default configuration (Go2 camera, all processors enabled)
python -m src.unified_app

# With options
python -m src.unified_app --host 0.0.0.0 --port 8000 --camera go2 --debug
```

### Camera Sources

```bash
# RealSense camera
python -m src.unified_app --camera g1

# USB camera
python -m src.unified_app --camera usb

# Video file
python -m src.unified_app --camera file
```

### Access Dashboard

Open browser: `http://jetson-ip:8000`

## API Endpoints

### Health & Info

```
GET /api/health              - Health check
GET /api/stats               - System statistics
GET /api/processors          - List all processors
```

### Processor Control

```
POST /api/processors/{id}/start   - Start processor (e.g., "yolo")
POST /api/processors/{id}/stop    - Stop processor
```

### Video Streaming

```
GET /api/streams/{id}/video_feed  - MJPEG stream for processor
```

Example: `http://jetson:8000/api/streams/yolo/video_feed`

## Configuration

Edit `config.py` to customize:

```python
# Camera
CAMERA_SOURCE = "go2"  # or "usb", "file"
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
FPS = 30

# YOLO
YOLO_MODEL = "src/yolo/model/yolov8n.pt"
YOLO_CONFIDENCE = 0.5
YOLO_DEVICE = "cpu"  # or "cuda"

# Hand Detection
HAND_DETECTION_MAX_HANDS = 2
HAND_DETECTION_CONFIDENCE = 0.5

# Streaming
MJPEG_QUALITY = 80
MJPEG_FRAME_QUEUE_SIZE = 3
```

## Adding a New Processor

### 1. Create processor class

```python
# processors/my_processor.py
from src.unified_app.processor import BaseProcessor
import numpy as np

class MyProcessor(BaseProcessor):
    async def process_frame(self, frame: np.ndarray) -> np.ndarray:
        # Your processing logic
        # return modified frame
        pass
```

### 2. Register in app

```python
# app.py
from src.unified_app.processors import MyProcessor

processors_to_create = {
    ...
    "my_proc": MyProcessor,
    ...
}
```

### 3. Add config

```python
# config.py
PROCESSORS_CONFIG = {
    ...
    "my_proc": {
        "enabled": True,
        "name": "My Processor",
        # processor-specific config
    },
    ...
}
```

## Performance Characteristics

### Memory Usage
- **Base**: ~80MB (Python + FastAPI)
- **Per processor**: ~50-100MB depending on model size
- **Total (all 4)**: ~300-400MB (vs ~600MB+ with old design)

### CPU Usage
- **Capture thread**: 5-10% (camera I/O)
- **Inference (YOLO)**: 20-40% (depends on input size)
- **Hand detection**: 10-20%
- **Streaming**: 5-10%

### Latency
- **Frame capture to stream**: 50-150ms
- **End-to-end (camera to browser)**: 100-300ms depending on JPEG quality

## Async vs Threads vs WebRTC

### Why Async?
- Single event loop coordinates all I/O
- Better resource efficiency than threads
- Cleaner code with async/await
- Scales well to many concurrent connections

### Why Threads for Inference?
- YOLO/MediaPipe release Python GIL
- Models run in parallel without blocking event loop
- `asyncio.to_thread()` handles scheduling

### Video Streaming
- HTTP MJPEG (current): Simple, low overhead on LAN
- Optional WebRTC (future): Better bandwidth efficiency, peer-to-peer capability

## Troubleshooting

### Camera not found
```
Check CAMERA_SOURCE in config.py
Verify Go2 SDK or USB camera is available
```

### Models not loading
```
Ensure model files exist and paths are correct in config.py
Check YOLO_MODEL path points to valid .pt file
Verify MediaPipe is installed: pip install mediapipe
```

### High latency
```
Reduce DISPLAY_WIDTH/HEIGHT
Increase YOLO_SKIP_FRAMES to process fewer frames
Reduce MJPEG_QUALITY for lower bandwidth
```

### Memory usage high
```
Reduce MJPEG_FRAME_QUEUE_SIZE
Disable unused processors
Use smaller YOLO model (nano vs. small)
```

## Future Enhancements

- [ ] WebRTC support for remote streaming
- [ ] Gesture-based robot control integration
- [ ] Telemetry dashboard (battery, temps, IMU)
- [ ] Multi-model inference pipeline
- [ ] GPU acceleration for YOLO
- [ ] Custom processor plugins
- [ ] Recording/playback functionality
- [ ] Performance profiling tools

## Files Structure

```
src/unified_app/
├── __init__.py
├── __main__.py                    # Entry point
├── config.py                      # Configuration
├── model_pool.py                  # Shared models
├── processor.py                   # Base class
├── frame_router.py                # Frame distribution
├── video_streaming.py             # MJPEG encoding
├── app.py                         # FastAPI app
├── processors/
│   ├── __init__.py
│   ├── yolo_processor.py
│   ├── hand_detection_processor.py
│   ├── depth_camera_processor.py
│   └── simple_camera_processor.py
└── templates/
    └── index.html                 # Dashboard UI
```

## Dependencies

Already in project but important to note:

```
fastapi >= 0.100.0
uvicorn >= 0.23.0
asyncio (built-in)
numpy
opencv-python
ultralytics (YOLO)
mediapipe
pyrealsense2 (for depth camera, optional)
```

---

**Built with:**
- FastAPI for web framework
- Asyncio for concurrency
- MJPEG for HTTP video streaming
- Shared model pooling for efficiency
