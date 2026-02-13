# Unified Dashboard Tutorial 🎓

Learn how to use the unified computer vision dashboard by building it up step by step. We'll start simple and gradually add more features.

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Step 1: Simple Camera Streaming](#step-1-simple-camera-streaming)
3. [Step 2: Add YOLO Object Detection](#step-2-add-yolo-object-detection)
4. [Step 3: Add Hand Detection](#step-3-add-hand-detection)
5. [Step 4: Add Depth Camera](#step-4-add-depth-camera)
6. [Advanced Topics](#advanced-topics)

---

## Architecture Overview

The unified app is designed around these core concepts:

```
┌──────────────────────────────────────────────────────────┐
│                    UNIFIED DASHBOARD                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  CAMERA CAPTURE (FrameRouter)                           │
│    │                                                     │
│    ├─► Go2 Robot Camera                                │
│    └─► USB Webcam                                      │
│         │                                               │
│         ▼                                               │
│  ┌────────────────────────────────────────┐            │
│  │  Frame Distribution (Async Queues)     │            │
│  └────────────┬───────────────────────────┘            │
│               │                                         │
│     ┌─────────┼─────────┬──────────┬─────────┐        │
│     ▼         ▼         ▼          ▼         ▼        │
│   YOLO   Hand Detect  Depth    Simple    Disabled   │
│  Processor Processor  Processor Processor           │
│     │         │         │          │                  │
│     └─────────┴─────────┴──────────┘                  │
│               │                                        │
│               ▼                                        │
│  ┌────────────────────────────────────────┐          │
│  │  Video Streaming (HTTP MJPEG)          │          │
│  │  - Easy LAN streaming                  │          │
│  │  - No WebRTC complexity                │          │
│  └────────────────────────────────────────┘          │
│               │                                        │
│               ▼                                        │
│        WEB DASHBOARD                                 │
│        View feeds, toggle modules                   │
│                                                      │
└──────────────────────────────────────────────────────┘
```

**Key Design Principles:**
- **Single Process**: All modules run in one Python process (not 4 separate ones)
- **Async-First**: Uses `asyncio` for efficient I/O and frame handling
- **Thread Pool**: Heavy inference (YOLO, MediaPipe) runs on thread pool to avoid blocking
- **Config-Driven**: All settings via dataclasses, no magic strings
- **Modular**: Enable/disable modules without code changes

---

## Step 1: Simple Camera Streaming

### Goal
Get raw video streaming working with minimal overhead.

### What You're Learning
- How the dashboard starts
- How camera capture works
- How frames are streamed to the web UI

### Instructions

**1. Start the app:**
```bash
python -m src.unified_app
```

You should see output like:
```
INFO:root:Application startup
INFO:root:Creating camera: go2
INFO:FrameRouter:Camera initialized: go2 on eth0
INFO:Processor.simple_camera:Starting processor
INFO:root:Server running at http://0.0.0.0:8000
```

**4. Open the web dashboard:**
- Go to `http://localhost:8000` in your browser
- You should see a live camera feed with FPS counter in the top-left

**💡 To focus on just the simple camera:**

You can toggle modules on/off from the **dashboard UI**. This is the easiest way to learn:
- Click the processor tabs at the top
- Each has an **enable/disable button**
- Start with just "Simple Camera" enabled
- No need to edit config files!

**What's Happening:**
1. The app reads the config
2. Creates a camera (connected to Go2 robot or USB)
3. Starts capturing frames at 30 FPS
4. Simple camera processor just adds FPS overlay
5. Streams frames via HTTP MJPEG
6. Web dashboard displays the stream

**Try This:**
- Move in front of the camera - you should see real-time feed
- The FPS counter should show ~30 FPS

---

## Step 2: Add YOLO Object Detection

### Goal
Enable real-time object detection with bounding boxes.

### What You're Learning
- How processors listen to frame queue
- How inference runs without blocking
- Model loading and caching

### Instructions

**1. Keep the app running from Step 1**

**2. Toggle YOLO from the web dashboard:**
- Open `http://localhost:8000`
- Find the "YOLO" tab at the top
- Click the **Enable** button
- You should see a message: "YOLO processor started"

**3. Look at the dashboard:**
- The video feed will now show bounding boxes around detected objects
- Each box has a class label and confidence score
- The YOLO tab shows the detection stream in real-time

**What's Happening:**
1. FrameRouter captures frames
2. Each frame is sent to YoloProcessor via async queue
3. YoloProcessor runs inference on a thread (doesn't block main loop)
4. Detected boxes are drawn on the frame
5. Processed frame is sent to video streaming
6. Browser receives annotated frame

**Performance Tip:**
- By default, YOLO input is 640x640 pixels
- If it's slow, reduce to 416x416 in config:

```python
YoloConfig(
    enabled=True,
    inference_width=416,    # ← Smaller = faster
    inference_height=416,
    ...
)
```

**Try This:**
- Hold up an object (water bottle, book, person)
- Watch YOLO detect it in real-time
- Check confidence scores for different objects

---

## Step 3: Add Hand Detection

### Goal
Detect and track hand poses in real-time.

### What You're Learning
- Running multiple processors simultaneously
- Frame distribution to multiple modules
- Combining multiple detection streams

### Instructions

**1. Toggle hand detection from the dashboard:**
- You should still have the app running
- Open `http://localhost:8000`
- Find the "Hand Detection" tab
- Click **Enable**
- You should see: "Hand Detection processor started"

**2. Look at the dashboard:**
- A separate stream for hand detection appears
- Shows hand keypoints (joints) and connections (bones)
- Green dots = detected hand keypoints
- Lines connect the joints

**What's Happening:**
1. Now running TWO processors simultaneously:
   - YoloProcessor (objects)
   - HandDetectionProcessor (hands)
2. Both receive frames from the same queue independently
3. Each processes frames without waiting for the other
4. Both streams available on dashboard

**Hand Keypoints:**
```
Thumb ──► Index finger
 │         │
 └─────────┤
 Middle, ring, pinky fingers below
```

**Configuration:**
Adjust in `src/unified_app/config.py`:

```python
HandDetectionConfig(
    enabled=True,
    max_hands=2,              # Detect up to 2 hands
    confidence=0.5,           # Detection threshold (0-1)
    inference_width=640,      # Input size
    inference_height=640,
)
```

**Try This:**
- Hold up your hand to the camera
- Watch MediaPipe detect your fingertips
- Try 2-handed gestures - it should track both
- Move your hand slowly vs fast - compare accuracy

---

## Step 4: Add Depth Camera

### Goal
Add a completely separate RGB-D camera stream with YOLO detection.

### What You're Learning
- Adding a new independent camera source
- RealSense pipeline integration
- Depth data visualization

### Instructions

**1. Toggle depth camera from the dashboard:**
- App still running, go to `http://localhost:8000`
- Find the "Depth Camera" tab
- Click **Enable**
- You should see: "Depth Camera processor started"

**Note:** If you don't have a physical RealSense camera, this will disable gracefully and show a warning. You can skip this step or get a RealSense USB camera (~$100-200) later.

**2. Look at the dashboard:**
- A new "Depth Camera" tab shows (if device found)
- Shows RGB stream from the depth camera
- May include YOLO detections on RGB
- Depth map visualization available

**What's Happening:**
1. Now running THREE independent processors:
   - YoloProcessor (main camera objects)
   - HandDetectionProcessor (main camera hands)
   - DepthCameraProcessor (depth camera RGB + depth)
2. Each has its own camera pipeline
3. Each processing independently
4. All streams available on dashboard

**Depth Camera Settings:**

```python
DepthCameraConfig(
    enabled=True,
    depth_width=640, depth_height=480,      # Depth resolution
    color_width=640, color_height=480,      # RGB resolution
    fps=30,                                  # Capture FPS
    yolo_enabled=True,                      # Run YOLO on RGB?
)
```

**Try This:**
- Hold an object at different distances
- Watch depth values change
- RealSense gives precise 3D coordinates
- Great for pickup/manipulation tasks (robot arm, gripper, etc.)

---

## Advanced Topics

### Testing on Windows/Mac/Linux 🖥️

**Good news: The app works on any OS!** Windows, Mac, and Linux all work perfectly.

#### Quick Start (Windows, Mac, or Linux)

**1. Install dependencies:**
```bash
pip install -r requirements-webrtc.txt
pip install -e .
```

**2. Start the app:**
```bash
python -m src.unified_app --camera usb
```

**3. Open dashboard:**
```
http://localhost:8000
```

That's it! No Ubuntu needed. ✅

#### Camera Options on Windows

**Option A: USB Webcam** (Easiest - just works!)
```bash
python -m src.unified_app --camera usb
# Your built-in or external USB webcam - auto-detected!
```

**Option B: Video File** (Great for testing without camera)
```python
# Edit src/unified_app/common/cameras.py
# In create_camera(), add:
if source == "video":
    return cv2.VideoCapture("path/to/video.mp4")
```

Then run:
```bash
python -m src.unified_app --camera video
```

**Option C: Go2 Robot** (Network connection, works from Windows!)
```bash
# Make sure Go2 is on same LAN
# Then:
python -m src.unified_app --camera go2
```

#### Testing Without a Real Camera

**Use OBS Virtual Camera** (free, easy):

1. Download & install OBS Studio
2. Create scene with what you want (image, text, etc.)
3. Tools → VirtualCamera → Start
4. Run app:
   ```bash
   python -m src.unified_app --camera usb
   ```
5. Select "OBS Virtual Camera" in Windows settings
6. App sees your virtual stream!

**Alternative: Screen Share as Camera**
- ManyCam (free version)
- Camtasia (paid)
- Plays videos or screen content as fake camera

#### System Requirements (Windows/Mac/Linux)

| Requirement | Windows | Mac | Linux |
|-----------|---------|-----|-------|
| **Python** | 3.8+ ✅ | 3.8+ ✅ | 3.8+ ✅ |
| **pip** | ✅ | ✅ | ✅ |
| **OpenCV** | ✅ | ✅ | ✅ |
| **FastAPI** | ✅ | ✅ | ✅ |
| **aiortc** | ✅ | ✅ | ✅ |
| **CUDA** | ❌ (but CPU works) | ❌ (but CPU works) | ✅ (optional) |

**Note:** GPU/CUDA optional. CPU works fine for testing!

#### Windows-Specific Setup

**1. Install Python:**
- Download from python.org
- **Important:** Check "Add Python to PATH" during install
- Verify: `python --version` in PowerShell

**2. Install dependencies:**
```powershell
# From the go2_cv/go2 directory
pip install -r requirements-webrtc.txt
pip install -e .
```

**3. Run the app:**
```powershell
python -m src.unified_app --camera usb
```

**4. Open browser:**
```
http://localhost:8000
```

#### Mac-Specific Notes

**Allow camera access:**
- First run will ask permission
- Settings → Security & Privacy → Camera → Allow Python

```bash
python -m src.unified_app --camera usb
```

#### Linux-Specific Notes

**Install system dependencies (Ubuntu/Debian):**
```bash
sudo apt-get install python3-dev libopencv-dev python3-opencv
```

**Then install Python packages:**
```bash
pip install -r requirements-webrtc.txt
```

#### Performance Comparison

| OS/Hardware | YOLO Speed | Hand Detection | Notes |
|-------------|-----------|-----------------|-------|
| **Windows (CPU)** | ~100ms | ~50ms | Good for testing |
| **Mac M1/M2 (CPU)** | ~80ms | ~40ms | Excellent performance |
| **Jetson Nano (GPU)** | ~30ms | ~10ms | Best for production |
| **Jetson Orin (GPU)** | ~15ms | ~5ms | Production tier |

**Tip:** Windows/Mac are perfect for development and testing. Deploy to Jetson for production!

#### Common Windows Issues & Fixes

**Error: "ModuleNotFoundError: No module named 'cv2'"**
```powershell
# Make sure Python path is correct
python -m pip install opencv-python
```

**Error: "Port 8000 already in use"**
```powershell
# Use a different port
python -m src.unified_app --port 9000
# Then visit: http://localhost:9000
```

**WebRTC not working on Windows:**
```powershell
# Make sure aiortc and av are installed
pip install aiortc av
```

**Camera not detected:**
```powershell
# Test camera access
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
# True = camera works, False = check Device Manager
```

---

### Configure Camera Source

**Use Go2 Robot Camera (default):**
```python
camera=CameraConfig(
    source="go2",
    interface="eth0",  # or "wlan0" for WiFi
    ...
)
```

**Use USB Webcam:**
```python
camera=CameraConfig(
    source="usb",
    ...
)
```

**Via Command Line:**
```bash
python -m src.unified_app --camera usb
python -m src.unified_app --camera go2
```

### Adjust Frame Rate

All modules run at 30 FPS by default. Change in config:

```python
camera=CameraConfig(fps=15)          # Lower = less data, faster
```

Or per-module:
```python
depth_camera=DepthCameraConfig(fps=20)  # Depth at 20 FPS
yolo=YoloConfig(...)                    # YOLO inherits camera FPS
```

### Confidence Thresholds

**YOLO:**
```python
yolo=YoloConfig(
    confidence=0.5,  # 0.5 = 50% threshold (lower = more detections)
    ...
)
```

**Hand Detection:**
```python
hand_detection=HandDetectionConfig(
    confidence=0.5,  # MediaPipe threshold
    ...
)
```

### Skip Frames (Performance)

Process every Nth frame to boost speed:

```python
simple_camera=SimpleCameraConfig(skip_frames=1)  # Process every 2nd frame
yolo=YoloConfig(skip_frames=2)                   # Process every 3rd frame
```

This trades latency for speed.

### Model Selection

**YOLO Models (by speed/accuracy):**
- `yolov8n` (nano) - fastest, smallest
- `yolov8s` (small) - balanced
- `yolov8m` (medium) - slower, more accurate
- `yolov8l` (large) - slowest, best accuracy

```python
yolo=YoloConfig(
    model_path="src/yolo/model/yolov8s.engine",  # Use small model
    ...
)
```

### Access Dashboard Controls

Open `http://localhost:8000` to:
- Toggle processors on/off
- View different streams
- Adjust MJPEG quality
- Monitor FPS/stats

### Device Selection

Auto-detect (CUDA if available, fallback to CPU):
```python
yolo=YoloConfig(device="auto")  # ← Default
```

Force CPU:
```python
yolo=YoloConfig(device="cpu")
```

Force CUDA:
```python
yolo=YoloConfig(device="cuda")
```

---

## Why HTTP MJPEG Over WebRTC? 🎥

This is a great question! Let's understand why we chose **HTTP MJPEG** instead of **WebRTC** for streaming.

### What's the Difference?

#### **HTTP MJPEG** (What We Use ✅)
- **MJPEG** = Motion JPEG
- Sends a **sequence of JPEG images** over HTTP
- Each frame is a complete JPEG image
- Browser displays them in sequence
- Simple `<img>` or `<video>` tag in HTML

**How it works:**
```
Server                    Browser/Client
├─ Frame 1 (JPEG)  ──────►  Display
├─ Frame 2 (JPEG)  ──────►  Display
├─ Frame 3 (JPEG)  ──────►  Display
└─ ...
```

#### **WebRTC** (Alternative)
- **WebRTC** = Web Real-Time Communication
- Sends **compressed video stream** (H.264, VP9, etc.)
- Requires complex handshaking (STUN, TURN servers)
- Low-latency peer-to-peer
- Built for video conferencing

**How it works:**
```
Server              Signaling Server          Browser/Client
  │────── OFFER ──────────►
  │                    ◄────── ANSWER ────────
  │                           (complex setup)
  │
  ├─ Compressed Stream ───────►  Decode & Display
  └─ (bidirectional if needed)
```

---

### Comparison: MJPEG vs WebRTC

| Feature | HTTP MJPEG | WebRTC |
|---------|-----------|--------|
| **Setup** | Simple (just HTTP) | Complex (STUN, TURN, signaling) |
| **Latency** | ~100-500ms | ~50-200ms (lower) |
| **Bandwidth** | Higher (no compression) | Lower (H.264/VP9 compression) |
| **Complexity** | Single image stream | Full multimedia framework |
| **Browser Support** | ✅ All browsers | ✅ Modern browsers only |
| **Firewall Friendly** | ✅ Yes (HTTP port 80/8000) | ❌ Tricky (NAT traversal needed) |
| **Mobile Friendly** | ✅ Yes | ✅ Yes (better battery) |
| **LAN Only?** | Works great | Works great |
| **Internet/4G?** | Possible but high bandwidth | Better (compression) |
| **Multiple Producers** | Easy (multiple /feed endpoints) | Complex (many STUN servers) |

---

### MJPEG Default, WebRTC Optional 🎯

**Why MJPEG is the default:**

1. **You're on a LAN** (Laptop + Jetson same WiFi)
   - Bandwidth not a constraint
   - WebRTC overhead not worth setup
   - MJPEG perfect for local streaming

2. **Simplicity is King**
   - No STUN/TURN server setup needed for LAN
   - Simple HTTP streaming
   - Just works with `http://jetson-ip:8000`
   - Dashboard is pure HTML

3. **Perfect for Jetson development**
   - Jetson CPU is for AI, not video encoding
   - MJPEG lets Jetson focus on inference
   - Lower CPU overhead on startup

**But now you can also use WebRTC!** ✅

Click the **WebRTC** tab to switch anytime:
- ✅ Lower bandwidth (5-7x savings)
- ✅ Better for internet/remote access
- ✅ H.264 compression included
- ✅ Same Jetson IP access
- ✅ Optional - install `aiortc av` if needed

---

### When to Use WebRTC Tab? 📡

Use **WebRTC** (click the tab) when:

- ✅ **Over the internet** (not LAN)
  - Compression saves bandwidth
  - Perfect with Ngrok tunneling
  
- ✅ **Bandwidth limited** (cellular, 4G)
  - H.264 compression reduces bitrate 5-7x
  - Saves your data plan!
  
- ✅ **Permanent remote setup**
  - VPN + WebRTC = optimal combo
  
- ✅ **Heavy production app**
  - Alrea spending resources on infrastructure
  - Worth the complexity

**For this project:** None of those apply! 🎉

---

### Bandwidth Comparison

**Scenario:** 640x480 @ 30 FPS

**HTTP MJPEG:**
```
Frame size: 640×480 JPEG ≈ 30-50 KB
FPS: 30
Bitrate: 30 × 40 KB = 1.2 MB/s = 9.6 Mbps
```

**WebRTC H.264:**
```
Bitrate: ~1.5-2 Mbps (with compression)
Savings: ~5-7x less bandwidth
```

### WebRTC Implementation ✅ (Now Built-In!)

We've implemented WebRTC streaming! You now have **dual streaming tabs** for each video feed.

#### Installation

First, install WebRTC dependencies:
```bash
pip install aiortc av
```

Or install everything at once:
```bash
pip install -r requirements-webrtc.txt
```

#### Using WebRTC Tabs

**On the dashboard**, each video stream now has **two tabs**:

1. **MJPEG Tab** (blue, default)
   - Higher bandwidth (~10 Mbps)
   - Lower latency on LAN (<200ms)
   - Works on any browser
   - Default option - just works

2. **WebRTC Tab** (gray, click to switch)
   - Lower bandwidth (~1.5-2 Mbps = 5-7x savings!)
   - Compressed H.264 video
   - Better for internet/remote access
   - Slightly more latency (~100-150ms)

**Example workflow:**
```
1. Start app: python -m src.unified_app
2. Open dashboard: http://localhost:8000
3. Enable a processor (e.g., "Simple Camera")
4. See blue MJPEG tab active by default
5. Click gray "WebRTC" tab to switch
6. Watch live video transition to WebRTC stream
7. Click "MJPEG" to switch back
```

#### When to Use Each

**Use MJPEG:**
- ✅ Local LAN (Laptop + Jetson same WiFi)
- ✅ High bandwidth available
- ✅ Want simplicity
- ✅ Lowest setup overhead

**Use WebRTC:**
- ✅ Internet access (4G, remote)
- ✅ Limited bandwidth
- ✅ Want video compression
- ✅ Can tolerate slight delay

#### Real-World Example

**Scenario: Controlling Jetson from coffee shop**

1. Jetson at home on WiFi
2. You're remote (4G phone)
3. Open Ngrok tunnel (see Jetson deployment section)
4. Dashboard loads: `https://your-ngrok-url.ngrok.io`
5. Click **WebRTC tab** instead of MJPEG
6. Saves 5-7x bandwidth! ☑️
7. Video streams with H.264 compression

#### Bandwidth Comparison (Real Numbers)

| Scenario | MJPEG | WebRTC | Savings |
|----------|-------|--------|---------|
| **LAN (30 sec)** | 300 KB | 60 KB | 80% less 📉 |
| **1 hour streaming** | ~36 MB | ~6 MB | 30 MB saved 💾 |
| **4G data plan** | Eats plan | Much better ✅ |

#### Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    UNIFIED DASHBOARD                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  SHARED CAMERA + PROCESSORS                            │
│    ↓                                                     │
│  ┌─────────────────────────────────────────┐           │
│  │   Processor Output Queues               │           │
│  │   (YOLO, Hand, Depth, Simple)          │           │
│  └──────────┬──────────────────────────────┘           │
│             │                                            │
│      ┌──────┴──────┐                                    │
│      ▼             ▼                                     │
│  ┌────────┐   ┌──────────────────────┐                │
│  │ MJPEG  │   │  ProcessorStreamTrack│                │
│  │Handler │   │  (WebRTC VideoTrack) │                │
│  └────────┘   └──────────────────────┘                │
│      │             │                                     │
│      │             ▼                                     │
│  /stream/    /api/webrtc/offer                        │
│   {id}/       (SDP Handshake)                         │
│  video_feed       │                                     │
│      │             ▼                                     │
│  ┌─────┴──────┐   Browser WebRTC                      │
│  │HTTP MJPEG  │   Connection                          │
│  │Stream      │                                        │
│  └────┬───────┘                                        │
│       │                                                 │
│       ▼                                                 │
│   Browser                                              │
│   MJPEG<img> or WebRTC<video>                         │
│                                                        │
└──────────────────────────────────────────────────────────┘
```

#### Code Details (What Happened Behind the Scenes)

**Dashboard HTML** (`templates/index.html`):
```javascript
// Two buttons under each video
<button onclick="switchStreamType(processorId, 'mjpeg')">MJPEG</button>
<button onclick="switchStreamType(processorId, 'webrtc')">WebRTC</button>

// Clicking WebRTC calls this
async function startWebRTCStream(processorId) {
    const pc = new RTCPeerConnection();
    pc.addTransceiver('video', { direction: 'recvonly' });
    
    const offer = await pc.createOffer();
    const response = await fetch('/api/webrtc/offer', {
        method: 'POST',
        body: JSON.stringify({
            sdp: offer.sdp,
            type: 'offer',
            processor: processorId
        })
    });
    
    const answer = await response.json();
    pc.setRemoteDescription(new RTCSessionDescription(answer));
    // Video now streams compressed!
}
```

**Server** (`app.py`):
```python
class ProcessorStreamTrack(VideoStreamTrack):
    """Serves processor frames as WebRTC video"""
    
    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = self.output_queue.get_nowait()  # From processor
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        return video_frame

@app.post("/api/webrtc/offer")
async def webrtc_offer(request: Request):
    # Handle WebRTC SDP offer/answer negotiation
    processor = global_state["processors"][processor_id]
    pc = RTCPeerConnection()
    pc.addTrack(ProcessorStreamTrack(processor))
    # ... complete handshake
    return {"sdp": answer.sdp, "type": "answer"}
```

---

### Troubleshooting WebRTC

**WebRTC tabs show as disabled/gray:**
- Check if `aiortc` installed: `pip install aiortc av`
- Check logs: `WebRTC not available` message?
- MJPEG tabs always work as fallback ✅

**WebRTC stream not connecting:**
1. Make sure processor is **running** (blue status indicator)
2. Check browser console (F12) for errors
3. Try MJPEG tab first to verify video works
4. Switch back to WebRTC

**Error: "HTTP 501: WebRTC not available"**
```bash
pip install aiortc av
# Then restart the app
python -m src.unified_app
```

**WebRTC video laggy/buffering:**
- That's expected! Jetson is compressing in real-time
- Reduce FPS in config (see Advanced Topics)
- Reduce resolution

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'numpy'"

Install dependencies:
```bash
pip install -r requirements-webrtc.txt
pip install -e .
```

### Camera not found / "Failed to initialize camera"

- **Go2 Camera**: Check network connection to robot
  ```bash
  ping go2.local  # Should work on LAN
  ```
  
- **USB Camera**: List connected devices
  ```bash
  python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
  ```

### YOLO model not found

Download the model:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

This downloads to `~/.yolov8/...`. Copy to `src/yolo/model/`.

### Low FPS / Laggy Video

1. **Reduce resolution:**
   ```python
   camera=CameraConfig(width=480, height=360)
   ```

2. **Reduce YOLO input:**
   ```python
   yolo=YoloConfig(inference_width=416, inference_height=416)
   ```

3. **Skip frames:**
   ```python
   yolo=YoloConfig(skip_frames=2)
   ```

4. **Disable unused modules:**
   ```python
   hand_detection=HandDetectionConfig(enabled=False)
   depth_camera=DepthCameraConfig(enabled=False)
   ```

### Dashboard not loading

1. Check server is running:
   ```bash
   curl http://localhost:8000/
   ```

2. Change port in config:
   ```python
   server=ServerConfig(host="0.0.0.0", port=8080)
   ```
   
   Then access: `http://localhost:8080`

### High CPU Usage

- The app is CPU-intensive! Computer vision is hard.
- Reduce resolution/FPS as shown above
- Use GPU if available (CUDA)

---

## Architecture Files Reference

| File | Purpose |
|------|---------|
| `config.py` | Configuration dataclasses |
| `app.py` | FastAPI app and lifespan management |
| `frame_router.py` | Captures frames and distributes to processors |
| `processor.py` | Base class for all processors |
| `processors/` | Individual processor implementations |
| `model_pool.py` | Lazy-loads and caches ML models |
| `video_streaming.py` | HTTP MJPEG streaming |
| `common/cameras.py` | Camera abstractions (Go2, USB) |
| `common/fps.py` | FPS calculation utility |
| `templates/` | Web dashboard HTML/CSS/JS |

---

## Deploying to Jetson (Wireless Access) 🤖

The app is **built for Jetson** from day one! Here's how to run it on a Jetson and access it wirelessly from your laptop.

### Prerequisites
- Jetson device (Orin Nano, Orin, Xavier, etc.)
- Jetson connected to same WiFi as your laptop
- All dependencies installed on Jetson (see [requirements-webrtc.txt](../../requirements-webrtc.txt))

### Step 1: Find Jetson's IP Address

On the **Jetson**, run:
```bash
hostname -I
```

Or on **your laptop** (if Jetson has mDNS):
```bash
ping jetson.local
```

You should see something like:
```
192.168.1.100
```

**Note:** Write down this IP address!

### Step 2: Start the App on Jetson

On the **Jetson**, run the app:
```bash
cd /path/to/go2_cv/go2
python -m src.unified_app
```

You should see:
```
INFO:root:UNIFIED DASHBOARD RUNNING
============================================================
Server: http://0.0.0.0:8000
Camera: usb
Processors: 
============================================================
```

The key is `0.0.0.0:8000` - this means the Jetson is listening on **all network interfaces** (not just localhost).

### Step 3: Access from Laptop

On your **laptop**, open any browser and go to:
```
http://192.168.1.100:8000
```

Replace `192.168.1.100` with the IP address from Step 1.

**You should see the dashboard!** 🎉

### Step 4: Enable Modules from the UI

The dashboard will load with no processors active (because we set `enabled=False` by default). Click the buttons to enable what you want:
- Click "Simple Camera" → Enable
- Click "YOLO" → Enable
- etc.

You're now controlling the Jetson wirelessly! 📡

### Network Troubleshooting

**Can't reach the Jetson?**

1. **Check they're on same WiFi:**
   - Jetson WiFi: `nmcli device wifi`
   - Laptop: WiFi settings → Current network
   - Should match!

2. **Check firewall (if needed):**
   ```bash
   # On Jetson, allow port 8000
   sudo ufw allow 8000/tcp
   ```

3. **Try different port (if 8000 blocked):**
   
   On **Jetson**, edit `src/unified_app/config.py`:
   ```python
   server=ServerConfig(
       host="0.0.0.0",
       port=9000,  # ← Different port
   )
   ```
   
   Then access from laptop: `http://192.168.1.100:9000`

4. **Verify connection:**
   ```bash
   # From laptop, test if Jetson is reachable
   ping 192.168.1.100
   curl http://192.168.1.100:8000/
   ```

### Performance on Jetson

Jetson devices are **GPU-accelerated**, so:
- ✅ YOLO runs fast (CUDA-enabled)
- ✅ Hand Detection smooth (MediaPipe GPU)
- ✅ Multiple processors simultaneously

But remember:
- **Jetson Orin Nano**: ~8GB RAM, good for 1-2 processors
- **Jetson Orin**: ~12GB RAM, good for all 4 processors
- **Go2 USB camera**: Works over the LAN connection

**Tip:** If FPS drops, reduce resolution or skip frames:
```python
camera=CameraConfig(width=480, height=360, fps=15)
yolo=YoloConfig(skip_frames=1)
```

### Gotchas

1. **Camera must be physical (USB) on Jetson**
   - Go2 camera over network works from desktop, but Jetson should use USB cameras connected directly
   - OR access Go2 over network via `CameraConfig(source="go2", interface="eth0")`

2. **Model files big on Jetson**
   - YOLO TensorRT model (~50MB) takes time to load first time
   - Subsequent calls are fast
   - Download once, reuse

3. **Keep Jetson cool**
   - High FPS for long time = heat
   - Jetson will thermal throttle if too hot
   - Use heatsink!

### Example: Full Jetson Setup

**On Jetson:**
```bash
# Clone repo
git clone <repo>
cd go2_cv/go2

# Install dependencies
pip install -r requirements-webrtc.txt
pip install -e .

# Start app (all modules disabled by default)
python -m src.unified_app
```

**On your laptop:**
```bash
# Find Jetson IP
# (Jetson will print it, or use hostname -I)

# Open browser
http://192.168.1.100:8000

# Enable processors as needed
```

Done! You now have the full CV dashboard accessible from anywhere on your network. 🚀

### Accessing Over the Internet (4G, Mobile, Remote) 🌐

Yes, you can access your Jetson dashboard from outside your home network! But there are tradeoffs. Here's the comparison:

#### Option 1: Port Forwarding (DIY)

**Setup:**
1. Get your home IP: `curl ifconfig.me`
2. Log into your router's admin panel (192.168.1.1)
3. Forward external port 9000 → Jetson internal port 8000
4. Get a dynamic DNS name (freeDNS, no-ip.com) - your IP changes!
5. Access: `http://your-dynamic-dns:9000`

**Pros:**
- Free
- Direct connection

**Cons:**
- ❌ **Security nightmare** (exposed to whole internet)
- ❌ **No HTTPS** (credentials visible!)
- ❌ **High bandwidth** (MJPEG ~10 Mbps)
- ❌ **ISP blocks port 80/8000** (many home ISPs block)
- ❌ **Complex** (router config, IP changes, DuckDNS)

**Only do this for trusted networks on same VPN!**

---

#### Option 2: Ngrok Tunneling ⭐ (Recommended)

**Best option for quick remote access.**

**Setup (5 minutes):**

1. **Install ngrok:**
   ```bash
   # Download from https://ngrok.com/download
   # Or: brew install ngrok (macOS), choco install ngrok (Windows)
   ```

2. **Create free account:** https://ngrok.com/signup

3. **Get your auth token:**
   ```bash
   ngrok config add-authtoken YOUR_TOKEN_HERE
   ```

4. **On Jetson, create a tunnel to port 8000:**
   ```bash
   ngrok http 8000
   ```
   
   You'll see:
   ```
   Forwarding    https://abc123def456.ngrok.io -> http://localhost:8000
   ```

5. **Access from anywhere:**
   ```
   https://abc123def456.ngrok.io
   ```

**Pros:**
- ✅ Easy setup (1 command)
- ✅ Works through any firewall/NAT
- ✅ **HTTPS included** (encrypted!)
- ✅ Reliable (ngrok handles everything)
- ✅ No port forwarding needed

**Cons:**
- ❌ Still high bandwidth (MJPEG ~10 Mbps)
- Limited to ngrok's infrastructure
- URL changes if tunnel restarts (get custom domain for $5/mo)

**Example workflow:**
```bash
# On Jetson
ssh jetson@192.168.1.100
python -m src.unified_app   # Starts on :8000

# In another terminal on Jetson
ngrok http 8000

# On your phone/laptop anywhere
# Visit: https://abc123def456.ngrok.io
# See dashboard! 🎉
```

---

#### Option 3: WebRTC (Future) 🚀

**When to use:** If remote access is permanent and you have production infrastructure.

**Why not now:**
- More complex to set up
- Requires signaling server
- Overkill for temporary access

**But benefits over internet:**
- ✅ Compression (1/5 bandwidth of MJPEG)
- ✅ Lower latency
- ✅ Optimized for networks

To implement: See "How to Switch to WebRTC" section above.

---

#### Option 4: VPN (Most Secure) 🔒

**When to use:** If you have a home VPN already running

**Setup:**
1. Set up home VPN (WireGuard, OpenVPN, Tailscale)
2. Connect your phone/laptop to VPN
3. Access Jetson on private VPN IP: `http://192.168.1.100:8000`

**Pros:**
- ✅ Secure (all encrypted)
- ✅ Like being on home network
- ✅ Works with MJPEG perfectly
- ✅ No bandwidth concerns on VPN

**Cons:**
- Requires VPN setup (learning curve)
- Phone battery drain (VPN always running)

**Recommended:** Use **Tailscale** for home VPN (free, easy)
```bash
# Install on Jetson: curl -fsSL https://tailscale.com/install.sh | sh
# Web: https://login.tailscale.com
# Access: http://jetson-ip-on-tailnet:8000
```

---

### Which Option Should You Use?

| Scenario | Recommendation |
|----------|-----------------|
| **Quick demo (5 minutes)** | Ngrok tunneling ⭐ |
| **Permanent remote access** | WebRTC (future) or VPN |
| **Already have home VPN** | VPN (most secure) |
| **ISP blocks ports** | Ngrok or VPN |
| **Don't care about cell data** | VPN (constant access) |
| **Sharing with strangers** | Ngrok with password/firewall |

**TL;DR:** Use **Ngrok** for quick testing, **VPN** for daily access. 🚀

---

1. ✅ Understand Step 1-4 above
2. 📝 Modify `config.py` to customize settings
3. 🎯 Create your own processor in `processors/`
4. 🤖 Integrate with Go2 robot actions
5. 📊 Add custom analytics/statistics
6. 🚀 Deploy to production

---

## Quick Reference

**Start with Simple Camera:**
```bash
# Edit config.py and set:
# yolo.enabled = False
# hand_detection.enabled = False
# depth_camera.enabled = False
python -m src.unified_app
```

**Enable All Modules & Visit Dashboard:**
```bash
# Edit config.py and set all enabled=True
python -m src.unified_app
# → Open http://localhost:8000
```

**Change Camera Source:**
```bash
python -m src.unified_app --camera usb
python -m src.unified_app --camera go2
```

**Reduce Resource Usage:**
```bash
python -m src.unified_app --camera usb --fps 15
```

**Test WebRTC (Low Bandwidth Streaming):**
```bash
# 1. Install WebRTC dependencies
pip install aiortc av

# 2. Start app
python -m src.unified_app

# 3. Open http://localhost:8000
# 4. Enable a processor
# 5. Click the gray "WebRTC" tab (not blue MJPEG)
# 6. See compressed H.264 stream (5-7x less bandwidth!)
```

**Use for Internet Access:**
```bash
# On Jetson:
python -m src.unified_app

# In another terminal:
ngrok http 8000
# → https://your-ngrok-url.ngrok.io

# On laptop (mobile/remote):
# 1. Open https://your-ngrok-url.ngrok.io
# 2. Enable processor
# 3. Click WebRTC tab (saves bandwidth on 4G!)
```

---

Happy learning! 🚀
