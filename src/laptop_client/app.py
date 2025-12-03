"""
Laptop Client Dashboard

A remote dashboard that runs on your laptop and connects to the Unitree Go2 via WiFi.
Unlike the robot dashboard, this uses WebRTC instead of DDS for communication.

Features:
- Same UI as robot dashboard
- Start/Stop CV modules (YOLO, Hand Detection) running locally on laptop
- WebRTC video stream from robot
- Send commands via WebRTC data channel

Usage:
    python -m src.laptop_client 192.168.123.18
    # or
    python -m src.laptop_client  # uses IP from config.py
"""

import sys
import os
import asyncio
import logging
import time
from queue import Queue, Empty

import psutil
import cv2

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager

# Reuse the ProcessManager from dashboard
from ..dashboard.process_manager import ProcessManager

from .config import CONFIG
from .webrtc_bridge import WebRTCBridge

# Parse robot IP from command line
ROBOT_IP = None
for arg in sys.argv[1:]:
    if not arg.startswith("-"):
        ROBOT_IP = arg
        break

if not ROBOT_IP:
    ROBOT_IP = CONFIG["webrtc"]["robot_ip"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
bridge: WebRTCBridge = None
manager = ProcessManager()

# Frame queues for different streams
frame_queues = {
    "raw": Queue(maxsize=5),
}

# FPS tracking
fps_stats = {
    "raw": {"fps": 0.0, "last_time": time.time(), "frame_count": 0}
}


def on_frame_received(frame):
    """Callback when a frame is received from WebRTC - updates all raw queues."""
    # Update FPS
    fps_stats["raw"]["frame_count"] += 1
    now = time.time()
    elapsed = now - fps_stats["raw"]["last_time"]
    if elapsed >= 1.0:
        fps_stats["raw"]["fps"] = fps_stats["raw"]["frame_count"] / elapsed
        fps_stats["raw"]["frame_count"] = 0
        fps_stats["raw"]["last_time"] = now
    
    # Put frame in queue
    q = frame_queues["raw"]
    if q.full():
        try:
            q.get_nowait()
        except Empty:
            pass
    q.put(frame.copy())


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bridge
    
    # Startup - cleanup any orphan processes
    manager.cleanup_orphans()
    
    logger.info(f"Connecting to Go2 at {ROBOT_IP}...")
    bridge = WebRTCBridge(robot_ip=ROBOT_IP)
    bridge.add_frame_callback(on_frame_received)
    bridge.start()
    
    yield
    
    # Shutdown
    await manager.stop_all()
    if bridge:
        bridge.stop()


app = FastAPI(lifespan=lifespan)

current_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(current_dir, "templates")
templates = Jinja2Templates(directory=templates_dir)

# Modules - these run LOCALLY on your laptop using the WebRTC video stream
# The commands spawn local processes that read from the shared frame queue
MODULES = {
    "yolo": {
        "cmd": ["python3", "-m", "src.laptop_client.modules.yolo"],
        "port": 8081,
        "name": "YOLO Object Detection",
        "description": "YOLOv8 running on your laptop GPU.",
    },
    "hand": {
        "cmd": ["python3", "-m", "src.laptop_client.modules.hand"],
        "port": 8082,
        "name": "Hand Detection",
        "description": "MediaPipe hand tracking on your laptop.",
    },
    "depth": {
        "cmd": ["python3", "-m", "src.laptop_client.modules.depth"],
        "port": 8083,
        "name": "Depth Camera",
        "description": "RealSense depth mapping with object distance.",
    },
    "simple": {
        "cmd": None,  # Built-in, no separate process
        "port": 8084,
        "name": "Raw Camera Feed",
        "description": "Direct WebRTC stream from the robot.",
    },
}


@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    host = request.headers.get("host", "localhost:8000").split(":")[0]
    return templates.TemplateResponse(
        "index.html", {
            "request": request,
            "modules": MODULES,
            "host": host,
            "robot_ip": ROBOT_IP,
        }
    )


@app.post("/start/{module_id}")
async def start_module(module_id: str):
    """Start a CV module running locally."""
    if module_id not in MODULES:
        return {"status": "error", "message": "Unknown module"}
    
    config = MODULES[module_id]
    
    # Simple module doesn't need a process - it's built-in
    if config["cmd"] is None:
        return {"status": "success", "message": "Built-in module"}
    
    cmd = config["cmd"].copy()
    # Pass robot IP to the module
    cmd.append(ROBOT_IP)
    
    success, msg = await manager.start_process(module_id, cmd)
    return {"status": "success" if success else "error", "message": msg}


@app.post("/stop/{module_id}")
async def stop_module(module_id: str):
    """Stop a CV module."""
    if module_id not in MODULES:
        return {"status": "error", "message": "Unknown module"}
    
    config = MODULES[module_id]
    if config["cmd"] is None:
        return {"status": "success", "message": "Built-in module"}
    
    success, msg = await manager.stop_process(module_id)
    return {"status": "success" if success else "error", "message": msg}


@app.get("/video_feed")
async def video_feed():
    """MJPEG stream of the raw robot camera via WebRTC."""
    def generate():
        while True:
            if bridge and bridge.is_connected:
                frame = bridge.get_frame(timeout=1.0)
                if frame is not None:
                    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                else:
                    time.sleep(0.01)
            else:
                time.sleep(0.5)
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/stats")
async def stats():
    """Return current video stats."""
    if bridge and bridge.is_connected:
        return {
            "fps": fps_stats["raw"]["fps"],
            "width": CONFIG["display"]["width"],
            "height": CONFIG["display"]["height"],
            "info": f"WebRTC → {ROBOT_IP}"
        }
    return {
        "fps": 0,
        "width": 0,
        "height": 0,
        "info": "Disconnected"
    }


@app.get("/status")
async def status():
    """Module status - same format as robot dashboard."""
    result = manager.get_status()
    # Simple is always "running" if WebRTC is connected
    if bridge and bridge.is_connected:
        result["simple"] = "running"
    return result


@app.get("/system_stats")
async def system_stats():
    """Local system stats (laptop CPU/RAM)."""
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    return {
        "cpu": cpu,
        "ram_percent": mem.percent,
        "ram_used": round(mem.used / (1024**3), 2),
        "ram_total": round(mem.total / (1024**3), 2)
    }


@app.get("/robot_stats")
async def robot_stats():
    """Robot telemetry from WebRTC data channel."""
    if bridge:
        return {
            "connected": bridge.is_connected,
            "battery": bridge.state.battery_soc,
            "temps": bridge.state.motor_temps
        }
    return {
        "connected": False,
        "battery": 0,
        "temps": []
    }


@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """Stream logs from modules and connection status."""
    await websocket.accept()
    queue = asyncio.Queue()
    manager.add_log_queue(queue)
    
    try:
        # Send initial connection status
        status = "Connected" if (bridge and bridge.is_connected) else "Connecting..."
        await websocket.send_text(f"[WebRTC] {status} | Robot: {ROBOT_IP}")
        
        while True:
            # Check for module logs
            try:
                data = await asyncio.wait_for(queue.get(), timeout=2.0)
                await websocket.send_text(data)
            except asyncio.TimeoutError:
                # Send periodic status updates
                if bridge:
                    status = "Connected" if bridge.is_connected else "Disconnected"
                    fps = fps_stats["raw"]["fps"]
                    await websocket.send_text(f"[WebRTC] {status} | FPS: {fps:.1f}")
    except WebSocketDisconnect:
        manager.remove_log_queue(queue)


@app.websocket("/ws/control")
async def control_endpoint(websocket: WebSocket):
    """Handle movement and action commands from the UI."""
    await websocket.accept()
    logger.info("Control WebSocket connected")
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if not bridge or not bridge.is_connected:
                continue
            
            # Check for specific actions
            if "action" in data:
                action = data["action"]
                logger.info(f"Received action: {action}")
                bridge.send_action(action)
            else:
                # Movement command
                vx = float(data.get("vx", 0.0))
                vy = float(data.get("vy", 0.0))
                vyaw = float(data.get("vyaw", 0.0))
                
                # Send via WebRTC data channel
                bridge.send_move(vx, vy, vyaw)
                
    except WebSocketDisconnect:
        logger.info("Control WebSocket disconnected")
        if bridge and bridge.is_connected:
            bridge.send_move(0, 0, 0)
    except Exception as e:
        logger.error(f"Control error: {e}")


def get_local_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def main():
    import uvicorn
    
    ip = get_local_ip()
    port = CONFIG["server"]["port"]
    
    logger.info("=== Laptop Client for Unitree Go2 ===")
    logger.info(f"Robot IP: {ROBOT_IP}")
    logger.info(f"Dashboard: http://{ip}:{port}")
    logger.info("=====================================")
    
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
