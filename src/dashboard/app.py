import sys
import asyncio
import logging
import psutil
import time
import os

# Global config for the interface
INTERFACE = None
# Simple argument parsing
for arg in sys.argv[1:]:
    if not arg.startswith("-"):
        INTERFACE = arg
        break

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
from .process_manager import ProcessManager

try:
    from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

manager = ProcessManager()
sport_client = None

class RobotTelemetry:
    def __init__(self):
        self.battery_soc = 0
        self.motor_temps = []
        self.connected = False
        self.last_update = 0
        self.subscriber = None

    def on_low_state(self, msg: LowState_):
        self.battery_soc = msg.bms_state.soc
        # Go2 has 12 motors usually
        self.motor_temps = [m.temperature for m in msg.motor_state[:12]]
        self.connected = True
        self.last_update = time.time()

    def start(self):
        if SDK_AVAILABLE:
            self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
            self.subscriber.Init(self.on_low_state, 10)
            logger.info("Robot telemetry subscriber initialized.")

    def stop(self):
        if self.subscriber:
            self.subscriber.Close()

telemetry = RobotTelemetry()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    # Clean up any lingering processes from previous runs
    manager.cleanup_orphans()
    
    global sport_client
    if SDK_AVAILABLE:
        try:
            if INTERFACE:
                ChannelFactoryInitialize(0, INTERFACE)
                logger.info(f"Initialized SDK with interface: {INTERFACE}")
            else:
                # Auto-detect environment
                # On Jetson (Robot), the internal interface is usually eth0
                # We check for Jetson-specific file or if we are on a Unitree board
                is_jetson = os.path.exists("/etc/nv_tegra_release")
                has_eth0 = "eth0" in psutil.net_if_addrs()
                
                if is_jetson and has_eth0:
                    ChannelFactoryInitialize(0, "eth0")
                    logger.info("Detected Jetson environment. Initialized SDK with eth0.")
                else:
                    ChannelFactoryInitialize(0, "lo")
                    logger.info("Initialized SDK with loopback interface")
            
            telemetry.start()
            
            sport_client = SportClient()
            sport_client.SetTimeout(10.0)
            sport_client.Init()
            logger.info("SportClient initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Unitree SDK: {e}")
    
    yield
    
    # Shutdown
    manager.stop_all()
    if SDK_AVAILABLE:
        telemetry.stop()

app = FastAPI(lifespan=lifespan)

current_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(current_dir, "templates")
templates = Jinja2Templates(directory=templates_dir)

MODULES = {
    "yolo": {
        "cmd": ["python3", "-m", "src.yolo.webrtc"],
        "port": 8081,
        "name": "YOLO Object Detection",
        "description": "YOLOv8 object detection with bounding boxes.",
    },
    "hand": {
        "cmd": ["python3", "-m", "src.hand_detection.webrtc"],
        "port": 8082,
        "name": "Hand Detection",
        "description": "MediaPipe hand tracking and gesture recognition.",
    },
    "depth": {
        "cmd": ["python3", "-m", "src.depth_camera.webrtc"],
        "port": 8083,
        "name": "Depth Camera",
        "description": "RealSense depth mapping with object distance.",
    },
    "simple": {
        "cmd": ["python3", "-m", "src.simple_camera.webrtc"],
        "port": 8084,
        "name": "Simple Camera",
        "description": "Raw camera feed without AI processing.",
    },
}


@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    # Get current host to construct video URLs correctly
    host = request.headers.get("host", "localhost:8000").split(":")[0]
    return templates.TemplateResponse(
        "index.html", {"request": request, "modules": MODULES, "host": host}
    )


@app.post("/start/{module_id}")
async def start_module(module_id: str):
    if module_id not in MODULES:
        return {"status": "error", "message": "Unknown module"}

    config = MODULES[module_id]
    cmd = config["cmd"].copy()
    if INTERFACE:
        cmd.append(INTERFACE)

    success, msg = await manager.start_process(module_id, cmd)
    return {"status": "success" if success else "error", "message": msg}


@app.post("/stop/{module_id}")
async def stop_module(module_id: str):
    success, msg = await manager.stop_process(module_id)
    return {"status": "success" if success else "error", "message": msg}


@app.get("/status")
async def status():
    return manager.get_status()

@app.get("/system_stats")
async def system_stats():
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
    # Check if data is stale (older than 2 seconds)
    is_stale = (time.time() - telemetry.last_update) > 2.0
    return {
        "connected": telemetry.connected and not is_stale,
        "battery": telemetry.battery_soc,
        "temps": telemetry.motor_temps
    }

@app.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    queue = asyncio.Queue()
    manager.add_log_queue(queue)
    try:
        while True:
            data = await queue.get()
            await websocket.send_text(data)
    except WebSocketDisconnect:
        manager.remove_log_queue(queue)

@app.websocket("/ws/control")
async def control_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            if sport_client:
                # Only send commands if robot is connected to avoid log spam
                if telemetry.connected:
                    # Check for specific actions
                    if "action" in data:
                        action = data["action"]
                        logger.info(f"Received action: {action}")
                        if action == "stand_up":
                            sport_client.StandUp()
                            # Switch to BalanceStand (Normal Mode) to enable velocity control
                            sport_client.BalanceStand()
                        elif action == "stand_down":
                            sport_client.StandDown()
                        elif action == "recovery":
                            sport_client.RecoveryStand()
                        elif action == "damp":
                            sport_client.Damp()
                        elif action == "hello":
                            sport_client.Hello()
                        elif action == "stretch":
                            sport_client.Stretch()
                        elif action == "dance1":
                            sport_client.Dance1()
                        elif action == "dance2":
                            sport_client.Dance2()
                        elif action == "heart":
                            sport_client.Heart()
                        elif action == "scrape":
                            sport_client.Scrape()
                        elif action == "front_jump":
                            sport_client.FrontJump()
                    else:
                        # Movement command
                        vx = float(data.get("vx", 0.0))
                        vy = float(data.get("vy", 0.0))
                        vyaw = float(data.get("vyaw", 0.0))
                        pitch = float(data.get("pitch", 0.0))

                        if abs(vx) > 0.01 or abs(vy) > 0.01 or abs(vyaw) > 0.01:
                            sport_client.Move(vx, vy, vyaw)
                        elif abs(pitch) > 0.01:
                            sport_client.Euler(0, pitch, 0)
                        else:
                            sport_client.Move(0, 0, 0)
                else:
                    # Optional: Send a warning back to UI?
                    pass
    except WebSocketDisconnect:
        if sport_client and telemetry.connected:
            try:
                sport_client.Move(0.0, 0.0, 0.0)
                logger.info("Control client disconnected, stopping robot")
            except Exception:
                pass
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
    logger.info(f"Starting Dashboard. Interface: {INTERFACE}")
    logger.info(f"Dashboard available at http://{ip}:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
