from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
import os
import psutil
import asyncio
import json
import time
import logging
from aiortc import RTCPeerConnection, RTCSessionDescription

from app.core.config import settings
from app.services.video import CameraStreamTrack, Go2CameraStreamTrack
from app.services.telemetry import telemetry

logger = logging.getLogger(__name__)

router = APIRouter()

JOINT_NAMES = [
    "FR_0", "FR_1", "FR_2",
    "FL_0", "FL_1", "FL_2",
    "RR_0", "RR_1", "RR_2",
    "RL_0", "RL_1", "RL_2",
]


@router.get("/", response_class=HTMLResponse)
async def index():
    """
    Serve the main dashboard UI.
    """
    index_path = os.path.join(settings.STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return f.read()
    return "<h1>Index.html not found in static folder</h1>"


@router.post("/offer")
async def offer(request: Request):
    """
    Handle WebRTC offer from the frontend to establish a video stream connection.
    """
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()

    # Initialize telemetry when WebRTC connection is established
    telemetry.init()

    @pc.on("datachannel")
    def on_datachannel(channel):
        logger.info(f"Server data channel received: {channel.label}")

        async def send_telemetry():
            while True:
                if channel.readyState == "open":
                    cpu_percent = psutil.cpu_percent(interval=None)
                    ram = psutil.virtual_memory()
                    uptime_seconds = int(time.time() - psutil.boot_time())
                    fps = getattr(camera_track, "current_fps", 0.0)
                    detections = getattr(camera_track, "latest_detections", [])
                    gestures = getattr(camera_track, "latest_gestures", [])
                    motor_temps = telemetry.motor_temps
                    avg_temp_c = (
                        sum(motor_temps) / len(motor_temps)
                        if motor_temps
                        else None
                    )
                    peak_temp_c = max(motor_temps) if motor_temps else None
                    peak_joint_name = None
                    if motor_temps:
                        peak_idx = max(range(len(motor_temps)), key=lambda i: motor_temps[i])
                        peak_joint_name = (
                            JOINT_NAMES[peak_idx]
                            if peak_idx < len(JOINT_NAMES)
                            else f"J{peak_idx}"
                        )
                    
                    data = json.dumps({
                        "type": "stats", 
                        "cpu_percent": cpu_percent, 
                        "ram_percent": ram.percent, 
                        "uptime": uptime_seconds, 
                        "detections": detections, 
                        "gestures": gestures,
                        "fps": fps,
                        "gesture_dispatch_enabled": getattr(getattr(camera_track, "gesture_dispatcher", None), "enabled", False),
                        "battery": telemetry.battery_soc,
                        "connected": telemetry.connected and ((time.time() - telemetry.last_update) < 2.0),
                        "motor_temps": motor_temps,
                        "avg_temp_c": avg_temp_c,
                        "peak_temp_c": peak_temp_c,
                        "peak_joint_name": peak_joint_name,
                        "travel_speed_mps": telemetry.travel_speed_mps,
                    })
                    try:
                        channel.send(data)
                    except Exception as e:
                        logger.error(f"Error sending telemetry: {e}")
                        break
                elif channel.readyState == "closed":
                    break
                await asyncio.sleep(1)

        asyncio.create_task(send_telemetry())

        @channel.on("message")
        def on_message(message):
            logger.debug(f"Received message: {message}")
            if message == "ping":
                channel.send("pong")
            elif message == "toggle_yolo":
                camera_track.yolo_processor.enabled = not camera_track.yolo_processor.enabled
            elif message == "toggle_gesture":
                camera_track.gesture_processor.enabled = not camera_track.gesture_processor.enabled
            elif message == "toggle_gesture_dispatch":
                dispatcher = getattr(camera_track, "gesture_dispatcher", None)
                if dispatcher is not None:
                    dispatcher.enabled = not dispatcher.enabled
                    logger.info(
                        "Gesture dispatch %s",
                        "enabled" if dispatcher.enabled else "disabled",
                    )

    mode = params.get("mode", "hd_view")

    if mode == "go2":
        camera_track = Go2CameraStreamTrack()
    else:
        camera_track = CameraStreamTrack()

    pc.addTrack(camera_track)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

