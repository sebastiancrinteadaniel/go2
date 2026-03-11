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
                    
                    data = json.dumps({
                        "type": "stats", 
                        "cpu_percent": cpu_percent, 
                        "ram_percent": ram.percent, 
                        "uptime": uptime_seconds, 
                        "detections": detections, 
                        "fps": fps,
                        "battery": telemetry.battery_soc,
                        "connected": telemetry.connected and ((time.time() - telemetry.last_update) < 2.0),
                        "motor_temps": telemetry.motor_temps
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

