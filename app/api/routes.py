from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import os
import io
import cv2
import psutil
import asyncio
import json
import time
import logging
from datetime import datetime
from aiortc import RTCPeerConnection, RTCSessionDescription

from app.core.config import settings
from app.services.video import CameraSource, Go2CameraSource, ViewerTrack
from app.services.telemetry import telemetry
from app.services.report_generator import ReportData, build_pdf, next_report_id

logger = logging.getLogger(__name__)

router = APIRouter()

_active_source: CameraSource | Go2CameraSource | None = None
_active_pcs: set[RTCPeerConnection] = set()
_active_mode: str = "hd_view"


async def _close_pc(pc: RTCPeerConnection) -> None:
    """Remove a single peer connection. Stop the source only when no viewers remain."""
    global _active_source, _active_pcs
    _active_pcs.discard(pc)
    await pc.close()
    if not _active_pcs and _active_source is not None:
        _active_source.stop()
        _active_source = None


async def close_all() -> None:
    """Tear down all peer connections and release the camera source."""
    global _active_source, _active_pcs
    for pc in list(_active_pcs):
        await pc.close()
    _active_pcs.clear()
    if _active_source is not None:
        _active_source.stop()
        _active_source = None


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
    Multiple viewers can connect simultaneously — they share one camera source.
    """
    global _active_source, _active_pcs, _active_mode

    params = await request.json()
    mode = params.get("mode", "hd_view")

    # If the mode changed, tear down everything and start fresh
    if mode != _active_mode and _active_source is not None:
        await close_all()

    _active_mode = mode

    # Start the shared camera source once; reuse it for subsequent viewers
    if _active_source is None:
        if mode == "go2":
            _active_source = Go2CameraSource()
        else:
            _active_source = CameraSource()
        telemetry.init()

    source = _active_source

    offer_desc = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()
    _active_pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("WebRTC connection state: %s", pc.connectionState)
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await _close_pc(pc)

    @pc.on("datachannel")
    def on_datachannel(channel):
        logger.info(f"Server data channel received: {channel.label}")

        async def send_telemetry():
            while True:
                if channel.readyState == "open":
                    cpu_percent = psutil.cpu_percent(interval=None)
                    ram = psutil.virtual_memory()
                    uptime_seconds = int(time.time() - psutil.boot_time())
                    fps = getattr(source, "current_fps", 0.0)
                    detections = getattr(source, "latest_detections", [])
                    weapons_detections = getattr(source, "latest_weapons_detections", [])
                    industrial_detections = getattr(source, "latest_industrial_detections", [])
                    gestures = getattr(source, "latest_gestures", [])
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

                    _dispatcher = getattr(source, "gesture_dispatcher", None)
                    dispatched_gesture = _dispatcher.pop_last_dispatch() if _dispatcher else None
                    data = json.dumps({
                        "type": "stats",
                        "initializing": getattr(source, "_initializing", False),
                        "camera_connected": getattr(source, "connected", True),
                        "cpu_percent": cpu_percent,
                        "ram_percent": ram.percent,
                        "uptime": uptime_seconds,
                        "detections": detections,
                        "weapons_detections": weapons_detections,
                        "industrial_detections": industrial_detections,
                        "gestures": gestures,
                        "fps": fps,
                        "dispatched_gesture": dispatched_gesture,
                        "yolo_enabled": source.yolo_processor.enabled,
                        "weapons_enabled": source.weapons_processor.enabled,
                        "industrial_enabled": source.industrial_processor.enabled,
                        "gesture_enabled": source.gesture_processor.enabled,
                        "gesture_dispatch_enabled": getattr(_dispatcher, "enabled", False),
                        "battery": telemetry.battery_soc,
                        "connected": telemetry.connected and ((time.time() - telemetry.last_update) < 2.0),
                        "motor_temps": motor_temps,
                        "avg_temp_c": avg_temp_c,
                        "peak_temp_c": peak_temp_c,
                        "peak_joint_name": peak_joint_name,
                        "imu_rpy": telemetry.imu_rpy,
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
                source.yolo_processor.enabled = not source.yolo_processor.enabled
            elif message == "toggle_weapons":
                source.weapons_processor.enabled = not source.weapons_processor.enabled
            elif message == "toggle_industrial":
                source.industrial_processor.enabled = not source.industrial_processor.enabled
            elif message == "toggle_gesture":
                source.gesture_processor.enabled = not source.gesture_processor.enabled
            elif message == "toggle_gesture_dispatch":
                dispatcher = getattr(source, "gesture_dispatcher", None)
                if dispatcher is not None:
                    dispatcher.enabled = not dispatcher.enabled
                    logger.info(
                        "Gesture dispatch %s",
                        "enabled" if dispatcher.enabled else "disabled",
                    )

    viewer_track = ViewerTrack(source)
    pc.addTrack(viewer_track)

    await pc.setRemoteDescription(offer_desc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


@router.post("/report")
async def generate_report(request: Request):
    """
    Snapshot current session telemetry + a live frame and return a QC PDF download.
    """
    body = await request.json()
    operator = body.get("operator", "Unknown")
    location = body.get("location", "—")

    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    uptime = int(time.time() - psutil.boot_time())

    motor_temps = telemetry.motor_temps
    avg_temp = sum(motor_temps) / len(motor_temps) if motor_temps else None
    peak_temp = max(motor_temps) if motor_temps else None
    peak_joint = None
    if motor_temps:
        idx = max(range(len(motor_temps)), key=lambda i: motor_temps[i])
        peak_joint = JOINT_NAMES[idx] if idx < len(JOINT_NAMES) else f"J{idx}"

    report_detections = list(getattr(_active_source, "session_detections", {}).values())
    frame_jpeg = None
    if _active_source is not None:
        try:
            frame = _active_source.get_latest_frame()
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_jpeg = buf.tobytes()
        except Exception:
            pass

    camera_source_map = {"go2": "Go2 Camera"}
    robot_connected = telemetry.connected and (time.time() - telemetry.last_update) < 2.0

    data = ReportData(
        report_id=next_report_id(),
        generated_at=datetime.now().astimezone(),
        operator=operator,
        location=location,
        mode=_active_mode,
        cpu_percent=cpu,
        ram_percent=ram.percent,
        uptime_seconds=uptime,
        battery_soc=telemetry.battery_soc or None,
        robot_connected=robot_connected,
        frame_rate=getattr(_active_source, "current_fps", 0.0),
        camera_source=camera_source_map.get(_active_mode, "Generic USB Cam"),
        imu_roll_rad=telemetry.imu_rpy[0] if telemetry.imu_rpy else None,
        imu_pitch_rad=telemetry.imu_rpy[1] if telemetry.imu_rpy else None,
        imu_yaw_rad=telemetry.imu_rpy[2] if telemetry.imu_rpy else None,
        detections=report_detections,
        motor_temps=motor_temps,
        avg_temp_c=avg_temp,
        peak_temp_c=peak_temp,
        peak_joint_name=peak_joint,
        frame_jpeg=frame_jpeg,
    )

    pdf_bytes = build_pdf(data)
    filename = f"QC-{data.generated_at.strftime('%Y%m%d')}-{data.report_id}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
