import cv2
import numpy as np
import asyncio
import logging
import threading
import time
import os
import sys
from queue import Queue, Empty
from aiohttp import web

from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)
from aiortc import MediaStreamTrack

# DDS imports for Jetson Native mode
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient

from .config import CONFIG

import socket

# Enable logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_local_ip():
    try:
        # Connect to a public DNS server to determine the best local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def get_connection() -> UnitreeWebRTCConnection:
    cfg = CONFIG["webrtc"]
    method_str = cfg["method"]
    
    if method_str == "LocalSTA":
        method = WebRTCConnectionMethod.LocalSTA
        ip = cfg["ip"]
        serial = cfg["serial_number"]
        if serial:
             return UnitreeWebRTCConnection(method, ip=ip, serialNumber=serial)
        return UnitreeWebRTCConnection(method, ip=ip)
        
    elif method_str == "LocalAP":
        method = WebRTCConnectionMethod.LocalAP
        return UnitreeWebRTCConnection(method)
        
    elif method_str == "Remote":
        method = WebRTCConnectionMethod.Remote
        serial = cfg["serial_number"]
        username = cfg["username"]
        password = cfg["password"]
        return UnitreeWebRTCConnection(method, serialNumber=serial, username=username, password=password)
    
    else:
        raise ValueError(f"Unknown WebRTC connection method: {method_str}")

def run():
    cfg = CONFIG["webrtc"]
    width = cfg["width"]
    height = cfg["height"]
    source = cfg.get("source", "WebRTC")
    
    frame_queue = Queue(maxsize=10)
    stop_event = threading.Event()

    # --- DDS / Native Mode (Jetson or PC via Ethernet) ---
    if source == "DDS":
        logger.info("Starting in DDS mode...")
        
        # Initialize DDS with interface if provided (e.g., python -m src.webrtc.run eth0)
        try:
            if len(sys.argv) > 1:
                iface = sys.argv[1]
                logger.info(f"Initializing DDS on interface: {iface}")
                ChannelFactoryInitialize(0, iface)
            else:
                ChannelFactoryInitialize(0)
        except Exception as e:
            logger.warning(f"DDS Init warning (might be already initialized): {e}")

        client = VideoClient()
        client.SetTimeout(3.0)
        client.Init()

        def dds_capture_loop():
            logger.info("DDS capture loop started.")
            while not stop_event.is_set():
                code, data = client.GetImageSample()
                if code == 0:
                    try:
                        # Convert bytes to numpy
                        image_data = np.frombuffer(bytes(data), dtype=np.uint8)
                        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                        
                        if img is not None:
                            if img.shape[1] != width or img.shape[0] != height:
                                img = cv2.resize(img, (width, height))
                            
                            if frame_queue.full():
                                try:
                                    frame_queue.get_nowait()
                                except Empty:
                                    pass
                            frame_queue.put(img)
                    except Exception as e:
                        logger.error(f"DDS decode error: {e}")
                else:
                    time.sleep(0.1) # Wait a bit if no frame
            logger.info("DDS capture loop stopped.")

        # Start DDS thread
        capture_thread = threading.Thread(target=dds_capture_loop, daemon=True)
        capture_thread.start()

        # Dummy asyncio setup for server compatibility
        async def setup_dummy():
            pass
        setup_func = setup_dummy

    # --- WebRTC Mode (Remote) ---
    else:
        conn = get_connection()
        logger.info(f"Connecting using method: {CONFIG['webrtc']['method']}")

        # Async function to receive video frames
        async def recv_camera_stream(track: MediaStreamTrack):
            while not stop_event.is_set():
                try:
                    frame = await track.recv()
                    img = frame.to_ndarray(format="bgr24")
                    
                    # Resize if needed
                    if img.shape[1] != width or img.shape[0] != height:
                        img = cv2.resize(img, (width, height))
                    
                    if frame_queue.full():
                        try:
                            frame_queue.get_nowait()
                        except Empty:
                            pass
                    frame_queue.put(img)
                except Exception as e:
                    logger.error(f"Error receiving frame: {e}")
                    break

        async def setup_webrtc():
            try:
                await conn.connect()
                conn.video.switchVideoChannel(True)
                conn.video.add_track_callback(recv_camera_stream)
                logger.info("WebRTC connected and video stream started.")
            except Exception as e:
                logger.error(f"Error in WebRTC connection: {e}")
                stop_event.set()
        
        setup_func = setup_webrtc


    # Web Server Handlers
    async def index(request):
        path = os.path.join(os.path.dirname(__file__), 'web', 'index.html')
        with open(path, 'r') as f:
            content = f.read()
        return web.Response(text=content, content_type='text/html')

    async def video_feed(request):
        response = web.StreamResponse()
        response.content_type = 'multipart/x-mixed-replace; boundary=frame'
        await response.prepare(request)

        try:
            while True:
                if not frame_queue.empty():
                    frame = frame_queue.get()
                    _, jpeg = cv2.imencode('.jpg', frame)
                    data = jpeg.tobytes()
                    
                    await response.write(b'--frame\r\n')
                    await response.write(b'Content-Type: image/jpeg\r\n\r\n')
                    await response.write(data)
                    await response.write(b'\r\n')
                else:
                    await asyncio.sleep(0.01)
        except Exception:
            pass
        return response

    def run_asyncio_loop(loop):
        asyncio.set_event_loop(loop)

        async def setup():
            try:
                await conn.connect()
                conn.video.switchVideoChannel(True)
                conn.video.add_track_callback(recv_camera_stream)
                logger.info("WebRTC connected and video stream started.")
            except Exception as e:
                logger.error(f"Error in WebRTC connection: {e}")
                stop_event.set()

        # Setup Web Server
        app = web.Application()
        app.router.add_get('/', index)
        app.router.add_get('/video_feed', video_feed)
        runner = web.AppRunner(app)
        
        async def start_server():
            await runner.setup()
            site = web.TCPSite(runner, '0.0.0.0', 8080)
            await site.start()
            local_ip = get_local_ip()
            logger.info(f"Web server started at http://localhost:8080")
            logger.info(f"To view on another device, visit http://{local_ip}:8080")

        loop.run_until_complete(setup_func())
        loop.run_until_complete(start_server())
        loop.run_forever()

    loop = asyncio.new_event_loop()
    asyncio_thread = threading.Thread(target=run_asyncio_loop, args=(loop,), daemon=True)
    asyncio_thread.start()

    try:
        while not stop_event.is_set():
            time.sleep(1)
            if not asyncio_thread.is_alive():
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join(timeout=1.0)
        logger.info("Exiting...")

if __name__ == "__main__":
    run()
