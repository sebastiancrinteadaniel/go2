"""Main FastAPI Application - Unified Async Dashboard"""

import asyncio
import logging
import logging.config
import cv2
import numpy as np
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, Response, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
    from av import VideoFrame
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

from src.unified_app.config import get_config, AppConfig
from src.unified_app.model_pool import ModelPool, get_model_pool
from src.unified_app.frame_router import FrameRouter
from src.unified_app.video_streaming import mjpeg_handler
from src.unified_app.processors import (
    YoloProcessor,
    HandDetectionProcessor,
    DepthCameraProcessor,
    SimpleCameraProcessor,
)

# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

if not WEBRTC_AVAILABLE:
    logger.warning("aiortc/av not installed - WebRTC streaming disabled. Install with: pip install aiortc av")

# ============================================================================
# WebRTC Streaming
# ============================================================================

class ProcessorStreamTrack(VideoStreamTrack):
    """WebRTC video track from processor output queue"""
    
    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.output_queue = processor.output_queue
        self.logger = logging.getLogger(f"WebRTC.{processor.name}")
    
    async def recv(self):
        """Get next video frame for WebRTC"""
        pts, time_base = await self.next_timestamp()
        
        try:
            frame = self.output_queue.get_nowait()
            if frame is not None and frame.size > 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
                video_frame.pts = pts
                video_frame.time_base = time_base
                return video_frame
        except asyncio.QueueEmpty:
            pass
        
        # Return black frame if no data
        black_frame = VideoFrame.from_ndarray(
            np.zeros((480, 640, 3), dtype="uint8"), format="rgb24"
        )
        black_frame.pts = pts
        black_frame.time_base = time_base
        return black_frame


# Track active WebRTC connections: {processor_id: {connection_id: RTCPeerConnection}}
webrtc_connections = {}

# ============================================================================
# Global State
# ============================================================================

global_state = {
    "config": None,  # Will be set in lifespan
    "frame_router": None,
    "model_pool": None,
    "processors": {},  # {processor_name: processor_instance}
    "processor_tasks": {},  # {processor_name: asyncio.Task}
}

# ============================================================================
# Lifespan Management
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Handles startup and shutdown.
    """

    # ========== STARTUP ==========
    logger.info("=" * 60)
    logger.info("STARTING UNIFIED DASHBOARD APPLICATION")
    logger.info("=" * 60)

    try:
        # Get configuration
        logger.info("Loading configuration...")
        global_state["config"] = get_config()

        # Initialize Frame Router
        logger.info(f"Initializing Frame Router (camera: {global_state['config'].camera.source})...")
        global_state["frame_router"] = FrameRouter(global_state["config"].camera)
        await global_state["frame_router"].initialize()

        # Initialize Processors
        logger.info("Initializing processors...")
        await initialize_processors()

        # Start Frame Router
        logger.info("Starting Frame Router...")
        await global_state["frame_router"].start()

        logger.info("=" * 60)
        logger.info("STARTUP COMPLETE")
        logger.info("=" * 60)

        yield  # Application runs here

    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        raise

    finally:
        # ========== SHUTDOWN ==========
        logger.info("=" * 60)
        logger.info("SHUTTING DOWN UNIFIED DASHBOARD APPLICATION")
        logger.info("=" * 60)

        # Stop all processors
        await stop_all_processors()

        # Stop frame router
        if global_state["frame_router"]:
            await global_state["frame_router"].stop()

        # Cleanup model pool
        if global_state["model_pool"]:
            global_state["model_pool"].unload_all()

        logger.info("=" * 60)
        logger.info("SHUTDOWN COMPLETE")
        logger.info("=" * 60)


# ============================================================================
# Processor Management
# ============================================================================


async def initialize_processors():
    """Initialize all enabled processors"""
    config = global_state["config"]
    processors_to_create = {
        "yolo": YoloProcessor,
        "hand": HandDetectionProcessor,
        "depth": DepthCameraProcessor,
        "simple": SimpleCameraProcessor,
    }

    for proc_id, proc_class in processors_to_create.items():
        # Get processor config from AppConfig
        if proc_id == "yolo":
            proc_config = config.yolo
        elif proc_id == "hand":
            proc_config = config.hand_detection
        elif proc_id == "depth":
            proc_config = config.depth_camera
        elif proc_id == "simple":
            proc_config = config.simple_camera
        else:
            continue

        if not proc_config.enabled:
            logger.info(f"Processor {proc_id} disabled in config")
            continue

        try:
            logger.info(f"Creating processor: {proc_id}")
            processor = proc_class(proc_config)
            global_state["processors"][proc_id] = processor

            # Register with frame router
            input_queue = await global_state["frame_router"].register_processor(
                proc_id
            )

            # Create output queue for streaming
            output_queue = asyncio.Queue(
                maxsize=config.streaming.mjpeg_frame_queue_size
            )

            # Start processor
            await processor.start(
                camera=None,  # Uses frame router's camera
                input_queue=input_queue,
                output_queue=output_queue,
                model_pool=global_state["model_pool"],
            )

            # Store output queue for streaming
            processor.output_queue = output_queue

        except Exception as e:
            logger.error(f"Failed to initialize processor {proc_id}: {e}", exc_info=True)


async def stop_all_processors():
    """Stop all running processors"""
    for proc_id, processor in global_state["processors"].items():
        try:
            logger.info(f"Stopping processor: {proc_id}")
            await processor.stop()
        except Exception as e:
            logger.error(f"Error stopping processor {proc_id}: {e}")

    global_state["processors"].clear()
    global_state["processor_tasks"].clear()


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Unified Go2 Dashboard",
    description="Single-process async dashboard for Go2 robot",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# HTTP Routes
# ============================================================================


@app.get("/")
async def root():
    """Serve dashboard HTML"""
    return FileResponse("src/unified_app/templates/index.html")


@app.get("/api/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "processors": len(global_state["processors"]),
        "frame_router_running": (
            global_state["frame_router"].running
            if global_state["frame_router"]
            else False
        ),
    }


@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    frame_router = global_state["frame_router"]
    processors = global_state["processors"]

    stats = {
        "frame_router": frame_router.get_stats() if frame_router else None,
        "processors": {
            proc_id: processor.get_stats()
            for proc_id, processor in processors.items()
        },
    }

    return stats


@app.get("/api/processors")
async def list_processors():
    """List all processors and their status"""
    processors_info = []

    for proc_id, processor in global_state["processors"].items():
        processors_info.append(
            {
                "id": proc_id,
                "name": processor.name,
                "running": processor.running,
                "stats": processor.get_stats(),
            }
        )

    return processors_info


@app.post("/api/processors/{processor_id}/start")
async def start_processor(processor_id: str):
    """Start a processor"""
    if processor_id not in global_state["processors"]:
        raise HTTPException(status_code=404, detail=f"Processor {processor_id} not found")

    processor = global_state["processors"][processor_id]

    if processor.running:
        return {"status": "already_running"}

    try:
        # Get registered queue
        input_queue = await global_state["frame_router"].register_processor(
            processor_id
        )

        # Start processor
        output_queue = asyncio.Queue(maxsize=config.MJPEG_FRAME_QUEUE_SIZE)
        await processor.start(
            camera=None,
            input_queue=input_queue,
            output_queue=output_queue,
            model_pool=global_state["model_pool"],
        )
        processor.output_queue = output_queue

        return {"status": "started", "processor": processor_id}

    except Exception as e:
        logger.error(f"Error starting processor {processor_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/processors/{processor_id}/stop")
async def stop_processor(processor_id: str):
    """Stop a processor"""
    if processor_id not in global_state["processors"]:
        raise HTTPException(status_code=404, detail=f"Processor {processor_id} not found")

    processor = global_state["processors"][processor_id]

    if not processor.running:
        return {"status": "already_stopped"}

    try:
        await processor.stop()
        await global_state["frame_router"].unregister_processor(processor_id)
        return {"status": "stopped", "processor": processor_id}

    except Exception as e:
        logger.error(f"Error stopping processor {processor_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WebRTC Routes
# ============================================================================

@app.post("/api/webrtc/offer")
async def webrtc_offer(request: Request):
    """
    WebRTC offer/answer endpoint.
    Browser sends offer, server responds with answer.
    Enables low-bandwidth compressed video streaming.
    """
    if not WEBRTC_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="WebRTC not available - install packages: pip install aiortc av"
        )
    
    try:
        params = await request.json()
        processor_id = params.get("processor")
        
        if not processor_id or processor_id not in global_state["processors"]:
            raise HTTPException(status_code=404, detail="Processor not found")
        
        processor = global_state["processors"][processor_id]
        
        if not processor.running:
            raise HTTPException(status_code=400, detail="Processor not running")
        
        # Create WebRTC peer connection
        pc = RTCPeerConnection()
        connection_id = str(uuid.uuid4())
        
        # Track this connection
        if processor_id not in webrtc_connections:
            webrtc_connections[processor_id] = {}
        webrtc_connections[processor_id][connection_id] = pc
        
        # Add video track from processor
        pc.addTrack(ProcessorStreamTrack(processor))
        
        # Handle remote offer from client
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        await pc.setRemoteDescription(offer)
        
        # Create and send answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        logger.info(f"WebRTC connection established for {processor_id} ({connection_id})")
        
        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "connection_id": connection_id
        }
        
    except Exception as e:
        logger.error(f"WebRTC offer error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

# ============================================================================
# Video Streaming Routes
# ============================================================================


@app.get("/api/streams/{stream_id}/video_feed")
async def video_feed(stream_id: str):
    """
    MJPEG video stream endpoint for a processor.

    Args:
        stream_id: Processor ID (e.g., "yolo", "hand_detection")

    Returns:
        StreamingResponse with MJPEG frames
    """
    if stream_id not in global_state["processors"]:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")

    processor = global_state["processors"][stream_id]

    if not processor.running or not hasattr(processor, "output_queue"):
        raise HTTPException(
            status_code=503,
            detail=f"Stream {stream_id} is not running",
        )

    # Create streaming response
    config = global_state["config"]
    return StreamingResponse(
        mjpeg_handler(processor.output_queue, quality=config.streaming.mjpeg_quality),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ============================================================================
# Startup Message
# ============================================================================


@app.on_event("startup")
async def startup_message():
    """Print startup message"""
    config = global_state["config"]
    logger.info("\n" + "=" * 60)
    logger.info("UNIFIED DASHBOARD RUNNING")
    logger.info("=" * 60)
    logger.info(f"Server: http://{config.server.host}:{config.server.port}")
    logger.info(f"Camera: {config.camera.source}")
    logger.info(f"Processors: {', '.join(global_state['processors'].keys())}")
    logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    import uvicorn

    config = get_config()
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level="info" if not config.server.debug else "debug",
    )
