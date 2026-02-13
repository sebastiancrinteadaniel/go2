"""Configuration for Unified Dashboard Application - Using Config Classes"""

from dataclasses import dataclass
from typing import Optional


# ============================================================================
# Camera Configuration
# ============================================================================

@dataclass
class CameraConfig:
    """Camera capture settings"""
    source: str = "go2"      # "go2" or "usb"
    width: int = 640          # Display width
    height: int = 480         # Display height
    fps: int = 30             # Target capture FPS
    interface: str = "eth0"   # For Go2: "eth0", "wlan0", etc.


# ============================================================================
# YOLO Object Detection Configuration
# ============================================================================

@dataclass
class YoloConfig:
    """YOLO object detection settings"""
    enabled: bool = True
    name: str = "YOLO Object Detection"
    model_path: str = "src/yolo/model/yolov8n.engine"  # Try .engine first, fallback to .pt
    confidence: float = 0.5
    inference_width: int = 640   # Input width for inference
    inference_height: int = 640  # Input height for inference
    device: str = "auto"         # "auto" = try cuda then cpu, "cpu", or "cuda"
    skip_frames: int = 0


# ============================================================================
# Hand Detection Configuration
# ============================================================================

@dataclass
class HandDetectionConfig:
    """Hand detection settings"""
    enabled: bool = True
    name: str = "Hand Detection"
    max_hands: int = 2
    confidence: float = 0.5
    inference_width: int = 640
    inference_height: int = 640
    skip_frames: int = 0


# ============================================================================
# Depth Camera Configuration
# ============================================================================

@dataclass
class DepthCameraConfig:
    """RealSense depth camera settings"""
    enabled: bool = True
    name: str = "Depth Camera"
    depth_width: int = 640
    depth_height: int = 480
    color_width: int = 640          # Same as display width
    color_height: int = 480         # Same as display height
    fps: int = 30                   # Depth camera capture FPS
    skip_frames: int = 0            # Skip frames for faster processing
    yolo_enabled: bool = True       # Run YOLO on RGB stream


# ============================================================================
# Simple Camera Configuration
# ============================================================================

@dataclass
class SimpleCameraConfig:
    """Simple camera for low-latency streaming"""
    enabled: bool = True
    name: str = "Simple Camera"
    skip_frames: int = 0
    draw_fps: bool = True


# ============================================================================
# Video Streaming Configuration
# ============================================================================

@dataclass
class StreamingConfig:
    """MJPEG video streaming settings"""
    mjpeg_quality: int = 80          # 0-100, higher = better quality, more CPU
    mjpeg_frame_queue_size: int = 3  # Max frames buffered before dropping old ones
    enable_fps_stats: bool = True


# ============================================================================
# Server Configuration
# ============================================================================

@dataclass
class ServerConfig:
    """FastAPI web server settings"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    log_level: str = "info"


# ============================================================================
# Application Configuration
# ============================================================================

@dataclass
class AppConfig:
    """Main application configuration"""
    camera: CameraConfig
    yolo: YoloConfig
    hand_detection: HandDetectionConfig
    depth_camera: DepthCameraConfig
    simple_camera: SimpleCameraConfig
    streaming: StreamingConfig
    server: ServerConfig

    def __post_init__(self):
        """Validate configuration after initialization"""
        # Camera validation
        if self.camera.source not in ("go2", "usb"):
            raise ValueError("Camera source must be 'go2' or 'usb'")
        if self.camera.fps <= 0:
            raise ValueError("Camera FPS must be positive")
        
        # Inference size validation
        if self.yolo.inference_width <= 0 or self.yolo.inference_height <= 0:
            raise ValueError("YOLO inference dimensions must be positive")
        if self.hand_detection.inference_width <= 0 or self.hand_detection.inference_height <= 0:
            raise ValueError("Hand detection inference dimensions must be positive")
        
        # Streaming validation
        if not (0 <= self.streaming.mjpeg_quality <= 100):
            raise ValueError("MJPEG quality must be 0-100")
        
        # Server validation
        if not (1 <= self.server.port <= 65535):
            raise ValueError("Server port must be 1-65535")

    def get_device(self) -> str:
        """
        Get the appropriate device for inference (auto fallback logic).
        
        Returns:
            "cuda" if available and requested, else "cpu"
        """
        if self.yolo.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.yolo.device


# ============================================================================
# Default Configuration Instance
# ============================================================================

DEFAULT_CONFIG = AppConfig(
    camera=CameraConfig(
        source="go2",
        width=640,
        height=480,
        fps=30,
        interface="eth0",
    ),
    yolo=YoloConfig(
        enabled=False,
        model_path="src/yolo/model/yolov8n.engine",
        confidence=0.5,
        inference_width=640,
        inference_height=640,
        device="auto",
    ),
    hand_detection=HandDetectionConfig(
        enabled=False,
        confidence=0.5,
        inference_width=640,
        inference_height=640,
    ),
    depth_camera=DepthCameraConfig(
        enabled=False,
        color_width=640,
        color_height=480,
        fps=30,
        skip_frames=0,
    ),
    simple_camera=SimpleCameraConfig(
        enabled=False,
        draw_fps=True,
    ),
    streaming=StreamingConfig(
        mjpeg_quality=80,
        mjpeg_frame_queue_size=3,
    ),
    server=ServerConfig(
        host="0.0.0.0",
        port=8000,
        debug=False,
    ),
)


# ============================================================================
# Config Helper Functions
# ============================================================================

def get_config() -> AppConfig:
    """Get global config instance"""
    return DEFAULT_CONFIG


def get_active_processors(config: AppConfig) -> dict:
    """Get list of active processors from config"""
    processors = {}
    
    if config.yolo.enabled:
        processors["yolo"] = config.yolo
    if config.hand_detection.enabled:
        processors["hand"] = config.hand_detection
    if config.depth_camera.enabled:
        processors["depth"] = config.depth_camera
    if config.simple_camera.enabled:
        processors["simple"] = config.simple_camera
    
    return processors
