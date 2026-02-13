"""Camera Abstractions - Self-contained camera implementations"""

import logging
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BaseCamera(ABC):
    """Abstract base class for camera sources"""

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.

        Returns:
            (success: bool, frame: np.ndarray or None)
        """
        pass

    @abstractmethod
    def release(self):
        """Release camera resources"""
        pass


class OpenCVCamera(BaseCamera):
    """OpenCV-based camera for USB or video files"""

    def __init__(self, source: int | str = 0):
        """
        Initialize OpenCV camera.

        Args:
            source: Camera index (0, 1, ...) or video file path
        """
        self.source = source
        self.cap = None
        self.logger = logging.getLogger("OpenCVCamera")

        try:
            self.cap = cv2.VideoCapture(source)

            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera: {source}")

            # Set properties for better performance
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

            self.logger.info(f"OpenCV camera initialized: {source}")

        except Exception as e:
            self.logger.error(f"Error initializing OpenCV camera: {e}")
            raise

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from camera"""
        if self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        """Release camera"""
        if self.cap:
            self.cap.release()
            self.logger.info("OpenCV camera released")


class Go2VideoClient(BaseCamera):
    """Go2 robot video client using unitree SDK"""

    def __init__(self, interface: str = "eth0"):
        """
        Initialize Go2 video client.

        Args:
            interface: Network interface to use (eth0, wlan0, etc.)
        """
        self.interface = interface
        self.logger = logging.getLogger("Go2VideoClient")
        self.client = None
        self.frame_queue = None

        try:
            # Import unitree SDK
            from unitree_sdk2py.go2.video.video_client import VideoClient

            self.logger.info(f"Initializing Go2 VideoClient on {interface}")

            # Create video client
            self.client = VideoClient(interface)
            self.client.start_recv()

            self.logger.info("Go2 VideoClient initialized")

        except ImportError as e:
            self.logger.error(
                f"unitree_sdk2py not available: {e}. "
                "Install with: pip install unitree_sdk2py"
            )
            raise
        except Exception as e:
            self.logger.error(f"Error initializing Go2 VideoClient: {e}")
            raise

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from Go2 camera"""
        if self.client is None:
            return False, None

        try:
            frame = self.client.frame
            if frame is not None:
                return True, frame
            return False, None
        except Exception as e:
            self.logger.error(f"Error reading frame from Go2: {e}")
            return False, None

    def release(self):
        """Release Go2 video client"""
        if self.client:
            try:
                self.client.stop_recv()
                self.logger.info("Go2 VideoClient stopped")
            except Exception as e:
                self.logger.error(f"Error stopping Go2 VideoClient: {e}")


class FileCamera(BaseCamera):
    """File-based camera for video file playback"""

    def __init__(self, file_path: str, loop: bool = True):
        """
        Initialize file-based camera.

        Args:
            file_path: Path to video file
            loop: Whether to loop video when it ends
        """
        self.file_path = file_path
        self.loop = loop
        self.cap = None
        self.logger = logging.getLogger("FileCamera")

        try:
            self.cap = cv2.VideoCapture(file_path)

            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video file: {file_path}")

            self.logger.info(f"File camera initialized: {file_path}")

        except Exception as e:
            self.logger.error(f"Error initializing file camera: {e}")
            raise

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from video file"""
        if self.cap is None:
            return False, None

        ret, frame = self.cap.read()

        # Handle end of file
        if not ret and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
            ret, frame = self.cap.read()

        return ret, frame

    def release(self):
        """Release camera"""
        if self.cap:
            self.cap.release()
            self.logger.info("File camera released")


def create_camera(source: str = "go2", interface: str = "eth0") -> BaseCamera:
    """
    Factory function to create appropriate camera instance.

    Args:
        source: Camera source type ("go2", "usb", "file")
        interface: Network interface for Go2 (e.g., "eth0", "wlan0")

    Returns:
        Camera instance (BaseCamera subclass)

    Raises:
        ValueError: If source type is unsupported
        RuntimeError: If camera initialization fails
    """
    logger.info(f"Creating camera: source={source}, interface={interface}")

    if source == "go2":
        try:
            return Go2VideoClient(interface)
        except Exception as e:
            logger.error(f"Failed to create Go2 camera: {e}")
            raise

    elif source == "usb":
        try:
            # Try camera index 0 first
            return OpenCVCamera(0)
        except Exception as e:
            logger.error(f"Failed to create USB camera: {e}")
            raise

    elif source == "file":
        # Default to a sample video file
        default_file = "sample.mp4"
        try:
            return FileCamera(default_file, loop=True)
        except Exception as e:
            logger.error(f"Failed to create file camera: {e}")
            raise

    else:
        raise ValueError(
            f"Unsupported camera source: {source}. "
            f"Supported: 'go2', 'usb', 'file'"
        )
