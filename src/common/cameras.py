from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Optional

import cv2 as cv
import numpy as np


@dataclass
class Frame:
    ok: bool
    image: Optional[np.ndarray]


class BaseCamera:
    def read(self) -> Frame:
        raise NotImplementedError

    def release(self) -> None:
        pass


class OpenCVCamera(BaseCamera):
    def __init__(self, device: int, width: int, height: int):
        self.cap = cv.VideoCapture(device)
        if width:
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> Frame:
        ok, image = self.cap.read()
        return Frame(ok=ok, image=image if ok else None)

    def release(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass


class VideoFileCamera(BaseCamera):
    def __init__(self, path: str, width: int, height: int):
        self.cap = cv.VideoCapture(path)
        if width:
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> Frame:
        ok, image = self.cap.read()
        return Frame(ok=ok, image=image if ok else None)

    def release(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass


class Go2Camera(BaseCamera):
    """Unitree GO2 camera provider using VideoClient.

    Returns BGR frames shaped (H, W, 3) or None when unavailable.
    """

    def __init__(self, width: int, height: int, timeout_sec: float = 3.0, init_channel: bool = True):
        # Lazy imports to avoid requiring sdk when unused
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.go2.video.video_client import VideoClient

        if init_channel:
            try:
                if len(sys.argv) > 1:
                    ChannelFactoryInitialize(0, sys.argv[1])
                else:
                    ChannelFactoryInitialize(0)
            except Exception:
                # Continue even if already initialized
                pass

        self.width = width
        self.height = height
        self.video_client = VideoClient()
        self.video_client.SetTimeout(timeout_sec)
        self.video_client.Init()

    def read(self) -> Frame:
        code, data = self.video_client.GetImageSample()
        if code != 0 or not data:
            return Frame(ok=False, image=None)

        try:
            if isinstance(data, (bytes, bytearray, memoryview)):
                image_data = np.frombuffer(data, dtype=np.uint8)
            else:
                image_data = np.frombuffer(bytes(data), dtype=np.uint8)
            image = cv.imdecode(image_data, cv.IMREAD_COLOR)
        except Exception:
            image = None

        if image is None:
            return Frame(ok=False, image=None)

        # Resize to desired display size if provided
        h, w = image.shape[:2]
        if self.width and self.height and (w != self.width or h != self.height):
            image = cv.resize(image, (self.width, self.height), interpolation=cv.INTER_AREA)

        return Frame(ok=True, image=image)

    def release(self) -> None:
        # No explicit release API on VideoClient; rely on GC
        pass


def create_camera(
    source: str,
    width: int,
    height: int,
    *,
    device: int = 0,
    video_path: str = "",
    go2_timeout: float = 3.0,
    go2_init_channel: bool = True,
) -> BaseCamera:
    source_l = (source or "").lower()
    if source_l == "opencv":
        return OpenCVCamera(device=device, width=width, height=height)
    if source_l == "video":
        return VideoFileCamera(path=video_path, width=width, height=height)
    if source_l == "go2":
        return Go2Camera(width=width, height=height, timeout_sec=go2_timeout, init_channel=go2_init_channel)
    raise ValueError(f"Unknown camera source: {source}")
