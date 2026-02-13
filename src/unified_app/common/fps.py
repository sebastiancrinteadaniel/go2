"""FPS Calculation - High-precision FPS tracking using OpenCV timer"""

from collections import deque
import cv2
import numpy as np


class FPSCalculator:
    """
    High-precision FPS calculator using OpenCV's timer.
    More accurate than time.time() because it uses CPU ticks.
    """

    def __init__(self, buffer_len: int = 1):
        """
        Initialize FPS calculator.

        Args:
            buffer_len: Number of frames to average over (higher = smoother but slower to update)
        """
        self._start_tick = cv2.getTickCount()
        self._freq = 1000.0 / cv2.getTickFrequency()  # Convert to milliseconds
        self._difftimes = deque(maxlen=buffer_len)

    def get(self) -> float:
        """
        Calculate and return current FPS.

        Returns:
            FPS value (rounded to 2 decimal places)
        """
        current_tick = cv2.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._difftimes.append(different_time)

        # Calculate average FPS from buffered frame times
        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        return round(fps, 2)


def draw_fps_on_frame(
    image: np.ndarray,
    fps: float,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.7,
    thickness: int = 2,
    text_color: tuple = (0, 255, 0),
    bg_color: tuple = (0, 0, 0),
    margin: int = 8,
) -> np.ndarray:
    """
    Draw FPS text on frame with background.

    Args:
        image: Frame to draw on (BGR)
        fps: FPS value to display
        font: OpenCV font
        font_scale: Text scale
        thickness: Text thickness
        text_color: RGB color for text
        bg_color: RGB color for background
        margin: Padding around text

    Returns:
        Modified frame with FPS text
    """
    if image is None:
        return image

    try:
        h, w = image.shape[:2]
    except Exception:
        return image

    text = f"FPS: {fps:.1f}"

    # Get text size
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate background rectangle
    x = margin
    y = margin + th
    x2 = x + tw + margin
    y2 = y + baseline + margin

    # Draw background rectangle
    cv2.rectangle(image, (x, y - th - margin), (x2, y2), bg_color, -1)

    # Draw FPS text
    cv2.putText(
        image,
        text,
        (x + margin // 2, y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )

    return image
