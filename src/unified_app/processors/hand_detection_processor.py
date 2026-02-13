"""Hand Detection Processor"""

import asyncio
import logging
import numpy as np
import cv2
from typing import Dict, Any
from src.unified_app.processor import BaseProcessor
from src.unified_app.config import HandDetectionConfig

logger = logging.getLogger(__name__)


class HandDetectionProcessor(BaseProcessor):
    """
    MediaPipe hand detection processor.
    Detects hand poses and draws keypoints/connections on frames.
    """

    def __init__(self, config: HandDetectionConfig):
        super().__init__("hand_detection", config)
        self.hands_model = None
        self.drawing_utils = None
        self.mp_hands = None
        self.max_hands = config.max_hands
        self.confidence = config.confidence

    async def start(self, camera, input_queue, output_queue, model_pool):
        """Start processor and initialize MediaPipe"""
        await super().start(camera, input_queue, output_queue, model_pool)

        # Initialize MediaPipe
        try:
            import mediapipe as mp

            self.mp_hands = mp
            self.hands_model = await self.model_pool.get_mediapipe_hands()
            self.drawing_utils = await self.model_pool.get_mediapipe_drawing()
            self.logger.info("MediaPipe Hands initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize MediaPipe: {e}")

    async def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect hands in frame using MediaPipe and draw keypoints.

        Args:
            frame: Input frame in BGR format

        Returns:
            Frame with hand keypoints overlaid
        """
        if self.hands_model is None or self.mp_hands is None:
            return frame

        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run hand detection in thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.hands_model.process(rgb_frame),
            )

            # Draw detections
            annotated_frame = frame.copy()

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    # Draw hand landmarks
                    self.drawing_utils.draw_landmarks(
                        annotated_frame,
                        hand_landmarks,
                        self.mp_hands.solutions.hands.HAND_CONNECTIONS,
                        self.mp_hands.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_hands.solutions.drawing_styles.get_default_hand_connections_style(),
                    )

                    # Add hand label (Left/Right)
                    label_text = f"{handedness.classification[0].label}"
                    cv2.putText(
                        annotated_frame,
                        label_text,
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

            return annotated_frame

        except Exception as e:
            self.logger.error(f"Error in hand detection: {e}", exc_info=True)
            return frame
