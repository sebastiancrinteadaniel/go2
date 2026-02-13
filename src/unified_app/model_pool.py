"""Model Pool - Singleton for shared model management"""

import asyncio
import logging
from typing import Optional, Dict
from ultralytics import YOLO
import mediapipe as mp

logger = logging.getLogger(__name__)


class ModelPool:
    """
    Singleton class to manage shared models across processors.
    Ensures models are loaded once and reused by all processors.
    """

    _instance: Optional["ModelPool"] = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._models: Dict[str, YOLO] = {}
        self._mediapipe_hands = None
        self._mediapipe_drawing = None
        self._model_locks: Dict[str, asyncio.Lock] = {}
        self._initialized = True
        logger.info("ModelPool initialized")

    async def get_yolo(self, model_path: str, device: str = "cpu") -> YOLO:
        """
        Get or load YOLO model asynchronously.
        Ensures only one model instance exists for given path.
        """
        cache_key = f"yolo_{model_path}_{device}"

        if cache_key not in self._model_locks:
            self._model_locks[cache_key] = asyncio.Lock()

        async with self._model_locks[cache_key]:
            if cache_key not in self._models:
                logger.info(f"Loading YOLO model: {model_path} on device {device}")
                # Run model loading in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                self._models[cache_key] = await loop.run_in_executor(
                    None, lambda: YOLO(model_path).to(device)
                )
                logger.info(f"YOLO model loaded successfully: {model_path}")
            return self._models[cache_key]

    async def get_mediapipe_hands(self):
        """
        Get or initialize MediaPipe Hands model.
        """
        if self._mediapipe_hands is None:
            logger.info("Initializing MediaPipe Hands")
            loop = asyncio.get_event_loop()
            self._mediapipe_hands = await loop.run_in_executor(
                None, lambda: mp.solutions.hands.Hands()
            )
            logger.info("MediaPipe Hands initialized")

        return self._mediapipe_hands

    async def get_mediapipe_drawing(self):
        """
        Get MediaPipe drawing utilities.
        """
        if self._mediapipe_drawing is None:
            loop = asyncio.get_event_loop()
            self._mediapipe_drawing = await loop.run_in_executor(
                None, lambda: mp.solutions.drawing_utils
            )
        return self._mediapipe_drawing

    def unload_all(self):
        """Unload all cached models"""
        self._models.clear()
        self._mediapipe_hands = None
        self._mediapipe_drawing = None
        logger.info("All models unloaded")


async def get_model_pool() -> ModelPool:
    """Get the global model pool instance"""
    return ModelPool()
