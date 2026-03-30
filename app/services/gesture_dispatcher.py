import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from unitree_sdk2py.go2.sport.sport_client import SportClient  # type: ignore[reportMissingImports]

    _SDK_AVAILABLE = True
except ImportError:
    SportClient = None  # type: ignore[assignment]
    _SDK_AVAILABLE = False


class GestureDispatcher:
    def __init__(
        self,
        enabled: bool = True,
        cooldown_seconds: float = 2.0,
        global_cooldown_seconds: Optional[float] = None,
        min_confidence: float = 0.75,
        min_stable_frames: int = 3,
    ):
        self.enabled = enabled
        self._cooldown_seconds = max(0.0, float(cooldown_seconds))
        # Use the same cooldown by default, but allow overriding global action throttling.
        if global_cooldown_seconds is None:
            self._global_cooldown_seconds = self._cooldown_seconds
        else:
            self._global_cooldown_seconds = max(0.0, float(global_cooldown_seconds))
        self._min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self._min_stable_frames = max(1, int(min_stable_frames))
        self._last_run: Dict[str, float] = {}
        self._last_run_any = 0.0
        self._lock = threading.Lock()
        self._action_lock = threading.Lock()
        self._candidate_label = ""
        self._candidate_count = 0

        self._sport_client: Optional[Any] = None
        self._gesture_actions: Dict[str, Callable[[], None]] = {}
        self._last_dispatch: Optional[dict] = None
        self._dispatch_event_lock = threading.Lock()
        self._init_client_and_actions()

    def _init_client_and_actions(self) -> None:
        if not _SDK_AVAILABLE:
            logger.warning("Gesture dispatch disabled: unitree_sdk2py not available.")
            self.enabled = False
            return

        try:
            self._sport_client = SportClient()
            self._sport_client.SetTimeout(10.0)
            self._sport_client.Init()
            self._gesture_actions = {
                "like": self._action_like,
                "dislike": self._action_dislike,
                "peacesign": self._action_peace_sign,
                "heart": self._action_heart,
                "fingerheart": self._action_heart,
                "pinkie": self._action_pinkie,
            }
            logger.info(
                "Gesture dispatcher initialized (cooldown=%.2fs, global_cooldown=%.2fs, min_confidence=%.2f, min_stable_frames=%d).",
                self._cooldown_seconds,
                self._global_cooldown_seconds,
                self._min_confidence,
                self._min_stable_frames,
            )
        except Exception as e:
            self.enabled = False
            logger.error("Gesture dispatch disabled: failed to initialize SportClient: %s", e)

    @staticmethod
    def _extract_label(gesture_class: str) -> str:
        if not gesture_class:
            return ""
        if ":" in gesture_class:
            return gesture_class.split(":", 1)[1].strip().lower()
        return gesture_class.strip().lower()

    def process(self, gestures: List[dict]) -> None:
        if not self.enabled or self._sport_client is None:
            return

        best_label = ""
        best_conf = 0.0
        for gesture in gestures:
            label = self._extract_label(str(gesture.get("class", "")))
            if label not in self._gesture_actions:
                continue
            confidence = float(gesture.get("conf", 0.0) or 0.0)
            if confidence < self._min_confidence:
                continue
            if confidence > best_conf:
                best_label = label
                best_conf = confidence

        if not best_label:
            self._candidate_label = ""
            self._candidate_count = 0
            return

        if best_label == self._candidate_label:
            self._candidate_count += 1
        else:
            self._candidate_label = best_label
            self._candidate_count = 1

        if self._candidate_count < self._min_stable_frames:
            return

        now = time.time()
        with self._lock:
            if now - self._last_run_any < self._global_cooldown_seconds:
                return
            last = self._last_run.get(best_label, 0.0)
            if now - last < self._cooldown_seconds:
                return
            self._last_run[best_label] = now
            self._last_run_any = now

        if not self._action_lock.acquire(blocking=False):
            logger.debug("Skipping gesture %s because another action is still running.", best_label)
            return

        action = self._gesture_actions[best_label]
        threading.Thread(
            target=self._run_action,
            args=(best_label, best_conf, action),
            daemon=True,
        ).start()

    def pop_last_dispatch(self) -> Optional[dict]:
        """Return and clear the last dispatched gesture event, or None if none since last call."""
        with self._dispatch_event_lock:
            ev = self._last_dispatch
            self._last_dispatch = None
            return ev

    def _run_action(self, label: str, confidence: float, action: Callable[[], None]) -> None:
        try:
            action()
            logger.info("Gesture dispatched: %s (conf=%.2f)", label, confidence)
            with self._dispatch_event_lock:
                self._last_dispatch = {"label": label, "conf": round(confidence, 2)}
        except Exception as e:
            logger.error("Gesture action failed for %s: %s", label, e)
        finally:
            self._action_lock.release()

    def _action_like(self) -> None:
        if self._sport_client is None:
            return
        self._sport_client.StandUp()
        self._sport_client.FreeWalk()

    def _action_heart(self) -> None:
        if self._sport_client is None:
            return
        self._sport_client.Heart()

    def _action_dislike(self) -> None:
        if self._sport_client is None:
            return
        self._sport_client.StopMove()
        self._sport_client.StandDown()

    def _action_peace_sign(self) -> None:
        if self._sport_client is None:
            return
        self._sport_client.Hello()

    def _action_pinkie(self) -> None:
        if self._sport_client is None:
            return
        # self._sport_client.Stretch()
        # self._sport_client.Sit()
        # self._sport_client.RiseSit()
        # self._sport_client.Content()
        # self._sport_client.Scrape()
        # self._sport_client.Dance1()
        # self._sport_client.Dance2()
        # self._sport_client.FrontFlip()
        # self._sport_client.BackFlip()
        # self._sport_client.FrontJump()
        # self._sport_client.HandStand(True)
        # self._sport_client.RecoveryStand()
