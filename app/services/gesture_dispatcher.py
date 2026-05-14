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
        self._max_linear: float = 0.7
        self._max_yaw: float = 1.0
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
            self._sport_client = None
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
        # self._sport_client.Heart()

    def _action_dislike(self) -> None:
        if self._sport_client is None:
            return
        self._sport_client.StopMove()
        self._sport_client.StandDown()

    def _action_peace_sign(self) -> None:
        if self._sport_client is None:
            return
        self._sport_client.Hello()

    _LINEAR_MIN = 0.1
    _LINEAR_MAX = 0.9
    _YAW_MIN = 0.1
    _YAW_MAX = 1.7

    def set_speed_limits(self, linear: float, yaw: float) -> None:
        self._max_linear = max(self._LINEAR_MIN, min(self._LINEAR_MAX, float(linear)))
        self._max_yaw = max(self._YAW_MIN, min(self._YAW_MAX, float(yaw)))
        logger.info("Speed limits updated: linear=%.2f yaw=%.2f", self._max_linear, self._max_yaw)

    def handle_joystick(self, lx: float, ly: float, rx: float) -> None:
        """Send a Move command from gamepad joystick values (all in [-1, 1])."""
        if self._sport_client is None:
            return
        vx = round(float(ly) * self._max_linear, 3)
        vy = round(-float(lx) * self._max_linear, 3)
        vyaw = round(-float(rx) * self._max_yaw, 3)
        threading.Thread(
            target=self._move_safe, args=(vx, vy, vyaw), daemon=True
        ).start()

    def _move_safe(self, vx: float, vy: float, vyaw: float) -> None:
        try:
            rc = self._sport_client.Move(vx, vy, vyaw)
            if rc != 0:
                logger.warning("Move returned error code %s (vx=%.3f vy=%.3f vyaw=%.3f)", rc, vx, vy, vyaw)
        except Exception as e:
            logger.error("Joystick move error: %s", e)

    def handle_action(self, cmd: str) -> None:
        """Execute a named action from the gamepad panel."""
        if self._sport_client is None:
            return
        if cmd == "stand":
            threading.Thread(target=self._action_stand, daemon=True).start()
            return
        sc = self._sport_client
        action_map: Dict[str, Callable[[], None]] = {
            # movement
            "damp":                 sc.Damp,
            "sit":                  sc.StandDown,
            "stop_move":            sc.StopMove,
            "balance_stand":        sc.BalanceStand,
            "free_walk":            sc.FreeWalk,
            "recover":              sc.RecoveryStand,
            "static_walk":          sc.StaticWalk,
            "trot_run":             sc.TrotRun,
            "switch_avoid":         sc.SwitchAvoidMode,
            # tricks
            "wave":                 sc.Hello,
            "dance":                sc.Dance1,
            "dance2":               sc.Dance2,
            "stretch":              sc.Stretch,
            "content":              sc.Content,
            "scrape":               sc.Scrape,
            "heart":                sc.Heart,
            "sit_pose":             sc.Sit,
            "rise_sit":             sc.RiseSit,
            # acrobatics
            "front_flip":           sc.FrontFlip,
            "front_jump":           sc.FrontJump,
            "front_pounce":         sc.FrontPounce,
            "left_flip":            sc.LeftFlip,
            "back_flip":            sc.BackFlip,
            # bool modes — on/off pairs
            "handstand_on":         lambda: sc.HandStand(True),
            "handstand_off":        lambda: sc.HandStand(False),
            "classic_walk_on":      lambda: sc.ClassicWalk(True),
            "classic_walk_off":     lambda: sc.ClassicWalk(False),
            "walk_upright_on":      lambda: sc.WalkUpright(True),
            "walk_upright_off":     lambda: sc.WalkUpright(False),
            "free_bound_on":        lambda: sc.FreeBound(True),
            "free_bound_off":       lambda: sc.FreeBound(False),
            "free_jump_on":         lambda: sc.FreeJump(True),
            "free_jump_off":        lambda: sc.FreeJump(False),
            "free_avoid_on":        lambda: sc.FreeAvoid(True),
            "free_avoid_off":       lambda: sc.FreeAvoid(False),
            "cross_step_on":        lambda: sc.CrossStep(True),
            "cross_step_off":       lambda: sc.CrossStep(False),
            "auto_recovery_on":     lambda: sc.AutoRecoverySet(True),
            "auto_recovery_off":    lambda: sc.AutoRecoverySet(False),
            # speed levels
            "speed_1":              lambda: sc.SpeedLevel(1),
            "speed_2":              lambda: sc.SpeedLevel(2),
            "speed_3":              lambda: sc.SpeedLevel(3),
        }
        action = action_map.get(cmd)
        if action:
            threading.Thread(
                target=self._call_safe, args=(cmd, action), daemon=True
            ).start()

    def _action_stand(self) -> None:
        try:
            self._sport_client.StandUp()
            logger.info("Gamepad action: stand")
            time.sleep(1.5)
            self._sport_client.FreeWalk()
            logger.info("Gamepad action: free_walk (auto after stand)")
        except Exception as e:
            logger.error("Stand action failed: %s", e)

    def _call_safe(self, label: str, fn: Callable[[], None]) -> None:
        try:
            fn()
            logger.info("Gamepad action: %s", label)
        except Exception as e:
            logger.error("Gamepad action %s failed: %s", label, e)

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
