import time
import logging
import math

logger = logging.getLogger(__name__)


class RobotTelemetry:
    def __init__(self):
        self.battery_soc = 0
        self.motor_temps = []
        self.travel_speed_mps = None
        self.connected = False
        self.last_update = 0
        self.subscriber = None
        self.sport_subscriber = None
        self._initialized = False

    def init(self):
        """Initialize telemetry - call when WebRTC connection is established."""
        if self._initialized:
            return

        try:
            from unitree_sdk2py.core.channel import ChannelSubscriber
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_

            self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
            self.subscriber.Init(self.on_low_state, 10)
            logger.info("Robot telemetry subscriber initialized.")

            try:
                self.sport_subscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
                self.sport_subscriber.Init(self.on_sport_state, 10)
                logger.info("Sport mode telemetry subscriber initialized.")
            except Exception as e:
                logger.warning(f"Sport mode telemetry unavailable: {e}")

            self._initialized = True
        except ImportError:
            logger.warning("'unitree_sdk2py' not found. Telemetry disabled.")
            self._initialized = True
        except Exception as e:
            logger.error(f"Error initializing telemetry subscriber: {e}")
            self._initialized = True

    def on_low_state(self, msg):
        self.battery_soc = msg.bms_state.soc
        self.motor_temps = [m.temperature for m in msg.motor_state[:12]]

        self.connected = True
        self.last_update = time.time()

    def on_sport_state(self, msg):
        vx, vy, _ = [float(v) for v in msg.velocity]
        self.travel_speed_mps = math.sqrt((vx * vx) + (vy * vy))


telemetry = RobotTelemetry()
