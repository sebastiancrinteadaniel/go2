import time
import logging

logger = logging.getLogger(__name__)


class RobotTelemetry:
    def __init__(self):
        self.battery_soc = 0
        self.motor_temps = [0.0] * 12
        self.imu_rpy = None  # [roll, pitch, yaw] in radians, or None if unavailable
        self.connected = False
        self.last_update = 0
        self.subscriber = None
        self._initialized = False

    def init(self):
        """Initialize telemetry - call when WebRTC connection is established."""
        if self._initialized:
            return

        try:
            from unitree_sdk2py.core.channel import ChannelSubscriber
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

            self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
            self.subscriber.Init(self.on_low_state, 10)
            logger.info("Robot telemetry subscriber initialized.")

            self._initialized = True
        except ImportError:
            logger.warning("'unitree_sdk2py' not found. Telemetry disabled.")
            self._initialized = True
        except Exception as e:
            logger.error(f"Error initializing telemetry subscriber: {e}")
            self._initialized = True

    def on_low_state(self, msg):
        self.battery_soc = msg.bms_state.soc
        self.motor_temps[:] = [m.temperature for m in msg.motor_state[:12]]
        self.imu_rpy = list(msg.imu_state.rpy)  # [roll, pitch, yaw] in radians

        self.connected = True
        self.last_update = time.time()


telemetry = RobotTelemetry()
