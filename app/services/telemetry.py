import time
import logging
import math

logger = logging.getLogger(__name__)


class RobotTelemetry:
    def __init__(self):
        self.battery_soc = 0
        self.motor_temps = []
        self.power_v = 0.0
        self.power_a = 0.0
        self.battery_cycle = 0
        self.cell_min_v = None
        self.cell_max_v = None
        self.imu_rpy = [0.0, 0.0, 0.0]
        self.imu_gyro = [0.0, 0.0, 0.0]
        self.imu_accel = [0.0, 0.0, 0.0]
        self.imu_temp_c = None
        self.joint_hottest_index = None
        self.joint_hottest_temp_c = None
        self.joint_lost_count = 0
        self.joint_avg_tau = None
        self.foot_force = [0, 0, 0, 0]
        self.foot_force_est = [0, 0, 0, 0]
        self.temp_ntc1_c = None
        self.temp_ntc2_c = None
        self.velocity_xyz = [0.0, 0.0, 0.0]
        self.travel_speed_mps = None
        self.sport_subscriber = None
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
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_

            self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
            self.subscriber.Init(self.on_low_state, 10)
            logger.info("Robot telemetry subscriber initialized.")

            # Sport mode state provides body velocity, used for travel speed in UI.
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
        motors = list(msg.motor_state[:12])
        self.motor_temps = [float(m.temperature) for m in motors]
        self.power_v = float(msg.power_v)
        self.power_a = float(msg.power_a)
        self.battery_cycle = int(msg.bms_state.cycle)

        cell_volts = [int(v) for v in msg.bms_state.cell_vol if int(v) > 0]
        if cell_volts:
            # SDK reports cell voltage as mV in a uint16 array.
            self.cell_min_v = min(cell_volts) / 1000.0
            self.cell_max_v = max(cell_volts) / 1000.0
        else:
            self.cell_min_v = None
            self.cell_max_v = None

        self.imu_rpy = [float(v) for v in msg.imu_state.rpy]
        self.imu_gyro = [float(v) for v in msg.imu_state.gyroscope]
        self.imu_accel = [float(v) for v in msg.imu_state.accelerometer]
        self.imu_temp_c = int(msg.imu_state.temperature)

        if self.motor_temps:
            hottest_index, hottest_temp = max(enumerate(self.motor_temps), key=lambda pair: pair[1])
            self.joint_hottest_index = int(hottest_index)
            self.joint_hottest_temp_c = float(hottest_temp)
        else:
            self.joint_hottest_index = None
            self.joint_hottest_temp_c = None

        self.joint_lost_count = sum(1 for m in motors if int(m.lost) != 0)
        if motors:
            self.joint_avg_tau = sum(abs(float(m.tau_est)) for m in motors) / len(motors)
        else:
            self.joint_avg_tau = None

        self.foot_force = [int(v) for v in msg.foot_force]
        self.foot_force_est = [int(v) for v in msg.foot_force_est]
        self.temp_ntc1_c = int(msg.temperature_ntc1)
        self.temp_ntc2_c = int(msg.temperature_ntc2)

        self.connected = True
        self.last_update = time.time()

    def on_sport_state(self, msg):
        self.velocity_xyz = [float(v) for v in msg.velocity]
        vx = self.velocity_xyz[0]
        vy = self.velocity_xyz[1]
        self.travel_speed_mps = math.sqrt((vx * vx) + (vy * vy))


telemetry = RobotTelemetry()
