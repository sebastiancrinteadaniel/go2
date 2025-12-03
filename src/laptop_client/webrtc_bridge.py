"""
WebRTC Bridge for Laptop Client

Handles the WebRTC connection to the Unitree Go2 and provides:
- Video frame queue (for local CV processing)
- Command sending via data channel
- Telemetry receiving (battery, etc.)
"""

import asyncio
import logging
import threading
import time
import json
from queue import Queue, Empty
from dataclasses import dataclass, field
from typing import Optional, Callable

import cv2
import numpy as np

from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)
from aiortc import MediaStreamTrack

from .config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RobotState:
    """Holds the current state received from the robot."""
    battery_soc: int = 0
    motor_temps: list = field(default_factory=list)
    connected: bool = False
    last_update: float = 0.0


class WebRTCBridge:
    """
    Manages WebRTC connection to the Go2.
    
    Usage:
        bridge = WebRTCBridge(robot_ip="192.168.123.18")
        bridge.start()
        
        # Get frames
        frame = bridge.get_frame()
        
        # Send commands
        bridge.send_command({"cmd": "move", "vx": 0.5, "vy": 0, "vyaw": 0})
        
        bridge.stop()
    """
    
    def __init__(self, robot_ip: str = None, method: str = None):
        cfg = CONFIG["webrtc"]
        self.robot_ip = robot_ip or cfg["robot_ip"]
        self.method = method or cfg["method"]
        self.serial = cfg.get("serial_number", "")
        
        disp = CONFIG["display"]
        self.width = disp["width"]
        self.height = disp["height"]
        
        self.frame_queue: Queue = Queue(maxsize=5)
        self.state = RobotState()
        
        self._conn: Optional[UnitreeWebRTCConnection] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        self._on_frame_callbacks: list[Callable[[np.ndarray], None]] = []
        
    def _create_connection(self) -> UnitreeWebRTCConnection:
        """Create the WebRTC connection object based on config."""
        if self.method == "LocalSTA":
            method = WebRTCConnectionMethod.LocalSTA
            if self.serial:
                return UnitreeWebRTCConnection(method, ip=self.robot_ip, serialNumber=self.serial)
            return UnitreeWebRTCConnection(method, ip=self.robot_ip)
        elif self.method == "LocalAP":
            method = WebRTCConnectionMethod.LocalAP
            return UnitreeWebRTCConnection(method)
        else:
            raise ValueError(f"Unknown connection method: {self.method}")
    
    def add_frame_callback(self, callback: Callable[[np.ndarray], None]):
        """Add a callback that receives every frame."""
        self._on_frame_callbacks.append(callback)
    
    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get the latest frame from the queue."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def get_frame_nowait(self) -> Optional[np.ndarray]:
        """Get frame without waiting, returns None if no frame available."""
        try:
            return self.frame_queue.get_nowait()
        except Empty:
            return None
    
    def send_command(self, cmd: dict):
        """
        Send a command to the robot via WebRTC data channel.
        
        The Go2 accepts commands in this format:
        - Movement: {"cmd": "move", "vx": 0.5, "vy": 0, "vyaw": 0}
        - Actions: {"cmd": "action", "action": "stand_up"}
        """
        if self._conn and self._conn.data_channel:
            try:
                # The Unitree Go2 WebRTC accepts specific JSON formats
                # We'll wrap our commands appropriately
                self._conn.data_channel.send(json.dumps(cmd))
            except Exception as e:
                logger.error(f"Failed to send command: {e}")
    
    def send_move(self, vx: float, vy: float, vyaw: float):
        """Convenience method to send movement command."""
        # Format expected by Go2 WebRTC data channel
        cmd = {
            "type": "msg",
            "topic": "rt/api/sport/request",
            "data": {
                "api_id": 1008,  # Move API
                "parameter": json.dumps({"x": vx, "y": vy, "z": vyaw})
            }
        }
        self.send_command(cmd)
    
    def send_action(self, action_name: str):
        """
        Send an action command (stand_up, sit_down, etc.)
        
        Supported actions (api_id):
        - 1001: Damp
        - 1002: BalanceStand
        - 1003: StopMove
        - 1004: StandUp
        - 1005: StandDown
        - 1006: RecoveryStand
        - 1009: Euler (body pose)
        - 1011: Hello
        - 1014: Stretch
        - 1015: Dance1
        - 1016: Dance2
        - 1036: Heart
        """
        action_map = {
            "damp": 1001,
            "balance_stand": 1002,
            "stop": 1003,
            "stand_up": 1004,
            "stand_down": 1005,
            "recovery": 1006,
            "hello": 1011,
            "stretch": 1014,
            "dance1": 1015,
            "dance2": 1016,
            "heart": 1036,
        }
        
        api_id = action_map.get(action_name)
        if api_id:
            cmd = {
                "type": "msg",
                "topic": "rt/api/sport/request",
                "data": {
                    "api_id": api_id,
                    "parameter": ""
                }
            }
            self.send_command(cmd)
            logger.info(f"Sent action: {action_name} (api_id={api_id})")
        else:
            logger.warning(f"Unknown action: {action_name}")
    
    async def _receive_video(self, track: MediaStreamTrack):
        """Async handler for receiving video frames."""
        logger.info("Video track callback started")
        while not self._stop_event.is_set():
            try:
                frame = await track.recv()
                img = frame.to_ndarray(format="bgr24")
                
                # Resize if needed
                if img.shape[1] != self.width or img.shape[0] != self.height:
                    img = cv2.resize(img, (self.width, self.height))
                
                # Update state
                self.state.connected = True
                self.state.last_update = time.time()
                
                # Put in queue (drop old frames if full)
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        pass
                self.frame_queue.put(img)
                
                # Call registered callbacks
                for cb in self._on_frame_callbacks:
                    try:
                        cb(img)
                    except Exception as e:
                        logger.error(f"Frame callback error: {e}")
                        
            except Exception as e:
                if not self._stop_event.is_set():
                    logger.error(f"Video receive error: {e}")
                break
        
        logger.info("Video track callback ended")
    
    def _on_data_channel_message(self, message: str):
        """Handle incoming data channel messages (telemetry, etc.)."""
        try:
            data = json.loads(message)
            # Parse robot state updates if available
            if "topic" in data and "lowstate" in data.get("topic", "").lower():
                # Extract battery and motor temps
                if "data" in data:
                    state_data = data["data"]
                    if "bms_state" in state_data:
                        self.state.battery_soc = state_data["bms_state"].get("soc", 0)
                    if "motor_state" in state_data:
                        self.state.motor_temps = [m.get("temperature", 0) for m in state_data["motor_state"][:12]]
        except json.JSONDecodeError:
            pass
        except Exception as e:
            logger.debug(f"Data channel parse error: {e}")
    
    async def _connect(self):
        """Establish WebRTC connection."""
        try:
            self._conn = self._create_connection()
            logger.info(f"Connecting to robot at {self.robot_ip} via {self.method}...")
            
            await self._conn.connect()
            
            # Enable video
            self._conn.video.switchVideoChannel(True)
            self._conn.video.add_track_callback(self._receive_video)
            
            # Setup data channel handler if available
            if hasattr(self._conn, 'data_channel') and self._conn.data_channel:
                self._conn.data_channel.on("message", self._on_data_channel_message)
            
            logger.info("WebRTC connected successfully!")
            self.state.connected = True
            
        except Exception as e:
            logger.error(f"WebRTC connection failed: {e}")
            self.state.connected = False
            raise
    
    def _run_event_loop(self):
        """Run the asyncio event loop in a thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._connect())
            self._loop.run_forever()
        except Exception as e:
            logger.error(f"Event loop error: {e}")
        finally:
            self._loop.close()
    
    def start(self):
        """Start the WebRTC connection in a background thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("Bridge already running")
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()
        
        # Wait for connection
        timeout = 10.0
        start = time.time()
        while not self.state.connected and (time.time() - start) < timeout:
            time.sleep(0.1)
        
        if not self.state.connected:
            logger.warning("Connection timeout - bridge may not be fully connected")
    
    def stop(self):
        """Stop the WebRTC connection."""
        self._stop_event.set()
        self.state.connected = False
        
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        
        if self._thread:
            self._thread.join(timeout=2.0)
        
        logger.info("WebRTC bridge stopped")
    
    @property
    def is_connected(self) -> bool:
        """Check if connected and receiving frames."""
        return self.state.connected and (time.time() - self.state.last_update) < 2.0


# Singleton instance for easy access
_bridge_instance: Optional[WebRTCBridge] = None

def get_bridge(robot_ip: str = None) -> WebRTCBridge:
    """Get or create the global WebRTC bridge instance."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = WebRTCBridge(robot_ip=robot_ip)
    return _bridge_instance
