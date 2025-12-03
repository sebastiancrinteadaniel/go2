"""
Laptop Client Configuration

This module runs on your LAPTOP (not the robot) and connects to the Go2 via WiFi.
"""

CONFIG = {
    "webrtc": {
        # Robot's IP address on the WiFi network
        # For LocalSTA mode (laptop on same WiFi as robot): usually 192.168.123.18
        # For LocalAP mode (laptop on robot's AP): usually 192.168.12.1
        "robot_ip": "192.168.123.18",
        
        # Connection method: "LocalSTA" or "LocalAP"
        # LocalSTA: Both laptop and robot on same WiFi network
        # LocalAP: Laptop connected to robot's own WiFi hotspot
        "method": "LocalSTA",
        
        # Optional serial number (leave empty if not needed)
        "serial_number": "",
    },
    
    "display": {
        "width": 640,
        "height": 480,
    },
    
    "server": {
        "host": "0.0.0.0",
        "port": 8000,
    },
    
    # Local CV modules that run on the laptop
    # These use the WebRTC video stream as input
    "modules": {
        "yolo": {
            "enabled": True,
            "port": 8081,
            "name": "YOLO Object Detection",
            "description": "YOLOv8 running locally on your laptop GPU.",
        },
        "hand": {
            "enabled": True,
            "port": 8082,
            "name": "Hand Detection",
            "description": "MediaPipe hand tracking running locally.",
        },
        "simple": {
            "enabled": True,
            "port": 8084,
            "name": "Raw Camera",
            "description": "Direct WebRTC camera feed from the robot.",
        },
    },
}
