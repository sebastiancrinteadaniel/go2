"""
WebRTC configuration for Unitree Go2 camera connection.
"""

CONFIG = {
    "webrtc": {
        # Source: "WebRTC" (for remote laptop) or "DDS" (for running ON the Jetson Nano)
        "source": "DDS",

        # Connection method: "LocalSTA", "LocalAP", or "Remote"
        "method": "LocalSTA",
        
        # For LocalSTA
        "ip": "192.168.123.18",
        # 100.100.61.110
        
        # For Remote or LocalSTA (sometimes required)
        "serial_number": "",
        
        # For Remote
        "username": "",
        "password": "",
        
        # Display settings
        "width": 1280,
        "height": 720,
    }
}
