"""
Standalone Go2 camera viewer + recorder.
No web server, no WebRTC, no inference — just raw camera feed.

Usage:
    python camera_viewer.py [--interface eth0]

Controls:
    r  — start / stop recording
    q  — quit
    s  — save a single screenshot
"""

import argparse
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.video.video_client import VideoClient
except ImportError:
    print("[ERROR] unitree_sdk2py not found. Install it or activate the correct environment.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_interface() -> str:
    """Return 'eth0' when a real network card is reachable, else 'lo'."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        # If we got a non-loopback address we are on a real interface
        if not ip.startswith("127."):
            return "eth0"
    except Exception:
        pass
    return "lo"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def init_sdk(interface: str) -> VideoClient:
    print(f"[INFO] Initialising Unitree SDK on interface: {interface}")
    ChannelFactoryInitialize(0, interface)
    client = VideoClient()
    client.SetTimeout(3.0)
    client.Init()
    return client


def get_frame(client: VideoClient):
    """Return a decoded BGR frame or None on failure."""
    code, data = client.GetImageSample()
    if code != 0 or not data:
        return None
    arr = np.frombuffer(bytes(data), dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(interface: str):
    client = init_sdk(interface)

    # Give SDK a moment to settle
    time.sleep(0.5)

    cv2.namedWindow("Go2 Camera", cv2.WINDOW_NORMAL)

    writer: cv2.VideoWriter | None = None
    recording = False
    output_path = ""

    # Try to get one frame to discover resolution
    test_frame = None
    for _ in range(10):
        test_frame = get_frame(client)
        if test_frame is not None:
            break
        time.sleep(0.2)

    if test_frame is None:
        print("[ERROR] Could not get any frame from camera. Check robot connection.")
        sys.exit(1)

    h, w = test_frame.shape[:2]
    print(f"[INFO] Camera resolution: {w}x{h}")
    print("[INFO] Press  r = record,  s = screenshot,  q = quit")

    fps_target = 15          # Go2 head camera is ~15 fps
    frame_delay = 1.0 / fps_target
    last_time = time.time()
    fps_display = 0.0
    fps_counter = 0
    fps_tick = time.time()

    while True:
        frame = get_frame(client)
        if frame is None:
            time.sleep(0.05)
            continue

        # FPS counter
        fps_counter += 1
        now = time.time()
        if now - fps_tick >= 1.0:
            fps_display = fps_counter / (now - fps_tick)
            fps_counter = 0
            fps_tick = now

        # Overlay
        display = frame.copy()
        status = f"REC  {output_path}" if recording else "LIVE"
        color = (0, 0, 220) if recording else (0, 220, 0)
        cv2.putText(display, f"{status}  |  {fps_display:.1f} fps",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        cv2.imshow("Go2 Camera", display)

        if recording and writer is not None:
            writer.write(frame)

        # Throttle to ~fps_target
        elapsed = time.time() - last_time
        wait_ms = max(1, int((frame_delay - elapsed) * 1000))
        last_time = time.time()

        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('r'):
            if not recording:
                os.makedirs("recordings", exist_ok=True)
                output_path = os.path.join("recordings", f"go2_{timestamp()}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, fps_target, (w, h))
                recording = True
                print(f"[REC] Started  → {output_path}")
            else:
                recording = False
                if writer:
                    writer.release()
                    writer = None
                print(f"[REC] Saved    → {output_path}")

        elif key == ord('s'):
            os.makedirs("screenshots", exist_ok=True)
            path = os.path.join("screenshots", f"go2_{timestamp()}.jpg")
            cv2.imwrite(path, frame)
            print(f"[SNAP] Saved   → {path}")

    # Cleanup
    if recording and writer:
        writer.release()
        print(f"[REC] Saved    → {output_path}")

    cv2.destroyAllWindows()
    print("[INFO] Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unitree Go2 standalone camera viewer")
    parser.add_argument(
        "--interface", "-i",
        default=None,
        help="Network interface name (e.g. eth0, enp3s0). Auto-detected if omitted."
    )
    args = parser.parse_args()

    iface = args.interface or detect_interface()
    run(iface)
