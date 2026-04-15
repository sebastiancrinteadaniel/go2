"""
Standalone OAK-D S2 depth camera recorder.
No web server, no WebRTC, no inference — just raw RGB + depth data.

Saves:
  recordings/depth_<timestamp>/
    rgb.mp4           — colour video (mp4v)
    depth/000000.png  — 16-bit PNG per frame (values in mm, 0–65535)
    meta.txt          — fps, resolution, depth scale

Controls:
    r  — start / stop recording
    s  — save a single RGB+depth snapshot
    q  — quit
"""

import argparse
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

try:
    import depthai as dai
except ImportError:
    print("[ERROR] depthai not found. Install it: pip install depthai")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_pipeline() -> dai.Pipeline:
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(30)

    cam_left = pipeline.create(dai.node.MonoCamera)
    cam_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    cam_right = pipeline.create(dai.node.MonoCamera)
    cam_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(False)
    cam_left.out.link(stereo.left)
    cam_right.out.link(stereo.right)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")

    cam_rgb.video.link(xout_rgb.input)
    stereo.depth.link(xout_depth.input)

    return pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def depth_colormap(depth_raw: np.ndarray) -> np.ndarray:
    """Convert uint16 depth (mm) to a BGR colourmap for display."""
    clipped = np.clip(depth_raw, 0, 5000).astype(np.float32)
    norm = (clipped / 5000.0 * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)


def overlay_depth_pip(frame: np.ndarray, depth_raw: np.ndarray) -> np.ndarray:
    """Bottom-right picture-in-picture depth minimap on *frame* (in-place copy)."""
    display = frame.copy()
    h, w = display.shape[:2]
    pip_w, pip_h = w // 4, h // 4

    d_color = cv2.resize(depth_colormap(depth_raw), (pip_w, pip_h))
    cv2.rectangle(d_color, (0, 0), (pip_w - 1, pip_h - 1), (180, 180, 180), 1)
    cv2.putText(d_color, "DEPTH (0-5m)", (4, pip_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    x_off, y_off = w - pip_w - 4, h - pip_h - 4
    display[y_off:y_off + pip_h, x_off:x_off + pip_w] = d_color
    return display


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(show_depth_side_by_side: bool = False):
    pipeline = build_pipeline()

    cv2.namedWindow("OAK-D Depth Recorder", cv2.WINDOW_NORMAL)

    with dai.Device(pipeline) as device:
        print(f"[INFO] OAK-D connected: {device.getDeviceName()}  "
              f"USB: {device.getUsbSpeed().name}")

        rgb_q = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        depth_q = device.getOutputQueue("depth", maxSize=4, blocking=False)

        # ----------------------------------------------------------------
        # State
        # ----------------------------------------------------------------
        recording = False
        session_dir = ""
        depth_dir = ""
        writer: cv2.VideoWriter | None = None
        frame_idx = 0
        rgb_w = rgb_h = 0          # filled on first frame

        fps_counter = 0
        fps_display = 0.0
        fps_tick = time.time()
        fps_target = 30
        frame_delay = 1.0 / fps_target
        last_time = time.time()

        latest_depth: np.ndarray | None = None

        print("[INFO] Press  r = record,  s = snapshot,  q = quit")

        while True:
            # Poll both queues
            rgb_msg = rgb_q.tryGet()
            depth_msg = depth_q.tryGet()

            if depth_msg is not None:
                latest_depth = cv2.flip(depth_msg.getFrame(), -1)

            if rgb_msg is None or latest_depth is None:
                time.sleep(0.002)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue

            # Camera is physically mounted upside-down
            rgb_frame = cv2.flip(rgb_msg.getCvFrame(), -1)

            if rgb_w == 0:
                rgb_h, rgb_w = rgb_frame.shape[:2]
                print(f"[INFO] RGB resolution: {rgb_w}x{rgb_h}")
                print(f"[INFO] Depth resolution: {latest_depth.shape[1]}x{latest_depth.shape[0]}")

            # FPS
            fps_counter += 1
            now = time.time()
            if now - fps_tick >= 1.0:
                fps_display = fps_counter / (now - fps_tick)
                fps_counter = 0
                fps_tick = now

            # Build display frame
            if show_depth_side_by_side:
                d_resized = cv2.resize(depth_colormap(latest_depth), (rgb_w, rgb_h))
                display = np.hstack([rgb_frame, d_resized])
            else:
                display = overlay_depth_pip(rgb_frame, latest_depth)

            status = f"REC  {session_dir}" if recording else "LIVE"
            color = (0, 0, 220) if recording else (0, 220, 0)
            cv2.putText(display, f"{status}  |  {fps_display:.1f} fps",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            cv2.imshow("OAK-D Depth Recorder", display)

            # Write current frame pair
            if recording and writer is not None:
                writer.write(rgb_frame)
                depth_path = os.path.join(depth_dir, f"{frame_idx:06d}.png")
                cv2.imwrite(depth_path, latest_depth)   # uint16 PNG — lossless mm values
                frame_idx += 1

            # Throttle
            elapsed = time.time() - last_time
            wait_ms = max(1, int((frame_delay - elapsed) * 1000))
            last_time = time.time()

            key = cv2.waitKey(wait_ms) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('r'):
                if not recording:
                    # Create session folder
                    ts = timestamp()
                    session_dir = os.path.join("recordings", f"depth_{ts}")
                    depth_dir = os.path.join(session_dir, "depth")
                    os.makedirs(depth_dir, exist_ok=True)

                    rgb_path = os.path.join(session_dir, "rgb.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(rgb_path, fourcc, fps_target, (rgb_w, rgb_h))
                    frame_idx = 0

                    # Write metadata
                    with open(os.path.join(session_dir, "meta.txt"), "w") as f:
                        f.write(f"fps={fps_target}\n")
                        f.write(f"rgb_width={rgb_w}\n")
                        f.write(f"rgb_height={rgb_h}\n")
                        f.write(f"depth_width={latest_depth.shape[1]}\n")
                        f.write(f"depth_height={latest_depth.shape[0]}\n")
                        f.write("depth_unit=mm\n")
                        f.write("depth_dtype=uint16\n")
                        f.write("depth_format=16bit_png\n")

                    recording = True
                    print(f"[REC] Started  → {session_dir}/")
                else:
                    recording = False
                    if writer:
                        writer.release()
                        writer = None
                    print(f"[REC] Saved    → {session_dir}/  ({frame_idx} frames)")

            elif key == ord('s'):
                ts = timestamp()
                snap_dir = os.path.join("screenshots", f"depth_{ts}")
                os.makedirs(snap_dir, exist_ok=True)
                cv2.imwrite(os.path.join(snap_dir, "rgb.jpg"), rgb_frame)
                cv2.imwrite(os.path.join(snap_dir, "depth.png"), latest_depth)  # uint16
                print(f"[SNAP] Saved   → {snap_dir}/")

    # Cleanup
    if recording and writer:
        writer.release()
        print(f"[REC] Saved    → {session_dir}/  ({frame_idx} frames)")

    cv2.destroyAllWindows()
    print("[INFO] Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OAK-D S2 RGB+depth recorder")
    parser.add_argument(
        "--side-by-side", "-s",
        action="store_true",
        help="Show RGB and depth colormap side-by-side instead of PiP overlay",
    )
    args = parser.parse_args()

    run(show_depth_side_by_side=args.side_by_side)
