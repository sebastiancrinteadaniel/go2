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
from datetime import datetime, timedelta

import cv2

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
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(15)

    cam_left = pipeline.create(dai.node.MonoCamera)
    cam_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    cam_right = pipeline.create(dai.node.MonoCamera)
    cam_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(False)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # warp depth to RGB perspective
    cam_left.out.link(stereo.left)
    cam_right.out.link(stereo.right)

    sync = pipeline.create(dai.node.Sync)
    sync.setSyncThreshold(timedelta(milliseconds=33))  # half a frame at 15 fps
    cam_rgb.video.link(sync.inputs["rgb"])
    stereo.depth.link(sync.inputs["depth"])

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("synced")
    sync.out.link(xout.input)

    return pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run():
    pipeline = build_pipeline()

    cv2.namedWindow("OAK-D Depth Recorder", cv2.WINDOW_NORMAL)

    with dai.Device(pipeline) as device:
        print(f"[INFO] OAK-D connected: {device.getDeviceName()}  "
              f"USB: {device.getUsbSpeed().name}")

        synced_q = device.getOutputQueue("synced", maxSize=4, blocking=False)

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
        fps_target = 15
        frame_delay = 1.0 / fps_target
        last_time = time.time()

        print("[INFO] Press  r = record,  s = snapshot,  q = quit")

        while True:
            synced_msg = synced_q.tryGet()

            if synced_msg is None:
                time.sleep(0.002)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue

            # Camera is physically mounted upside-down
            rgb_frame = cv2.flip(synced_msg["rgb"].getCvFrame(), -1)
            latest_depth = cv2.flip(synced_msg["depth"].getFrame(), -1)

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
            display = rgb_frame.copy()

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
    argparse.ArgumentParser(description="OAK-D S2 RGB+depth recorder").parse_args()
    run()
