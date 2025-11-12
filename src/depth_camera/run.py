import pyrealsense2 as rs
import numpy as np
import cv2

# --- CONFIGURARE PIPELINE ---
pipeline = rs.pipeline()
config = rs.config()

# Detectăm câte camere sunt disponibile
ctx = rs.context()
devices = ctx.query_devices()
if len(devices) == 0:
    print("Nu a fost detectată nicio cameră RealSense!")
    exit(1)

# Folosim prima cameră detectată
print(devices)
device = devices[0]
serial = device.get_info(rs.camera_info.serial_number)
print(f"Folosim camera cu serial: {serial}")

config.enable_device(serial)  # selectăm exact această cameră
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# --- START PIPELINE ---
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convertim în numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Normalizăm și aplicăm colormap pentru vizualizare
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # Afișăm imaginile alăturat
        images = np.hstack((color_image, depth_colormap))
        cv2.imshow('Color + Depth', images)

        key = cv2.waitKey(1)
        if key == 27:  # ESC pentru exit
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
