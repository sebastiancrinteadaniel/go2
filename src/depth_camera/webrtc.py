import logging

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

from ..common.web_server import WebStreamer
from .config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_realsense(width, height, fps):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    
    profile = pipeline.start(config)
    
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    return pipeline, align, depth_scale, depth_intrinsics

def main():
    # Initialize Web Streamer
    streamer = WebStreamer(port=8083) # Use 8083 for Depth Camera
    streamer.start()

    # Config
    depth_cfg = CONFIG.get("depth_camera", {})
    width = depth_cfg.get("width", 640)
    height = depth_cfg.get("height", 480)
    fps = depth_cfg.get("fps", 30)
    model_path = depth_cfg.get("model_path", "src/yolo/model/yolov8n.pt")

    # Initialize RealSense
    try:
        pipeline, align, depth_scale, depth_intrinsics = initialize_realsense(width, height, fps)
    except Exception as e:
        logger.error(f"Failed to initialize RealSense: {e}")
        return

    # Load YOLO
    logger.info(f"Loading YOLO model: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    logger.info("Depth Camera WebRTC started.")

    fps_val = 0.0
    prev_time = cv2.getTickCount()
    
    # Optimization variables
    frame_count = 0
    skip_frames = 2 # Run YOLO every 3 frames
    last_results = None

    try:
        while True:
            frames = pipeline.wait_for_frames()
            
            curr_time = cv2.getTickCount()
            time_diff = (curr_time - prev_time) / cv2.getTickFrequency()
            prev_time = curr_time
            fps_val = 1.0 / max(1e-6, time_diff)

            aligned_frames = align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
                
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # YOLO Inference (Optimized)
            if frame_count % (skip_frames + 1) == 0:
                results = model(color_image, verbose=False)
                last_results = results
            else:
                results = last_results
            
            frame_count += 1
            
            # Draw results
            if results:
                annotated_frame = results[0].plot()
                
                # Add depth info to annotated frame
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        
                        if 0 <= cx < width and 0 <= cy < height:
                            dist = depth_frame.get_distance(cx, cy)
                            label = f"{dist:.2f}m"
                            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                annotated_frame = color_image.copy()
            
            # Create Depth Map Visualization
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            # Combine Side-by-Side (RGB + Depth)
            # Resize to ensure they match (they should, but good practice)
            if annotated_frame.shape[:2] != depth_colormap.shape[:2]:
                depth_colormap = cv2.resize(depth_colormap, (annotated_frame.shape[1], annotated_frame.shape[0]))
                
            combined_frame = np.hstack((annotated_frame, depth_colormap))
            
            # Resize combined frame to reasonable streaming size if needed
            # For now, we stream the full double-width image (1280x480)
            
            # Update Stats
            h, w = combined_frame.shape[:2]
            num_objects = len(results[0].boxes) if results else 0
            streamer.update_stats({
                "fps": fps_val,
                "width": w,
                "height": h,
                "info": f"Objects: {num_objects} | Mode: RGB+Depth"
            })

            # Stream
            streamer.put_frame(combined_frame)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Runtime error: {e}")
    finally:
        streamer.stop()
        pipeline.stop()

if __name__ == "__main__":
    main()
