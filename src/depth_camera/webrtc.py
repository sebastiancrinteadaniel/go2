import logging
import threading
import queue
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

from ..common.web_server import WebStreamer
from ..common.fps import CvFpsCalc
from .config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def frame_reader_thread(pipeline, align, frame_queue, stop_event):
    """Reads frames from RealSense in a separate thread."""
    while not stop_event.is_set():
        try:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            aligned_frames = align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
                
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            frame_queue.put((depth_image, color_image))
            
        except RuntimeError:
            continue
        except Exception as e:
            logger.error(f"Frame reader error: {e}")
            break

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

    # Use the common FPS calculator for stable readings (moving average)
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    
    # Optimization variables
    frame_count = 0
    skip_frames = 2 # Run YOLO every 3 frames
    last_results = None

    # Start frame reader thread
    frame_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    reader_thread = threading.Thread(target=frame_reader_thread, 
                                   args=(pipeline, align, frame_queue, stop_event))
    reader_thread.daemon = True
    reader_thread.start()

    try:
        while True:
            try:
                depth_image, color_image = frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            # Calculate FPS using the smoothed calculator
            fps_val = cvFpsCalc.get()

            # YOLO Inference (Optimized)
            if frame_count % (skip_frames + 1) == 0:
                results = model(color_image, verbose=False)
                last_results = results
            else:
                results = last_results
            
            frame_count += 1
            
            # Draw results
            if results:
                # Use plot() but we might want to customize it to match run.py style better
                # For now, plot() is fine, but let's ensure we add the depth info cleanly
                annotated_frame = results[0].plot()
                
                # Add depth info to annotated frame
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        
                        if 0 <= cx < width and 0 <= cy < height:
                            # OPTIMIZATION: Use center crop for median calculation
                            # Full ROI median is too slow for large objects (blocking)
                            # Single pixel is too noisy
                            
                            # Ensure coordinates are within bounds
                            x1_c, y1_c = max(0, x1), max(0, y1)
                            x2_c, y2_c = min(width, x2), min(height, y2)
                            
                            roi_h = y2_c - y1_c
                            roi_w = x2_c - x1_c
                            
                            # Crop size: max 40x40, min 4x4, or 20% of ROI
                            crop_h = min(40, max(4, int(roi_h * 0.2)))
                            crop_w = min(40, max(4, int(roi_w * 0.2)))
                            
                            cx_roi = (x1_c + x2_c) // 2
                            cy_roi = (y1_c + y2_c) // 2
                            
                            start_y = max(y1_c, cy_roi - crop_h // 2)
                            end_y = min(y2_c, start_y + crop_h)
                            start_x = max(x1_c, cx_roi - crop_w // 2)
                            end_x = min(x2_c, start_x + crop_w)
                            
                            obj_depth_roi = depth_image[start_y:end_y, start_x:end_x]
                            valid_depths = obj_depth_roi[obj_depth_roi > 0]
                            
                            if valid_depths.size > 0:
                                dist = np.median(valid_depths) * depth_scale
                            else:
                                dist = 0.0
                            
                            label = f"{dist:.2f}m"
                            # Draw slightly larger text with background for visibility
                            (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + w_text, y1 - 5), (0, 0, 0), -1)
                            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
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
                "info": f"Objects: {num_objects}"
            })

            # Stream
            streamer.put_frame(combined_frame)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Runtime error: {e}")
    finally:
        stop_event.set()
        streamer.stop()
        pipeline.stop()

if __name__ == "__main__":
    main()
