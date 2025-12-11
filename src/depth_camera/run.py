import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO


from ..common.fps import draw_info_fps, CvFpsCalc
from ..common.visual import resize_for_display
from .config import CONFIG


# --- PARAMETRI DE CONFIGURARE ---
# Citim configurari din fisierele de config
depth_cfg = CONFIG.get("depth_camera", {})
disp_cfg = CONFIG.get("display", {})

WIDTH = depth_cfg.get("width", 640)
HEIGHT = depth_cfg.get("height", 480)
FPS = depth_cfg.get("fps", 30)
MODEL_PATH = depth_cfg.get("model_path", "src/yolo/model/yolov8n.pt")

# Setari afisare din common config
WINDOW_NAME = disp_cfg.get("window_name", "Depth Camera + YOLO")
DRAW_FPS = disp_cfg.get("draw_fps", True)
DISPLAY_WIDTH = disp_cfg.get("width", 640)
DISPLAY_HEIGHT = disp_cfg.get("height", 480)

# Definirea unei zone de interes (ROI) în pixeli (toata imaginea)
ROI_X_START = 0
ROI_Y_START = 0
ROI_WIDTH = WIDTH
ROI_HEIGHT = HEIGHT
ROI_X_END = WIDTH
ROI_Y_END = HEIGHT

# Punctele pentru calculul mărimii (colțurile ROI-ului)
TOP_LEFT = (ROI_X_START, ROI_Y_START)
TOP_RIGHT = (ROI_X_END, ROI_Y_START)
BOTTOM_LEFT = (ROI_X_START, ROI_Y_END)

def initialize_realsense():
    """Inițializează pipeline-ul, aliniază și returnează datele de calibrare."""
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()

    align_to = rs.stream.color
    align = rs.align(align_to)

    return pipeline, align, depth_scale, depth_intrinsics

def process_frames(pipeline, align, depth_scale, depth_intrinsics):
    """Procesează cadrele și calculează distanța și mărimea obiectului în ROI, afișând mai multe vizualizări."""
    
    # Incarcam modelul YOLO
    print(f"Incarcare model YOLO din: {MODEL_PATH} ...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Eroare la incarcarea modelului: {e}")
        return

    # Initializare calculator FPS
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    
    # Variabile pentru optimizare si vizualizare
    frame_count = 0
    skip_frames = 2  # Ruleaza YOLO la fiecare (skip_frames + 1) cadre
    last_results = None
    view_mode = 0  # 0: RGB, 1: Depth, 2: Side-by-Side

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # --- DETECTIE OBIECTE CU YOLO (Optimizat) ---
            if frame_count % (skip_frames + 1) == 0:
                results = model(color_image, verbose=False)
                last_results = results
            else:
                results = last_results
            
            frame_count += 1
            
            info_lines = []
            closest_obj_dist = float('inf')
            closest_obj_info = "Niciun obiect detectat"

            if results:
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        # Coordonate Bounding Box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Clasa si incredere
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = model.names[cls]

                        # --- CALCUL ADANCIME PENTRU OBIECTUL CURENT ---
                        # Extragem ROI din depth map pentru acest obiect
                        # Ne asiguram ca coordonatele sunt in limitele imaginii
                        x1_c = max(0, x1)
                        y1_c = max(0, y1)
                        x2_c = min(WIDTH, x2)
                        y2_c = min(HEIGHT, y2)

                        obj_depth_roi = depth_image[y1_c:y2_c, x1_c:x2_c]
                        
                        # Filtram valorile 0
                        valid_depths = obj_depth_roi[obj_depth_roi > 0]
                        
                        if valid_depths.size > 0:
                            depth_Z = np.median(valid_depths) * depth_scale
                        else:
                            depth_Z = 0

                        # --- CALCUL DIMENSIUNI REALE ---
                        if depth_Z > 0:
                            # Punctele 3D pentru colturile bounding box-ului
                            # Folosim centrul bounding box-ului pentru Z, dar colturile 2D pentru X,Y
                            point3D_TL = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x1, y1], depth_Z)
                            point3D_BR = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x2, y2], depth_Z)
                            
                            width_m = abs(point3D_BR[0] - point3D_TL[0])
                            height_m = abs(point3D_BR[1] - point3D_TL[1])
                            
                            dim_text = f"{width_m:.2f}m x {height_m:.2f}m"
                        else:
                            dim_text = "N/A"

                        # --- DESENARE PE IMAGINE ---
                        # Dreptunghi
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Text deasupra cutiei
                        text_label = f"{label} {conf:.2f} | Z: {depth_Z:.2f}m | {dim_text}"
                        cv2.putText(color_image, text_label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Actualizam informatiile pentru bara de sus (cel mai apropiat obiect)
                        if depth_Z > 0 and depth_Z < closest_obj_dist:
                            closest_obj_dist = depth_Z
                            closest_obj_info = f"Cel mai apropiat: {label} la {depth_Z:.2f}m ({dim_text})"

            # Afisam valorile si in consola daca am gasit ceva
            if closest_obj_dist != float('inf'):
                print(closest_obj_info)
                info_lines = [closest_obj_info]
            else:
                info_lines = ["Cautare obiecte..."]

            # Calcul FPS
            if DRAW_FPS:
                fps = cvFpsCalc.get()
                info_lines.append(f"FPS: {fps:.1f}")

            # --- VIZUALIZARE GENERALĂ ---
            
            # Pregatire Depth Map pentru vizualizare
            # Scalam valorile pentru a fi vizibile (alpha=0.03 este aproximativ 255/8000mm)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            # Desenam info pe RGB
            color_image = draw_info_fps(color_image, info_lines)
            
            # Selectie imagine de afisat
            final_image = None
            
            if view_mode == 0: # RGB
                final_image = color_image
            elif view_mode == 1: # Depth
                # Adaugam info si pe depth map
                depth_colormap = draw_info_fps(depth_colormap, info_lines + ["Mod: Depth Map"])
                final_image = depth_colormap
            elif view_mode == 2: # Side-by-Side
                # Redimensionam ambele la jumatate din latime pentru a incapea
                h, w = color_image.shape[:2]
                # Asiguram ca au aceeasi dimensiune
                depth_colormap_resized = cv2.resize(depth_colormap, (w, h))
                
                # Concatenam orizontal si apoi redimensionam la dimensiunea de afisare
                combined = np.hstack((color_image, depth_colormap_resized))
                final_image = cv2.resize(combined, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

            # Redimensionam pentru afisare daca este necesar (pentru modurile single)
            if view_mode != 2:
                final_image = resize_for_display(final_image, DISPLAY_WIDTH, DISPLAY_HEIGHT)

            cv2.imshow(WINDOW_NAME, final_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'): # Toggle view mode
                view_mode = (view_mode + 1) % 3
                print(f"Schimbare mod vizualizare: {view_mode}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    pipeline, align, depth_scale, depth_intrinsics = initialize_realsense()
    process_frames(pipeline, align, depth_scale, depth_intrinsics)