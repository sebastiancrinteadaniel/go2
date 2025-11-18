import cv2
import threading
import queue
import time
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

frame_queue = queue.Queue(maxsize=3)
result_queue = queue.Queue(maxsize=3)

running = True

camera_fps = 0
yolo_fps = 0


def camera_thread():
    global running, camera_fps
    cap = cv2.VideoCapture(0)

    prev_time = time.time()

    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        # Calcul FPS cameră
        now = time.time()
        camera_fps = 1 / (now - prev_time)
        prev_time = now

        if not frame_queue.full():
            frame_queue.put(frame)

    cap.release()


def yolo_thread():
    global running, yolo_fps
    prev_time = time.time()

    while running:
        if not frame_queue.empty():
            frame = frame_queue.get()

            results = model(frame, verbose=False)

            # Calcul FPS YOLO
            now = time.time()
            yolo_fps = 1 / (now - prev_time)
            prev_time = now

            result_queue.put((frame, results))


# Pornește thread-urile
t_cam = threading.Thread(target=camera_thread)
t_yolo = threading.Thread(target=yolo_thread)

t_cam.start()
t_yolo.start()

# Loop principal: afișare rezultate
try:
    while True:
        if not result_queue.empty():
            frame, results = result_queue.get()
            annotated = results[0].plot()

            # Afișăm ambele FPS-uri
            cv2.putText(annotated, f"Camera FPS: {camera_fps:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.putText(annotated, f"YOLO FPS: {yolo_fps:.2f}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.imshow("YOLO - THREADING PARALLEL", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass

running = False
t_cam.join()
t_yolo.join()
cv2.destroyAllWindows()
