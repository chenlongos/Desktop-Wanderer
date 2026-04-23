"""
Camera-only viewer: streams the robot camera with tennis ball detection overlay.
Shows bounding box, diameter (px), and estimated distance (cm).

Usage:
    python camera_view.py
    Then open http://<device-ip>:8080 in a browser.
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import time
import cv2
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

from src.setup import init_app, get_fps
from src.robot_setup import init_robot, get_robot
from src.yolov import yolo_infer

# Calibration: distance_cm = CAL_M / diameter_px + CAL_C
CAL_M = 2892.91
CAL_C = 0.27

# --- Minimal MJPEG stream server ---
_latest_frame = None
_frame_lock = threading.Lock()

_PAGE = b"""\
<!DOCTYPE html>
<html>
<head><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  body{margin:0;background:#111;display:flex;justify-content:center;align-items:center;height:100vh}
  img{width:100%;max-width:640px}
</style></head>
<body><img src="/stream"></body>
</html>
"""

class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    with _frame_lock:
                        if _latest_frame is None:
                            continue
                        _, jpeg = cv2.imencode('.jpg', _latest_frame)
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n')
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b'\r\n')
            except BrokenPipeError:
                pass
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(_PAGE)

    def log_message(self, *_):
        pass

class _ThreadedServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

def _start_server(port=8080):
    server = _ThreadedServer(('0.0.0.0', port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Stream at http://0.0.0.0:{port}")

# --- Main loop ---
def main():
    global _latest_frame

    init_app()
    init_robot()
    robot = get_robot()
    robot.connect()
    _start_server()

    fps = get_fps()
    print("Streaming camera with tennis ball detection. Ctrl+C to stop.")

    try:
        while True:
            t0 = time.perf_counter()

            obs = robot.get_observation()
            frame = obs["front"]

            # Detect tennis balls
            detections = yolo_infer(frame)

            for box in detections:
                x, y, w, h = box.x, box.y, box.w, box.h
                cx, cy = x + w // 2, y + h // 2
                diameter_px = max(w, h)
                distance_cm = CAL_M / diameter_px + CAL_C

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"{diameter_px}px  {distance_cm:.1f}cm",
                            (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            with _frame_lock:
                _latest_frame = frame

            elapsed = time.perf_counter() - t0
            time.sleep(max(1.0 / fps - elapsed, 0))
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        robot.disconnect()

if __name__ == '__main__':
    main()
