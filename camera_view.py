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
import numpy as np
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

from src.setup import init_app, get_fps
from src.robot_setup import init_robot, get_robot
from src.yolov import yolo_infer

# Calibration: distance_cm = CAL_M / diameter_px + CAL_C
CAL_M = 2892.91
CAL_C = 0.27

# --- Tunable heatmap parameters (thread-safe) ---
_params_lock = threading.Lock()
_hue_target = 40       # H value the heatmap peaks at (0-180)
_sat_target = 200      # S value (0-255)
_val_target = 200      # V value (0-255)
_blur_radius = 9       # GaussianBlur kernel size (odd, >=1)
_hue_sigma = 80.0      # Gaussian sigma for cylindrical HSV distance

# --- Minimal MJPEG stream server ---
_latest_frame = None
_frame_lock = threading.Lock()

_PAGE = b"""\
<!DOCTYPE html>
<html>
<head><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  body{margin:0;background:#111;color:#eee;font-family:monospace;
       display:flex;flex-direction:column;align-items:center;padding:10px}
  img{width:100%;max-width:960px}
  .controls{display:flex;gap:18px;flex-wrap:wrap;justify-content:center;
            padding:8px 0;font-size:13px}
  .controls label{display:flex;align-items:center;gap:6px}
  .controls input[type=range]{width:120px}
  .val{min-width:28px;text-align:right}
</style></head>
<body>
<div class="controls">
  <label>Hue
    <input type="range" id="hue" min="0" max="180" value="40">
    <span class="val" id="hueV">40</span>
  </label>
  <label>Sat
    <input type="range" id="sat" min="0" max="255" value="200">
    <span class="val" id="satV">200</span>
  </label>
  <label>Val
    <input type="range" id="val" min="0" max="255" value="200">
    <span class="val" id="valV">200</span>
  </label>
  <label>Blur
    <input type="range" id="blur" min="1" max="31" step="2" value="9">
    <span class="val" id="blurV">9</span>
  </label>
  <label>Sigma
    <input type="range" id="sigma" min="1" max="120" value="80">
    <span class="val" id="sigmaV">80</span>
  </label>
</div>
<img src="/stream">
<script>
function send(k,v){fetch('/api/params?'+k+'='+v)}
document.getElementById('hue').oninput=function(){
  document.getElementById('hueV').textContent=this.value;send('hue',this.value)};
document.getElementById('sat').oninput=function(){
  document.getElementById('satV').textContent=this.value;send('sat',this.value)};
document.getElementById('val').oninput=function(){
  document.getElementById('valV').textContent=this.value;send('val',this.value)};
document.getElementById('blur').oninput=function(){
  var v=this.value%2===0?+this.value+1:+this.value;this.value=v;
  document.getElementById('blurV').textContent=v;send('blur',v)};
document.getElementById('sigma').oninput=function(){
  document.getElementById('sigmaV').textContent=this.value;send('sigma',this.value)};
</script>
</body>
</html>
"""

class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        global _hue_target, _sat_target, _val_target, _blur_radius, _hue_sigma
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
        elif self.path.startswith('/api/params'):
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(self.path).query)
            with _params_lock:
                if 'hue' in qs:
                    _hue_target = max(0, min(180, int(qs['hue'][0])))
                if 'sat' in qs:
                    _sat_target = max(0, min(255, int(qs['sat'][0])))
                if 'val' in qs:
                    _val_target = max(0, min(255, int(qs['val'][0])))
                if 'blur' in qs:
                    v = max(1, min(31, int(qs['blur'][0])))
                    _blur_radius = v if v % 2 == 1 else v + 1
                if 'sigma' in qs:
                    _hue_sigma = max(1.0, min(120.0, float(qs['sigma'][0])))
            self.send_response(204)
            self.end_headers()
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

            # --- Cylindrical HSV distance heatmap (computed on clean frame) ---
            with _params_lock:
                target_hue = _hue_target
                target_sat = _sat_target
                target_val = _val_target
                blur_k = _blur_radius
                sigma = _hue_sigma

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hue = hsv_frame[:, :, 0].astype(np.float32)   # 0-180
            sat = hsv_frame[:, :, 1].astype(np.float32)   # 0-255
            val = hsv_frame[:, :, 2].astype(np.float32)   # 0-255

            # Convert H+S to Cartesian on the color wheel, V as z-axis
            hue_rad = hue * (np.pi / 90.0)          # 0-180 → 0-2π
            target_rad = target_hue * (np.pi / 90.0)

            dx = sat * np.cos(hue_rad) - target_sat * np.cos(target_rad)
            dy = sat * np.sin(hue_rad) - target_sat * np.sin(target_rad)
            dz = val - target_val

            dist = np.sqrt(dx * dx + dy * dy + dz * dz)

            # Gaussian similarity from distance
            similarity = np.exp(-0.5 * (dist / sigma) ** 2)
            heat = (similarity * 255).astype(np.uint8)

            # Spatial blur to suppress per-pixel noise
            heat = cv2.GaussianBlur(heat, (blur_k, blur_k), 0)

            heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

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

            # --- Center crosshair + Gaussian-weighted average color ---
            fH, fW = frame.shape[:2]
            mid_x, mid_y = fW // 2, fH // 2
            cross_len = 20

            # Draw crosshair
            cv2.line(frame, (mid_x - cross_len, mid_y), (mid_x + cross_len, mid_y), (255, 255, 255), 1)
            cv2.line(frame, (mid_x, mid_y - cross_len), (mid_x, mid_y + cross_len), (255, 255, 255), 1)

            # Sample a patch around center and compute Gaussian-weighted average.
            # Gaussian weighting emphasizes the very center and smoothly falls off,
            # so edge pixels / background bleed don't skew the reading.
            R = 30  # patch radius
            y1, y2 = max(mid_y - R, 0), min(mid_y + R, fH)
            x1, x2 = max(mid_x - R, 0), min(mid_x + R, fW)
            patch = frame[y1:y2, x1:x2].astype(np.float64)

            # 2D Gaussian kernel (sigma = R/2 keeps ~95% weight inside the patch)
            gy = np.exp(-0.5 * ((np.arange(patch.shape[0]) - patch.shape[0] / 2) / (R / 2)) ** 2)
            gx = np.exp(-0.5 * ((np.arange(patch.shape[1]) - patch.shape[1] / 2) / (R / 2)) ** 2)
            kernel = np.outer(gy, gx)
            kernel /= kernel.sum()

            avg_b = (patch[:, :, 0] * kernel).sum()
            avg_g = (patch[:, :, 1] * kernel).sum()
            avg_r = (patch[:, :, 2] * kernel).sum()
            avg_bgr = (int(avg_b), int(avg_g), int(avg_r))

            # HSV is more useful for tuning color-based detection
            hsv_pixel = cv2.cvtColor(np.uint8([[avg_bgr]]), cv2.COLOR_BGR2HSV)[0][0]

            # Color swatch + HSV readout next to crosshair
            cv2.rectangle(frame, (mid_x + cross_len + 6, mid_y - 12),
                          (mid_x + cross_len + 30, mid_y + 12), avg_bgr, -1)
            cv2.putText(frame, f"H:{hsv_pixel[0]} S:{hsv_pixel[1]} V:{hsv_pixel[2]}",
                        (mid_x + cross_len + 35, mid_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Stack original frame and heatmap side-by-side
            combined = np.hstack((frame, heatmap))

            with _frame_lock:
                _latest_frame = combined

            elapsed = time.perf_counter() - t0
            time.sleep(max(1.0 / fps - elapsed, 0))
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        robot.disconnect()

if __name__ == '__main__':
    main()
