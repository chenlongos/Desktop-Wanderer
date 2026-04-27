"""
Arm grab-distance calibration tool.

Workflow:
  1. Connects to the robot, starts the camera stream with ball detection.
  2. Disables torque on all arm motors so you can pose them freely.
  3. Place the ball at some distance in front of the robot.
  4. Manually move the arm so the gripper is perpendicular to the ground
     and ready to grab the ball.
  5. Press Enter in the terminal — the script records:
       - estimated ball distance (cm) from the camera
       - all six arm joint positions (calibrated degrees)
  6. Repeat for as many distances as you like.
  7. Press 'q' + Enter (or Ctrl-C) to finish. Data is saved to a JSON file.

Usage:
    python calibrate_arm.py
    Then open http://<device-ip>:8080 to see the camera feed.
"""
import sys, os
sys.path.append(os.path.dirname(__file__))

import time
import json
import threading
import cv2
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

from src.setup import init_app, get_fps
from src.robot_setup import init_robot, get_robot
from src.yolov import yolo_infer
from src.arm_inverse_controller import apply_joint_calibration
from src.move_controller import GRAB_GOAL_CX, CENTER_GRAB_TOLERANCE_PX

# Distance estimation: distance_cm = CAL_M / diameter_px + CAL_C
CAL_M = 2892.91
CAL_C = 0.27

OUTPUT_FILE = "arm_distance_calibration.json"

# --- MJPEG stream (minimal, from camera_view.py) ---
_latest_frame = None
_frame_lock = threading.Lock()

_PAGE = b"""\
<!DOCTYPE html>
<html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{margin:0;background:#111;display:flex;justify-content:center;padding:10px}
img{width:100%;max-width:960px}</style></head>
<body><img src="/stream"></body></html>
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

# --- Latest detection state (shared between camera thread and main thread) ---
_detection_lock = threading.Lock()
_latest_distance_cm = None  # None = no ball detected
_latest_diameter_px = None

ARM_JOINT_NAMES = [
    "arm_shoulder_pan",
    "arm_shoulder_lift",
    "arm_elbow_flex",
    "arm_wrist_flex",
    "arm_wrist_roll",
    "arm_gripper",
]


def _camera_loop(robot, fps):
    """Runs in a background thread: reads frames, runs detection, updates stream."""
    global _latest_frame, _latest_distance_cm, _latest_diameter_px

    while True:
        t0 = time.perf_counter()
        obs = robot.get_observation()
        frame = obs["front"]

        detections = yolo_infer(frame)

        dist_cm = None
        diam_px = None
        for box in detections:
            x, y, w, h = box.x, box.y, box.w, box.h
            cx, cy = x + w // 2, y + h // 2
            diameter_px = max(w, h)
            distance_cm = CAL_M / diameter_px + CAL_C if diameter_px > 0 else 999.0

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"{diameter_px}px  {distance_cm:.1f}cm",
                        (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # Purple center grab tolerance lines
            cv2.line(frame, (GRAB_GOAL_CX - CENTER_GRAB_TOLERANCE_PX, 0),
                     (GRAB_GOAL_CX - CENTER_GRAB_TOLERANCE_PX, 480), (255, 0, 255), 1)
            cv2.line(frame, (GRAB_GOAL_CX + CENTER_GRAB_TOLERANCE_PX, 0),
                     (GRAB_GOAL_CX + CENTER_GRAB_TOLERANCE_PX, 480), (255, 0, 255), 1)

            # Keep the largest detection (closest ball)
            if dist_cm is None or diameter_px > diam_px:
                dist_cm = distance_cm
                diam_px = diameter_px

        with _detection_lock:
            _latest_distance_cm = dist_cm
            _latest_diameter_px = diam_px

        with _frame_lock:
            _latest_frame = frame

        elapsed = time.perf_counter() - t0
        time.sleep(max(1.0 / fps - elapsed, 0))


def _read_arm_positions(obs):
    """Read calibrated arm joint positions from an observation dict."""
    positions = {}
    for name in ARM_JOINT_NAMES:
        key = f"{name}.pos"
        if key in obs:
            raw = obs[key]
            calibrated = apply_joint_calibration(name, raw)
            positions[name] = round(float(calibrated), 3)
    return positions

def main():
    init_app()
    init_robot()
    robot = get_robot()
    robot.connect()

    # Disable torque on arm motors so user can pose them freely
    print("Disabling arm torque — you can now move the arm by hand.")
    robot.bus.disable_torque(robot.arm_motors)

    _start_server()
    fps = get_fps()

    # Start camera + detection in background
    cam_thread = threading.Thread(target=_camera_loop, args=(robot, fps), daemon=True)
    cam_thread.start()

    samples = []

    print("\n=== Arm Distance Calibration ===")
    print("1. Place the ball at a distance in front of the robot.")
    print("2. Move the arm so the gripper is perpendicular to the ground,")
    print("   positioned as if grabbing the ball.")
    print("3. Press Enter to record the sample.")
    print("4. Type 'q' + Enter to finish and save.\n")

    try:
        while True:
            user_input = input("Press Enter to record (or 'q' to quit): ").strip().lower()
            if user_input == 'q':
                break

            # Read current ball distance
            with _detection_lock:
                dist_cm = _latest_distance_cm
                diam_px = _latest_diameter_px

            if dist_cm is None:
                print("  ⚠ No ball detected — make sure the ball is visible. Try again.")
                continue

            # Read current arm joint positions (need a fresh observation)
            obs = robot.get_observation()
            positions = _read_arm_positions(obs)

            sample = {
                "distance_cm": round(dist_cm, 2),
                "diameter_px": diam_px,
                "joints": positions,
            }
            samples.append(sample)

            print(f"  ✓ Sample #{len(samples)}: distance={dist_cm:.1f}cm, "
                  f"diameter={diam_px}px")
            for name, val in positions.items():
                print(f"      {name}: {val}°")
            print()

    except KeyboardInterrupt:
        print("\nInterrupted.")

    # Save
    if samples:
        # Load existing data if present, append new samples
        existing = []
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, 'r') as f:
                existing = json.load(f)
        existing.extend(samples)
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(existing, f, indent=2)
        print(f"\nSaved {len(samples)} new sample(s) to {OUTPUT_FILE} "
              f"({len(existing)} total).")
    else:
        print("\nNo samples recorded.")

    robot.disconnect()


if __name__ == '__main__':
    main()
