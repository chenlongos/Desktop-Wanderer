"""
Grab test: streams camera with ball detection, waits for Enter,
then picks the ball up, holds it, and drops it.

Mimics main.py behavior exactly: YOLO only runs in idle phase,
control loop runs unthrottled during grab, same p_control_loop + step logic.

Usage:
    python grab_test.py
    Open http://<device-ip>:8080 to see the camera feed.
    Press Enter to trigger a grab cycle. 'q' + Enter to quit.
"""
import sys, os
sys.path.append(os.path.dirname(__file__))

import time
import threading
import logging
import cv2
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

from src.setup import init_app, get_fps
from src.robot_setup import init_robot, get_robot, get_target_positions, reset_robot
from src.yolov import yolo_infer
from src.arm_inverse_controller import (
    p_control_loop, return_to_start_position, generate_catch_action,
)
from src.move_controller import estimate_distance, GRAB_GOAL_CX, CENTER_GRAB_TOLERANCE_PX
from src.utils import busy_wait

logging.root.handlers.clear()
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

HOLD_SECONDS = 2.0

# --- MJPEG stream ---
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


def main():
    global _latest_frame

    init_app()
    init_robot()
    robot = get_robot()
    robot.connect()
    _start_server()

    fps = get_fps()

    start_obs = robot.get_observation()
    start_positions = get_target_positions()
    return_to_start_position(robot, start_obs, start_positions, 0.9, fps)

    x0, y0 = 0.0989, 0.125
    current_x, current_y = x0, y0

    catch_action = None
    command_step = 0
    hold_start = 0.0
    phase = "idle"  # idle, grab, hold, return

    _trigger = threading.Event()
    _quit = threading.Event()

    def _input_thread():
        while not _quit.is_set():
            line = input("Press Enter to grab (or 'q' to quit): ").strip().lower()
            if line == 'q':
                _quit.set()
                return
            _trigger.set()

    threading.Thread(target=_input_thread, daemon=True).start()
    print("\n=== Grab Test ===")
    print("Place a ball in front of the robot, then press Enter.\n")

    try:
        while not _quit.is_set():
            t0 = time.perf_counter()

            current_obs = robot.get_observation()
            frame = current_obs["front"]

            # --- Only run YOLO in idle phase (same as main.py: only in SEARCH) ---
            result = []
            if phase == "idle":
                result = yolo_infer(frame)

            # --- Overlay drawing (only when there are results, like main.py) ---
            for box in result:
                x, y, w, h = box.x, box.y, box.w, box.h
                center_x = x + w // 2
                center_y = y + h // 2
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.line(frame, (GRAB_GOAL_CX - CENTER_GRAB_TOLERANCE_PX, 0),
                         (GRAB_GOAL_CX - CENTER_GRAB_TOLERANCE_PX, 480), (255, 0, 255), 1)
                cv2.line(frame, (GRAB_GOAL_CX + CENTER_GRAB_TOLERANCE_PX, 0),
                         (GRAB_GOAL_CX + CENTER_GRAB_TOLERANCE_PX, 480), (255, 0, 255), 1)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                diameter_px = max(w, h)
                distance_cm = 3632.9975 / diameter_px + -1.3094
                cv2.putText(frame, f"{diameter_px}px {distance_cm:.1f}cm", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            with _frame_lock:
                _latest_frame = frame

            arm_action = {}

            # --- State machine ---
            if phase == "idle":
                if _trigger.is_set():
                    _trigger.clear()
                    if result and len(result) > 0:
                        best = max(result, key=lambda b: max(b.w, b.h))
                        ball_diam = max(best.w, best.h)
                        ball_dist = estimate_distance(ball_diam)
                        catch_action = generate_catch_action(ball_dist)
                        if catch_action is None:
                            print("  No calibration data, cannot grab.")
                        else:
                            command_step = 0
                            current_x, current_y = x0, y0
                            phase = "grab"
                            print(f"  Grabbing at {ball_dist:.1f}cm...")
                    else:
                        print("  No ball detected.")

            elif phase == "grab":
                # --- Exact same logic as main.py PICK state ---
                arm_action, current_x, current_y = p_control_loop(
                    catch_action[command_step], current_x, current_y, current_obs, kp=0.8)

                step = catch_action[command_step]
                has_move_to = False
                if isinstance(step, list):
                    move_target = None
                    for c in step:
                        if c[0] == "move_to":
                            has_move_to = True
                            move_target = c[1]
                    if has_move_to:
                        if abs(current_x - move_target[0]) < 0.002 and abs(current_y - move_target[1]) < 0.002:
                            command_step += 1
                    else:
                        command_step += 1
                elif step[0] == "move_to":
                    has_move_to = True
                    if abs(current_x - step[1][0]) < 0.002 and abs(current_y - step[1][1]) < 0.002:
                        command_step += 1
                else:
                    command_step += 1

                if command_step == len(catch_action):
                    phase = "hold"
                    hold_start = time.monotonic()
                    print(f"  Sequence done. Holding for {HOLD_SECONDS}s...")

            elif phase == "hold":
                # Keep holding the lift position (last step of catch_action)
                arm_action, current_x, current_y = p_control_loop(
                    catch_action[-1], current_x, current_y, current_obs, kp=0.8)
                if time.monotonic() - hold_start >= HOLD_SECONDS:
                    phase = "return"
                    print("  Returning to start...")

            elif phase == "return":
                return_to_start_position(robot, current_obs, start_positions, 0.9, fps)
                reset_robot()
                current_x, current_y = x0, y0
                phase = "idle"
                print("  Done. Ready for next grab.\n")

            robot.send_action({**arm_action, "x.vel": 0, "y.vel": 0, "theta.vel": 0})

            # No busy_wait during grab (same as main.py: no wait during PICK)
            # if phase != "grab":
            #     busy_wait(max(1.0 / fps - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        robot.disconnect()


if __name__ == '__main__':
    main()
