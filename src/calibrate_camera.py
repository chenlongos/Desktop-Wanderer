"""
Camera focal length calibration using a tennis ball.

Usage: python -m src.calibrate_camera

1. Place the tennis ball at a known distance from the camera.
2. The stream shows the detected ball with its bounding box.
3. Type the real distance (in cm) in the terminal and press Enter.
4. Repeat at different distances (at least 3-5 samples).
5. Type 'done' to compute the focal length and save the result.
"""
import sys
import os
import json
import time

import cv2
import numpy as np

from src.setup import init_app
from src.robot_setup import init_robot, get_robot
from src.yolov import yolo_infer
from src.stream_server import start_stream_server, update_frame

BALL_DIAMETER_CM = 6.54  # tennis ball diameter
CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), 'camera_calibration.json')


def main():
    init_app()
    init_robot()
    robot = get_robot()
    robot.connect()
    start_stream_server()

    samples = []  # list of (distance_cm, diameter_px)

    print("=== Camera Calibration ===")
    print(f"Tennis ball diameter: {BALL_DIAMETER_CM} cm")
    print("Place the ball at a known distance, then type the distance in cm.")
    print("Type 'done' when finished.\n")

    try:
        while True:
            obs = robot.get_observation()
            frame = obs["front"]
            result = yolo_infer(frame)

            diameter_px = None
            if result:
                # Pick the largest detection
                best = max(result, key=lambda b: max(b.w, b.h))
                x, y, w, h = best.x, best.y, best.w, best.h
                cx, cy = x + w // 2, y + h // 2
                diameter_px = max(w, h)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"{diameter_px}px", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Show sample count
            cv2.putText(frame, f"Samples: {len(samples)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            update_frame(frame)

            # Non-blocking check for terminal input
            import select as sel
            ready, _, _ = sel.select([sys.stdin], [], [], 0.03)
            if not ready:
                continue

            line = sys.stdin.readline().strip()
            if line.lower() == 'done':
                break
            if not line:
                continue

            try:
                distance_cm = float(line)
            except ValueError:
                print("  Invalid number, try again.")
                continue

            if diameter_px is None:
                print("  No ball detected in frame, try again.")
                continue

            # focal_length = (diameter_px * distance_cm) / BALL_DIAMETER_CM
            f_sample = (diameter_px * distance_cm) / BALL_DIAMETER_CM
            samples.append((distance_cm, diameter_px, f_sample))
            print(f"  Recorded: distance={distance_cm}cm, bbox={diameter_px}px, f={f_sample:.1f}px")

    finally:
        robot.disconnect()

    if len(samples) < 2:
        print("\nNot enough samples. Need at least 2.")
        return

    distances = np.array([s[0] for s in samples])       # D in cm
    inv_pixels = np.array([1.0 / s[1] for s in samples]) # 1/P

    # Linear regression: D = m * (1/P) + c
    # np.polyfit with degree 1: coeffs[0] = m, coeffs[1] = c
    m, c = np.polyfit(inv_pixels, distances, 1)

    # R² to show fit quality
    predicted = m * inv_pixels + c
    ss_res = np.sum((distances - predicted) ** 2)
    ss_tot = np.sum((distances - np.mean(distances)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    print(f"\n=== Results (Linear Regression: D = m/P + c) ===")
    print(f"Samples: {len(samples)}")
    for d, px, _ in samples:
        d_pred = m / px + c
        print(f"  {d:6.1f} cm  |  {px:4d} px  |  predicted = {d_pred:.1f} cm")
    print(f"\n  m = {m:.2f}")
    print(f"  c = {c:.2f}")
    print(f"  R² = {r_squared:.6f}")
    print(f"\n  Distance formula: D = {m:.2f} / P + ({c:.2f})")

    cal = {
        "m": round(float(m), 4),
        "c": round(float(c), 4),
        "r_squared": round(float(r_squared), 6),
        "ball_diameter_cm": BALL_DIAMETER_CM,
        "samples": [{"distance_cm": d, "diameter_px": px} for d, px, _ in samples],
    }
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(cal, f, indent=2)
    print(f"\nSaved to {CALIBRATION_FILE}")


if __name__ == '__main__':
    main()
