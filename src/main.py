import sys
import os

sys.path.append(os.path.dirname(__file__))
import time

import cv2

from lekiwi import LeKiwiConfig
from lekiwi.lekiwi import LeKiwi
from lekiwi.utils import busy_wait

from lekiwi.key_board_teleop import KeyboardTeleop
from lekiwi.direction_control import DirectionControl
import logging
import yaml

from yolov.process import process_img

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
logger = logging.getLogger(__name__)
LOG_LEVEL = config['log_level']
logging.basicConfig(level=getattr(logging, LOG_LEVEL))

PORT = config['port']
TARGET_W = config['target_w']
TARGET_H = config['target_h']

logging.info("正在打开摄像头...")
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
height, width = frame.shape[:2]

left = (width - TARGET_W) // 2
top = (height - TARGET_H) // 2
right = left + TARGET_W
bottom = top + TARGET_H

left = max(0, left)
top = max(0, top)
right = min(width, right)
bottom = min(height, bottom)

TARGET_CX = left + TARGET_W // 2
TARGET_CY = top + TARGET_H // 2


FPS = 30

def main():
    cfg = LeKiwiConfig(port=PORT)
    robot = LeKiwi(cfg)
    robot.connect()

    teleop = KeyboardTeleop()
    direction = DirectionControl()
    print("WASD: 移动 | QE: 旋转 | []: 调速 | ESC: 退出")

    logging.info("正在打开摄像头...")
    cap = cv2.VideoCapture(0)
    in_zone_frame_count = 0

    try:
        while True:
            t0 = time.perf_counter()

            ret, frame = cap.read()
            result = process_img(frame)

            if True:
                for box in result:
                    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
                    center_x = x + w // 2
                    center_y = y + h // 2
                    pt1, pt2 = (x, y), (x + w, y + h)
                    cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
                    cv2.rectangle(frame, (left, top), (right, bottom), color=(255, 255, 0), thickness=2)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                cv2.imshow("frame", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            if result and len(result) > 0:
                box = result[0]
                if len(result) > 1:
                    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
                    center_x = x + w // 2
                    center_y = y + h // 2
                    dist = (TARGET_CX - center_x) ** 2 + (TARGET_CY - center_y) ** 2
                    for other_box in result[1:]:
                        x, y, w, h = other_box["x"], other_box["y"], other_box["w"], other_box["h"]
                        center_x = x + w // 2
                        center_y = y + h // 2
                        if dist > (TARGET_CX - center_x) ** 2 + (TARGET_CY - center_y) ** 2:
                            box = other_box

                x, y, w, h = box["x"], box["y"], box["w"], box["h"]
                center_x = x + w // 2
                center_y = y + h // 2
                if center_x < left:
                    if abs(TARGET_CX - center_x) < TARGET_W:
                        action = direction.get_action("rotate_left", 0)
                    else:
                        action = direction.get_action("rotate_left")
                    in_zone_frame_count = 0
                elif center_x > right:
                    if abs(TARGET_CX - center_x) < TARGET_W:
                        action = direction.get_action("rotate_right", 0)
                    else:
                        action = direction.get_action("rotate_right")
                    in_zone_frame_count = 0
                elif center_y < top:
                    if abs(TARGET_CY - center_y) < TARGET_H:
                        action = direction.get_action("forward", 0)
                    else:
                        action = direction.get_action("forward")
                    in_zone_frame_count = 0
                elif center_y > bottom:
                    if abs(TARGET_CY - center_y) < TARGET_H:
                        action = direction.get_action("backward", 0)
                    else:
                        action = direction.get_action("backward")
                    in_zone_frame_count = 0
                else:
                    action = direction.get_action(None)
                    logging.debug(f"[TEST] Wait for catch ball for {in_zone_frame_count} step.")
                    in_zone_frame_count += 1
                    if in_zone_frame_count > 10:
                        in_zone_frame_count = 0
            else:
                action = teleop.get_action()
                in_zone_frame_count = 0

            _action_sent = robot.send_action(action)

            # 控制循环频率
            busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        pass
    finally:
        teleop.stop()
        robot.disconnect()

if __name__ == '__main__':
    main()
