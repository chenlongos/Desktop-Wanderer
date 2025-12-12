import sys
import os

from src.utils import busy_wait
from .move_controller import move_controller

sys.path.append(os.path.dirname(__file__))
import time

import cv2

from src.lekiwi import LeKiwiConfig
from src.lekiwi.lekiwi import LeKiwi

from src.lekiwi.key_board_teleop import KeyboardTeleop
from src.lekiwi.direction_control import DirectionControl
import logging
import yaml

from yolov.process import yolo_infer

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
logger = logging.getLogger(__name__)
LOG_LEVEL = config['log_level']
logging.basicConfig(level=getattr(logging, LOG_LEVEL))

PORT = config['port']
TARGET_W = config['target_w']
TARGET_H = config['target_h']
HARDWARE_MODE = config['hardware_mode']

logging.info("正在打开摄像头...")
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
height, width = frame.shape[:2]

left = (width - TARGET_W) // 2
top = (height - TARGET_H) // 2
right = left + TARGET_W
bottom = top + TARGET_H

FPS = 30


def main():
    cfg = LeKiwiConfig(port=PORT)
    robot = LeKiwi(cfg)
    robot.connect()

    teleop = KeyboardTeleop()
    direction = DirectionControl()

    try:
        while True:
            t0 = time.perf_counter()

            ret, frame = cap.read()
            result = yolo_infer(frame)

            if True:
                for box in result:
                    x, y, w, h = box.x, box.y, box.w, box.h
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

            move_action = move_controller(direction, result)

            _action_sent = robot.send_action(move_action)
            busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    finally:
        teleop.stop()
        robot.disconnect()


if __name__ == '__main__':
    main()
