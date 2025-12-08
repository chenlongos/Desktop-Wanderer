import sys
import os
sys.path.append(os.path.dirname(__file__))
import time

import cv2

from lekiwi import LeKiwiConfig
from lekiwi.lekiwi import LeKiwi
from lekiwi.utils import busy_wait

from lekiwi.key_board_teleop import KeyboardTeleop
import logging
import yaml

from yolov.process import process_img

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
logger = logging.getLogger(__name__)
LOG_LEVEL = config['log_level']
logging.basicConfig(level=getattr(logging, LOG_LEVEL))

FPS = 30

def main():
    cfg = LeKiwiConfig(port="/dev/tty.usbmodem5A7C1231451")
    robot = LeKiwi(cfg)
    robot.connect()

    teleop = KeyboardTeleop()
    print("WASD: 移动 | QE: 旋转 | []: 调速 | ESC: 退出")

    logging.info("正在打开摄像头...")
    cap = cv2.VideoCapture(0)

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
                    # cv2.rectangle(frame, (left, top), (right, bottom), color=(255, 255, 0), thickness=2)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                cv2.imshow("frame", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            action = teleop.get_action()
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
