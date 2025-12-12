import yaml
import cv2

from src.lekiwi import DirectionControl
from .utils import get_nearly_target_box
from src.yolov import Box

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
height, width = frame.shape[:2]

TARGET_W = config['target_w']
TARGET_H = config['target_h']
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

TARGET_POSITION = max(TARGET_W, TARGET_H)

def move_controller(direction: DirectionControl, result: list[Box]) -> dict[str, float]:
    if result and len(result) > 0:
        box = get_nearly_target_box(result, TARGET_CX, TARGET_CY)
        x, y, w, h = box.x, box.y, box.w, box.h
        center_x = x + w // 2
        center_y = y + h // 2
        position = max(w, h)
        if center_x < left:
            if abs(TARGET_CX - center_x) < TARGET_W:
                action = direction.get_action("rotate_left", 0)
            else:
                action = direction.get_action("rotate_left")
        elif center_x > right:
            if abs(TARGET_CX - center_x) < TARGET_W:
                action = direction.get_action("rotate_right", 0)
            else:
                action = direction.get_action("rotate_right")
        elif position < TARGET_POSITION:
            if TARGET_POSITION - position < TARGET_H * 2 // 3:
                action = direction.get_action("forward", 0)
            else:
                action = direction.get_action("forward")
        elif center_y > bottom:
            action = direction.get_action(None)
        else:
            action = direction.get_action(None)
    else:
        # action = teleop.get_action()
        # action = direction.get_action("rotate_right", 0)
        action = direction.get_action(None)
    return action
