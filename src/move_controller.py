from src.lekiwi import DirectionControl
from .setup import get_left, get_bottom, get_right, get_top, get_target_w, get_target_h, set_robot_status, RobotStatus
from .utils import get_nearly_target_box
from src.yolov import Box

target_w = get_target_w()
target_h = get_target_h()

left = get_left()
top = get_top()
right = get_right()
bottom = get_bottom()

TARGET_CX = left + target_w // 2
TARGET_CY = top + target_h // 2

TARGET_POSITION = max(target_w, target_h)

_cycle_time = 0

_last_ball_center_x = None

def move_controller(direction: DirectionControl, result: list[Box]) -> dict[str, float]:
    global _cycle_time, _last_ball_center_x
    if result and len(result) > 0:
        box = get_nearly_target_box(result, TARGET_CX, TARGET_CY)
        x, y, w, h = box.x, box.y, box.w, box.h
        center_x = x + w // 2
        position = max(w, h)
        _last_ball_center_x = center_x
        if center_x < left:
            if abs(TARGET_CX - center_x) < target_w:
                action = direction.get_action("rotate_left", 0)
            else:
                action = direction.get_action("rotate_left")
            _cycle_time = 0
        elif center_x > right:
            if abs(TARGET_CX - center_x) < target_w:
                action = direction.get_action("rotate_right", 0)
            else:
                action = direction.get_action("rotate_right")
            _cycle_time = 0
        elif position < (TARGET_POSITION - 20) * 2:
            if TARGET_POSITION - position < target_h:
                action = direction.get_action("forward", 0)
            else:
                action = direction.get_action("forward")
            _cycle_time = 0
        elif position > (TARGET_POSITION + 20) * 2:
            action = direction.get_action("backward", 0)
            _cycle_time = 0
        else:
            action = direction.get_action(None)
            _cycle_time += 1
            if _cycle_time > 10:
                set_robot_status(RobotStatus.CATCH)
                _cycle_time = 0
    else:
        if _last_ball_center_x is not None:
            frame_center = (left + right) // 2
            if _last_ball_center_x < frame_center:
                action = direction.get_action("rotate_left", 0)
            else:
                action = direction.get_action("rotate_right", 0)
        else:
            action = direction.get_action(None)
    return action

def get_empty_move_action(direction: DirectionControl):
    return direction.get_action(None)