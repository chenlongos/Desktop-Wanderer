from src.lekiwi import DirectionControl
from .state import get_left, get_bottom, get_right, get_top, get_target_w, get_target_h, set_robot_status, RobotStatus
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

def move_controller(direction: DirectionControl, result: list[Box]) -> dict[str, float]:
    if result and len(result) > 0:
        box = get_nearly_target_box(result, TARGET_CX, TARGET_CY)
        x, y, w, h = box.x, box.y, box.w, box.h
        center_x = x + w // 2
        position = max(w, h)
        if center_x < left:
            if abs(TARGET_CX - center_x) < target_w:
                action = direction.get_action("rotate_left", 0)
            else:
                action = direction.get_action("rotate_left")
        elif center_x > right:
            if abs(TARGET_CX - center_x) < target_w:
                action = direction.get_action("rotate_right", 0)
            else:
                action = direction.get_action("rotate_right")
        elif position < (TARGET_POSITION - 20) * 2:
            if TARGET_POSITION - position < target_h:
                action = direction.get_action("forward", 0)
            else:
                action = direction.get_action("forward")
        elif position > (TARGET_POSITION + 20) * 2:
            action = direction.get_action("backward", 0)
        else:
            action = direction.get_action(None)
            set_robot_status(RobotStatus.CATCH)
    else:
        # action = teleop.get_action()
        # action = direction.get_action("rotate_right", 0)
        action = direction.get_action(None)
    return action

def get_empty_move_action(direction: DirectionControl):
    return direction.get_action(None)