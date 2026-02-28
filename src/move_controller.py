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
        box = get_nearly_target_box(result)
        x, y, w, h = box.x, box.y, box.w, box.h
        center_x = x + w // 2
        position = max(w, h)
        _last_ball_center_x = center_x
        if center_x < left: # 如果球位于目标框左侧
            if abs(TARGET_CX - center_x) < target_w * 1.5: # 球与目标框中点距离小于目标框的1.5倍，慢速前进
                action = direction.get_action("rotate_left", 1)
            else:
                action = direction.get_action("rotate_left")
            _cycle_time = 0
        elif center_x > right: # 如果球位于目标框右侧
            if abs(TARGET_CX - center_x) < target_w * 1.5: # 与目标框中点距离小于目标框的1.5倍，慢速前进
                action = direction.get_action("rotate_right", 1)
            else:
                action = direction.get_action("rotate_right")
            _cycle_time = 0
        elif position < (TARGET_POSITION - 10): # 如果球在摄像头中的直径小于，目标框-8的2倍 ，则前进（可调）
            if position * 0.8 > TARGET_POSITION:
                action = direction.get_action("forward", 0)
            else:
                action = direction.get_action("forward", 3)
            _cycle_time = 0
        elif position > (TARGET_POSITION + 10): # 如果球在摄像头中的直径大于，目标框+10的2倍 ，则后退（可调）
            action = direction.get_action("backward", 1)
            _cycle_time = 0
        else:
            action = direction.get_action(None)
            _cycle_time += 1
            if _cycle_time > 10: # 10帧稳定存在，则进入下一流程
                set_robot_status(RobotStatus.PICK)
                _cycle_time = 0
    else:
        if _last_ball_center_x is not None:
            frame_center = (left + right) // 2
            if _last_ball_center_x < frame_center:
                action = direction.get_action("rotate_left")
            else:
                action = direction.get_action("rotate_right")
        else:
            action = direction.get_action(None)
    return action

def move_controller_for_bucket(direction: DirectionControl, result: list[Box]) -> dict[str, float]:
    global _cycle_time, _last_ball_center_x
    if result and len(result) > 0:
        box = get_nearly_target_box(result)
        x, y, w, h = box.x, box.y, box.w, box.h
        center_x = x + w // 2
        position = min(w, h)
        _last_ball_center_x = center_x
        if center_x < left: # 如果桶位于目标框左侧
            if abs(TARGET_CX - center_x) < target_w:
                action = direction.get_action("rotate_left")
            else:
                action = direction.get_action("rotate_left")
            _cycle_time = 0
        elif center_x > right: # 如果桶位于目标框右侧
            if abs(TARGET_CX - center_x) < target_w:
                action = direction.get_action("rotate_right")
            else:
                action = direction.get_action("rotate_right")
            _cycle_time = 0
        elif position < TARGET_POSITION * 2.6: # 如果桶在摄像头中的直径小于目标框的2.6倍，则前进
            if TARGET_POSITION - position < target_h: # 保证快速前进，if可以去掉
                action = direction.get_action("forward")
            else:
                action = direction.get_action("forward")
            _cycle_time = 0
        elif position > TARGET_POSITION * 3: # 如果桶在摄像头中的直径大于目标框的3倍，则后退
            action = direction.get_action("backward", 1)
            _cycle_time = 0
        else:
            action = direction.get_action(None)
            _cycle_time += 1
            if _cycle_time > 10: # 10帧稳定存在，则进入下一流程
                set_robot_status(RobotStatus.PUT_BALL)
                _cycle_time = 0
    else:
        if _last_ball_center_x is not None:
            frame_center = (left + right) // 2
            if _last_ball_center_x < frame_center:
                action = direction.get_action("rotate_left")
            else:
                action = direction.get_action("rotate_right")
        else:
            action = direction.get_action(None)
    return action

def get_empty_move_action(direction: DirectionControl):
    return direction.get_action(None)