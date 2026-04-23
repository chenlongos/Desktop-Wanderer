from src.lekiwi import DirectionControl
from .setup import get_left, get_bottom, get_right, get_top, get_target_w, get_target_h, set_robot_status, RobotStatus, get_fps
from .utils import get_nearly_target_box
from src.yolov import Box
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 摄像头标定参数: D = M / P + C (cm)
_CAL_M = 2892.91
_CAL_C = 0.27
BEST_DISTANCE_CM = 15
DISTANCE_TOLERANCE_CM = 0.5
CENTER_FIND_TOLERANCE_PX = 50
CENTER_SLOWDOWN_PX = 300
CENTER_GRAB_TOLERANCE_PX = 10
FRAME_WIDTH = 640
FIND_GOAL_CX = FRAME_WIDTH // 2
GRAB_GOAL_CX = FRAME_WIDTH // 2 + 20

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
_stable_count = 0
_move_frame_count = 0
_search_rotate_deg = 0.0  # 累计搜索旋转角度
_search_circles = 0       # 完成了几圈搜索
_search_pause_counter = 0 # 当前暂停帧计数

_last_ball_center_x = None
_last_bucket_center_x = None


def _estimate_distance(diameter_px: int) -> float:
    """从bbox像素直径估算距离(cm)"""
    if diameter_px <= 0:
        return 999.0
    return _CAL_M / diameter_px + _CAL_C


def move_controller(direction: DirectionControl, result: list[Box]) -> dict[str, float]:
    global _cycle_time, _last_ball_center_x, _stable_count, _move_frame_count, _search_rotate_deg, _search_circles, _search_pause_counter
    if result and len(result) > 0:
        box = get_nearly_target_box(result)
        x, y, w, h = box.x, box.y, box.w, box.h
        center_x = x + w // 2
        diameter_px = max(w, h)
        _last_ball_center_x = center_x

        # 第一步：先旋转对准球心（中心 +-10px）
        offset = center_x - FIND_GOAL_CX
        if abs(offset) > CENTER_FIND_TOLERANCE_PX:
            if offset < 0:
                action = direction.get_action(None)
                action['theta.vel'] = 15
            else:
                action = direction.get_action(None)
                action['theta.vel'] = -15
            _stable_count = 0
            return action

        # 第二步：球已居中，根据距离前进/后退
        distance_cm = _estimate_distance(diameter_px)
        error_cm = distance_cm - BEST_DISTANCE_CM

        if abs(error_cm) <= DISTANCE_TOLERANCE_CM:
            _search_rotate_deg = 0.0
            _search_circles = 0
            _search_pause_counter = 0
            offset = center_x - GRAB_GOAL_CX
            if abs(offset) > CENTER_GRAB_TOLERANCE_PX:
                if offset < 0:
                    action = direction.get_action(None)
                    action['theta.vel'] = 5
                else:
                    action = direction.get_action(None)
                    action['theta.vel'] = -5
                _stable_count = 0
                return action
            
            # 距离合适，稳定计数
            if _move_frame_count > 0:
                logger.info(f"stopped after {_move_frame_count} move frames")
                _move_frame_count = 0
            action = direction.get_action(None)
            _stable_count += 1
            if _stable_count > 10:
                set_robot_status(RobotStatus.PICK)
                _stable_count = 0
        else:
            _stable_count = 0
            _move_frame_count += 1
            # 速度 = (距离误差cm / 100) / 帧时间 * 0.8 → m/s
            frame_time = 1.0 / get_fps()
            if error_cm > 10:
                factor = 0.9
            else:
                factor = 0.05
            speed_mps = (error_cm / 100.0) / frame_time * factor
            # 限幅到合理范围
            action = direction.get_action(None)
            action["x.vel"] = speed_mps
            dir_str = "forward" if speed_mps > 0 else "backward"
            logger.info(f"{dir_str}: dist={distance_cm:.1f}cm err={error_cm:.1f}cm vel={speed_mps:.3f}m/s frame#{_move_frame_count}")
    else:
        _stable_count = 0
        frame_time = 1.0 / get_fps()

        # 检查是否完成了一圈（360°）
        circle_threshold = (_search_circles + 1) * 360
        if _search_rotate_deg >= circle_threshold:
            _search_circles += 1
            _search_pause_counter = 0
            logger.info(f"Completed circle #{_search_circles}, will pause {_search_circles} frame(s) between moves")

        # 第一圈正常速度，之后每多一圈，每次移动后暂停更多帧
        if _search_circles == 0:
            speed_level = 2
            deg_per_frame = 100 * frame_time
        elif _search_circles == 1:
            speed_level = 1
            deg_per_frame = 60 * frame_time
        else:
            # 暂停逻辑：每 _search_circles 帧才动一帧
            _search_pause_counter += 1
            if _search_pause_counter <= _search_circles - 2:
                # 暂停帧，不动
                return direction.get_action(None)
            _search_pause_counter = 0
            speed_level = 0
            deg_per_frame = 30 * frame_time

        _search_rotate_deg += deg_per_frame
        logger.info(f"Searched degrees: {_search_rotate_deg:.0f}, circle: {_search_circles}")

        if _last_ball_center_x is not None:
            if _last_ball_center_x < FIND_GOAL_CX:
                action = direction.get_action("rotate_left", speed_level)
            else:
                action = direction.get_action("rotate_right", speed_level)
        else:
            action = direction.get_action(None)
    return action

def move_controller_for_bucket(direction: DirectionControl, result: list[Box], change_status: bool = True) -> dict[str, float]:
    global _cycle_time, _last_bucket_center_x
    if result and len(result) > 0:
        box = get_nearly_target_box(result)
        x, y, w, h = box.x, box.y, box.w, box.h
        center_x = x + w // 2
        position = min(w, h)
        _last_bucket_center_x = center_x
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
            if change_status:
                if _cycle_time > 10: # 10帧稳定存在，则进入下一流程
                    set_robot_status(RobotStatus.PUT_BALL)
                    _cycle_time = 0
    else:
        if _last_bucket_center_x is not None:
            frame_center = (left + right) // 2
            if _last_bucket_center_x < frame_center:
                action = direction.get_action("rotate_left")
            else:
                action = direction.get_action("rotate_right")
        else:
            action = direction.get_action("rotate_left")
    return action

def get_empty_move_action(direction: DirectionControl):
    return direction.get_action(None)
