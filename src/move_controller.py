import math
from src.lekiwi import DirectionControl
from .setup import get_left, get_bottom, get_right, get_top, get_target_w, get_target_h, set_robot_status, RobotStatus, get_fps
from .utils import get_nearly_target_box
from .led_controller import set_state as set_led, BALL_FOUND, SEARCH_A, SEARCH_B
from src.yolov import Box
import logging
import time
import cv2
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Named constants ---

# Camera calibration: D = M / P + C (cm)
_CAL_M = 3632.9975
_CAL_C = -1.3094
BEST_DISTANCE_CM = 15.2
DISTANCE_TOLERANCE_CM = 0.5
CENTER_FIND_TOLERANCE_PX = 50
CENTER_SLOWDOWN_PX = 300
CENTER_GRAB_TOLERANCE_PX = 10
FRAME_WIDTH = 640
FIND_GOAL_CX = FRAME_WIDTH // 2
GRAB_GOAL_CX = FRAME_WIDTH // 2 + 24

# Ball-recently-lost: charge forward duration (seconds)
BALL_LOST_CHARGE_DURATION_S = 2.0
# HSV heatmap: target HSV and cylindrical distance parameters
HSV_TARGET_HUE = 31
HSV_TARGET_SAT = 129
HSV_TARGET_VAL = 255
HSV_SIMILARITY_THRESHOLD = 0.7
HSV_SIGMA = 93
HSV_BLUR_KERNEL = 9
# Blob: center-x must be within this many px of frame center to charge
BLOB_CENTER_TOLERANCE_PX = 100
# Charge-toward-blob duration (seconds)
BLOB_CHARGE_DURATION_S = 2.0
# Search rotation: degrees per full circle
FULL_CIRCLE_DEG = 360

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

_last_ball_center_x = None
_last_bucket_center_x = None
_last_chosen_ball_box: Box | None = None
_last_chosen_bucket_box: Box | None = None
_last_hsv_blob_box: Box | None = None
_last_ball_seen_time = 0.0  # monotonic timestamp of last ball detection

# HSV blob search state
_last_pass_max_blob = 0        # max blob size from the previous completed circle
_current_pass_max_blob = 0     # max blob size being recorded during current circle
_blob_charge_start_time = 0.0  # when we started charging toward a blob
_blob_charging = False          # currently in blob-charge mode


def _estimate_distance(diameter_px: int) -> float:
    """从bbox像素直径估算距离(cm)"""
    if diameter_px <= 0:
        return 999.0
    return _CAL_M / diameter_px + _CAL_C


def _hsv_mask(frame):
    """Compute the HSV similarity mask and connected-component stats.

    Returns (num_labels, stats, centroids) from connectedComponentsWithStats.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    # Cylindrical HSV → Cartesian distance
    hue_rad = hue * (np.pi / 90.0)
    target_rad = HSV_TARGET_HUE * (np.pi / 90.0)

    dx = sat * np.cos(hue_rad) - HSV_TARGET_SAT * np.cos(target_rad)
    dy = sat * np.sin(hue_rad) - HSV_TARGET_SAT * np.sin(target_rad)
    dz = val - HSV_TARGET_VAL

    dist = np.sqrt(dx * dx + dy * dy + dz * dz)
    similarity = np.exp(-0.5 * (dist / HSV_SIGMA) ** 2)

    # Spatial blur to reduce noise
    sim_u8 = (similarity * 255).astype(np.uint8)
    sim_u8 = cv2.GaussianBlur(sim_u8, (HSV_BLUR_KERNEL, HSV_BLUR_KERNEL), 0)
    similarity = sim_u8.astype(np.float32) / 255.0

    # Threshold
    mask = (similarity >= HSV_SIMILARITY_THRESHOLD).astype(np.uint8)

    # 4-connected component analysis
    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    return num_labels, stats, centroids


def _hsv_blob_analysis(frame) -> tuple[int, float]:
    """Compute the largest 4-connected LIKELY blob and its average x coordinate.

    Uses cylindrical HSV distance: H+S mapped to Cartesian (x,y), V as z.
    Returns (max_blob_size, avg_x).  avg_x is -1 if no blob found.
    """
    num_labels, stats, centroids = _hsv_mask(frame)

    max_size = 0
    max_avg_x = -1.0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > max_size:
            max_size = area
            max_avg_x = centroids[i][0]

    return max_size, max_avg_x


# Minimum blob area (pixels) to be considered a valid HSV fallback target
HSV_FALLBACK_MIN_BLOB_AREA = 200
# Distance threshold: if nearest YOLO box is farther than this, prefer HSV blob
HSV_FALLBACK_DISTANCE_CM = 40.0


HSV_BLOB_STICKINESS_WEIGHT = 10
HSV_BLOB_STICKINESS_RADIUS = 300  # px distance where bonus decays to 0


def _hsv_largest_blob_box(frame) -> Box | None:
    """Return a Box for the highest-scoring HSV-matching blob, or None if too small.

    Score = area * (1 + stickiness_bonus), where the bonus favours blobs
    close to the previously chosen HSV blob.
    """
    global _last_hsv_blob_box

    num_labels, stats, _centroids = _hsv_mask(frame)

    best_idx = -1
    best_score = -1e9
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < HSV_FALLBACK_MIN_BLOB_AREA:
            continue

        score = float(area)

        if _last_hsv_blob_box is not None:
            bx = int(stats[i, cv2.CC_STAT_LEFT])
            by = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            cx = bx + bw / 2
            cy = by + bh / 2
            prev_cx = _last_hsv_blob_box.x + _last_hsv_blob_box.w / 2
            prev_cy = _last_hsv_blob_box.y + _last_hsv_blob_box.h / 2
            dist = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
            proximity = max(0.0, 1.0 - dist / HSV_BLOB_STICKINESS_RADIUS)
            score *= (1.0 + HSV_BLOB_STICKINESS_WEIGHT * proximity)

        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx < 0:
        return None

    bx = int(stats[best_idx, cv2.CC_STAT_LEFT])
    by = int(stats[best_idx, cv2.CC_STAT_TOP])
    bw = int(stats[best_idx, cv2.CC_STAT_WIDTH])
    bh = int(stats[best_idx, cv2.CC_STAT_HEIGHT])
    result_box = Box(bx, by, bw, bh)
    if _last_hsv_blob_box is None:
        _last_hsv_blob_box = result_box
    else:
        _last_hsv_blob_box.x = math.floor(_last_hsv_blob_box.x * 0.99 + result_box.x * 0.01)
        _last_hsv_blob_box.y = math.floor(_last_hsv_blob_box.y * 0.99 + result_box.y * 0.01)
        _last_hsv_blob_box.w = math.floor(_last_hsv_blob_box.w * 0.99 + result_box.w * 0.01)
        _last_hsv_blob_box.h = math.floor(_last_hsv_blob_box.h * 0.99 + result_box.h * 0.01)
    return result_box


def move_controller(direction: DirectionControl, result: list[Box], frame=None) -> dict[str, float]:
    global _cycle_time, _last_ball_center_x, _stable_count, _move_frame_count
    global _search_rotate_deg, _search_circles
    global _last_ball_seen_time, _last_pass_max_blob, _current_pass_max_blob, _blob_charge_start_time, _blob_charging
    global _last_chosen_ball_box

    now = time.monotonic()

    # If we're in blob-charge mode, keep going forward unless a ball appears or time's up
    if _blob_charging:
        if result and len(result) > 0:
            # Ball found during charge — stop charging, fall through to normal tracking
            _blob_charging = False
            logger.info("Ball detected during blob charge, switching to tracking")
        elif now - _blob_charge_start_time < BLOB_CHARGE_DURATION_S:
            action = direction.get_action("forward", 3)
            logger.info("Blob charge: driving forward")
            return action
        else:
            # Charge time expired — reset and continue spinning for another 360°
            _blob_charging = False
            _search_rotate_deg = 0.0
            _search_circles = 0
            _last_pass_max_blob = 0
            _current_pass_max_blob = 0
            logger.info("Blob charge expired, resuming search")
            return direction.get_action(None)

    if result and len(result) > 0:
        set_led(BALL_FOUND)
        _last_ball_seen_time = now
        box = get_nearly_target_box(result, _last_chosen_ball_box)
        if _last_chosen_ball_box is None:
            _last_chosen_ball_box = box
        else:
            _last_chosen_ball_box.x = math.floor(_last_chosen_ball_box.x * 0.99 + box.x * 0.01)
            _last_chosen_ball_box.y = math.floor(_last_chosen_ball_box.y * 0.99 + box.y * 0.01)
            _last_chosen_ball_box.w = math.floor(_last_chosen_ball_box.w * 0.99 + box.w * 0.01)
            _last_chosen_ball_box.h = math.floor(_last_chosen_ball_box.h * 0.99 + box.h * 0.01)
        x, y, w, h = box.x, box.y, box.w, box.h
        center_x = x + w // 2
        diameter_px = min(w, h)
        _last_ball_center_x = center_x

        # If the best YOLO box is too far away, try HSV blob as a closer target
        distance_cm = _estimate_distance(diameter_px)
        if distance_cm > HSV_FALLBACK_DISTANCE_CM and frame is not None:
            hsv_box = _hsv_largest_blob_box(frame)
            if hsv_box is not None:
                logger.info(f"YOLO box too far ({distance_cm:.0f}cm), using HSV blob fallback")
                box = hsv_box
                x, y, w, h = box.x, box.y, box.w, box.h
                center_x = x + w // 2
                diameter_px = min(w, h)
                _last_ball_center_x = center_x

        # 第一步：先旋转对准球心（中心 +-10px）
        offset = center_x - FIND_GOAL_CX
        if abs(offset) > 200:
            if offset < 0:
                action = direction.get_action(None)
                action['theta.vel'] = 45
            else:
                action = direction.get_action(None)
                action['theta.vel'] = -45
            _stable_count = 0
            return action
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
            _last_pass_max_blob = 0
            _current_pass_max_blob = 0
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
        # Ball just disappeared — if seen less than threshold ago, charge forward
        if now - _last_ball_seen_time < BALL_LOST_CHARGE_DURATION_S:
            action = direction.get_action("forward", 2)
            logger.info("Ball lost recently, charging forward")
            return action

        _stable_count = 0
        frame_time = 1.0 / get_fps()

        # --- HSV blob analysis during search ---
        blob_size = 0
        blob_avg_x = -1.0
        if frame is not None:
            blob_size, blob_avg_x = _hsv_blob_analysis(frame)

        # Check if we completed a full circle
        circle_threshold = (_search_circles + 1) * FULL_CIRCLE_DEG
        if _search_rotate_deg >= circle_threshold:
            _search_circles += 1
            # Promote current pass max to last pass max for next circle
            _last_pass_max_blob = _current_pass_max_blob
            _current_pass_max_blob = 0
            logger.info(f"Completed circle #{_search_circles}, last pass max blob: {_last_pass_max_blob}")

        set_led(SEARCH_A if _search_circles % 2 == 0 else SEARCH_B)
        
        # Always record the max blob of the current pass
        if blob_size > _current_pass_max_blob:
            _current_pass_max_blob = blob_size

        # From circle 1 onward: charge if blob is at least 90% of last pass's max
        if (_search_circles >= 1
                and _last_pass_max_blob > 0
                and blob_size >= 0.85 * _last_pass_max_blob
                and blob_avg_x >= 0):
            offset_from_center = abs(blob_avg_x - FIND_GOAL_CX)
            if offset_from_center <= BLOB_CENTER_TOLERANCE_PX:
                _blob_charging = True
                _blob_charge_start_time = now
                action = direction.get_action("forward", 2)
                logger.info(f"Blob charge triggered (circle {_search_circles}): size={blob_size} >= 90% of last_max={_last_pass_max_blob} avg_x={blob_avg_x:.0f}")
                return action

        speed_level = 2
        deg_per_frame = 125 * frame_time

        _search_rotate_deg += deg_per_frame
        logger.info(f"Searched degrees: {_search_rotate_deg:.0f}, circle: {_search_circles}, blob: {blob_size}")

        if _last_ball_center_x is not None:
            if _last_ball_center_x < FIND_GOAL_CX:
                action = direction.get_action("rotate_left", speed_level)
            else:
                action = direction.get_action("rotate_right", speed_level)
        else:
            action = direction.get_action("rotate_left", speed=2)
    return action

def move_controller_for_bucket(direction: DirectionControl, result: list[Box], change_status: bool = True) -> dict[str, float]:
    global _cycle_time, _last_bucket_center_x, _last_chosen_bucket_box
    if result and len(result) > 0:
        box = get_nearly_target_box(result, _last_chosen_bucket_box)
        _last_chosen_bucket_box = box
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
        elif position < TARGET_POSITION * 2.4: # 如果桶在摄像头中的直径小于目标框的2.6倍，则前进
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
