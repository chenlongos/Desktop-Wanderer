import platform
import time

from .yolov import Box

def get_nearly_target_box(result: list[Box], target_center_x: int, target_center_y: int) -> Box:
    box = result[0]
    if len(result) > 1:
        x, y, w, h = box.x, box.y, box.w, box.h
        center_x = x + w // 2
        center_y = y + h // 2
        dist = (target_center_x - center_x) ** 2 + (target_center_y - center_y) ** 2
        for other_box in result[1:]:
            x, y, w, h = other_box.x, other_box.y, other_box.w, other_box.h
            center_x = x + w // 2
            center_y = y + h // 2
            if dist > (target_center_x - center_x) ** 2 + (target_center_y - center_y) ** 2:
                box = other_box
    return box

def busy_wait(seconds):
    if platform.system() == "Darwin" or platform.system() == "Windows":
        # On Mac and Windows, `time.sleep` is not accurate and we need to use this while loop trick,
        # but it consumes CPU cycles.
        end_time = time.perf_counter() + seconds
        while time.perf_counter() < end_time:
            pass
    else:
        # On Linux time.sleep is accurate
        if seconds > 0:
            time.sleep(seconds)