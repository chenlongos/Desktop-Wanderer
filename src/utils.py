import platform
import time

from .yolov import Box

from .setup import get_left, get_target_w

target_w = get_target_w()

left = get_left()

TARGET_CX = left + target_w // 2

def get_nearly_target_box(result: list[Box]) -> Box:
    box = result[0]
    if len(result) > 1:
        x, y, w, h = box.x, box.y, box.w, box.h
        center_x = x + w // 2
        area = w * h - (abs(TARGET_CX - center_x) * h) * 0.5
        for other_box in result[1:]:
            x, y, w, h = other_box.x, other_box.y, other_box.w, other_box.h
            center_x = x + w // 2
            if area < w * h - (abs(TARGET_CX - center_x) * h) * 0.5:
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
