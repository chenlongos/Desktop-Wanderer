import platform
import time

from .yolov import Box


def get_nearly_target_box(result: list[Box]) -> Box:
    box = result[0]
    if len(result) > 1:
        x, y, w, h = box.x, box.y, box.w, box.h
        area = w * h
        for other_box in result[1:]:
            x, y, w, h = other_box.x, other_box.y, other_box.w, other_box.h
            if area < w * h:
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
