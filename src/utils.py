import platform
import time

from .yolov import Box

from .setup import get_left, get_target_w
from .move_controller import FRAME_WIDTH

target_w = get_target_w()

left = get_left()

TARGET_CX = left + target_w // 2
HALF_WIDTH = FRAME_WIDTH / 2

def get_nearly_target_box(result: list[Box]) -> Box:
    # Score = area, with a bonus for being near center (up to +50% for dead center)
    def _score(b: Box) -> float:
        area = b.w * b.h
        cx = b.x + b.w / 2
        center_ratio = 1.0 - abs(cx - HALF_WIDTH) / HALF_WIDTH  # 1.0 at center, 0.0 at edge
        return area * (1.0 + 0.5 * center_ratio)
    return max(result, key=_score)


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
