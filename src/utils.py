import platform
import time

from .yolov import Box

from .setup import get_left, get_target_w

target_w = get_target_w()

left = get_left()

TARGET_CX = left + target_w // 2
FRAME_WIDTH = 640
HALF_WIDTH = FRAME_WIDTH / 2

def get_nearly_target_box(result: list[Box], prev_box: Box | None = None) -> Box:
    # Score = area, with a bonus for being near center (up to +50% for dead center)
    # and a stickiness bonus for being close to the previously chosen box (up to +80%)
    STICKINESS_WEIGHT = 10
    STICKINESS_RADIUS = 30  # distance at which bonus decays to 0

    def _score(b: Box) -> float:
        area = b.w * b.h
        cx = b.x + b.w / 2
        center_ratio = 1.0 - abs(cx - HALF_WIDTH) / HALF_WIDTH  # 1.0 at center, 0.0 at edge
        score = area

        if prev_box is not None:
            prev_cx = prev_box.x + prev_box.w / 2
            prev_cy = prev_box.y + prev_box.h / 2
            cy = b.y + b.h / 2
            dist = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
            proximity = max(0.0, 1.0 - dist / STICKINESS_RADIUS)
            score *= (1.0 + STICKINESS_WEIGHT * proximity)

        return score
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
