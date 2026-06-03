"""LED controller for Orange Pi 5 Plus onboard LEDs.

Four states:
  BALL_FOUND  – green on, blue off
  SEARCH_A    – blue on, green off  (odd circles)
  SEARCH_B    – green on, blue on   (even circles)
  OFF         – both off
"""

import logging

logger = logging.getLogger(__name__)

_LED_BASE = "/sys/class/leds"
_LEDS = ("green_led", "blue_led")

BALL_FOUND = "ball_found"
SEARCH_A = "search_a"
SEARCH_B = "search_b"
OFF = "off"

_STATE_MAP = {
    BALL_FOUND: {"green_led": 1, "blue_led": 0},
    SEARCH_A:   {"green_led": 1, "blue_led": 1},
    SEARCH_B:   {"green_led": 0, "blue_led": 1},
    OFF:        {"green_led": 0, "blue_led": 0},
}

_current_state: str | None = None


def _write(path: str, value: str) -> None:
    """Write value to path."""
    try:
        with open(path, 'w') as f:
            f.write(value)
    except Exception as e:
        logger.warning(f"LED write failed ({path}): {e}")


def init() -> None:
    """Disable hardware triggers so brightness is fully manual."""
    for led in _LEDS:
        _write(f"{_LED_BASE}/{led}/trigger", "none")
    set_state(OFF)


def set_state(state: str) -> None:
    """Set LED state. No-op if already in that state."""
    global _current_state
    if state == _current_state:
        return
    values = _STATE_MAP.get(state)
    if values is None:
        logger.warning(f"Unknown LED state: {state}")
        return
    for led, brightness in values.items():
        _write(f"{_LED_BASE}/{led}/brightness", str(brightness))
    _current_state = state
    logger.info(f"LED state -> {state}")