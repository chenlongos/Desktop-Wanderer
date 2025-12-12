import cv2
import yaml
#
_is_initialized: bool = False
_hardware_mode: str = "normal"
#
_left: int = 0
_top: int = 0
_right: int = 0
_bottom: int = 0
_target_w: int = 0
_target_h: int = 0
#
_port: str = "/dev/ttyUSB0"
_log_level: str = "INFO"


def init_app():
    global _is_initialized, _hardware_mode, _left, _top, _right, _bottom, _port, _log_level, _target_w, _target_h
    if _is_initialized:
        return
    print("Initializing...")
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    _hardware_mode = config['hardware_mode']
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    _target_w = config['target_w']
    _target_h = config['target_h']
    _left = max(0, (width - _target_w) // 2)
    _top = max(0, (height - _target_h) // 2)
    _right = min(width, _left + _target_w)
    _bottom = min(height, _top + _target_h)

    _port = config['port']
    _log_level = config['log_level']
    _is_initialized = True


def get_hardware_mode():
    if not _is_initialized:
        init_app()
    return _hardware_mode


def get_left():
    if not _is_initialized:
        init_app()
    return _left


def get_top():
    if not _is_initialized:
        init_app()
    return _top


def get_right():
    if not _is_initialized:
        init_app()
    return _right


def get_bottom():
    if not _is_initialized:
        init_app()
    return _bottom


def get_port():
    if not _is_initialized:
        init_app()
    return _port


def get_log_level():
    if not _is_initialized:
        init_app()
    return _log_level


def get_target_w():
    if not _is_initialized:
        init_app()
    return _target_w


def get_target_h():
    if not _is_initialized:
        init_app()
    return _target_h
