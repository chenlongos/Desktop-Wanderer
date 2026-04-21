"""
通过 /dev/input 读取物理键盘的按键按下/释放事件。
需要 evdev 库: pip install evdev
运行时需要 root 权限或将用户加入 input 组。
"""
import threading
import evdev
from evdev import ecodes


_space_held = False
_quit_pressed = False
_lock = threading.Lock()


def _find_keyboard():
    """自动查找第一个有 KEY 事件能力的输入设备。"""
    devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
    for dev in devices:
        caps = dev.capabilities(verbose=False)
        # EV_KEY = 1
        if ecodes.EV_KEY in caps:
            return dev
    raise RuntimeError("No keyboard found in /dev/input/. Check permissions or plug in a keyboard.")


def _reader_loop(dev):
    global _space_held, _quit_pressed
    for event in dev.read_loop():
        if event.type != ecodes.EV_KEY:
            continue
        if event.code == ecodes.KEY_SPACE:
            with _lock:
                # value: 1=press, 0=release, 2=repeat
                _space_held = event.value != 0
        elif event.code == ecodes.KEY_Q:
            if event.value == 1:  # press
                with _lock:
                    _quit_pressed = True


def start_key_listener():
    """启动后台线程监听键盘事件。调用一次即可。"""
    dev = _find_keyboard()
    print(f"Listening on keyboard: {dev.name} ({dev.path})")
    t = threading.Thread(target=_reader_loop, args=(dev,), daemon=True)
    t.start()


def is_space_held():
    with _lock:
        return _space_held


def is_quit_pressed():
    with _lock:
        return _quit_pressed
