from pynput import keyboard


class KeyboardTeleop:
    def __init__(self):
        # 当前按键状态（True=按下）
        self.keys = {
            "forward": False,    # W
            "backward": False,   # S
            "left": False,       # A
            "right": False,      # D
            "rotate_left": False,   # Q
            "rotate_right": False,  # E
            "speed_up": False,      # ]
            "speed_down": False,    # [
        }

        # 速度档位
        self.speed_levels = [
            {"xy": 0.1, "theta": 30},   # 慢
            {"xy": 0.25, "theta": 60},  # 中
            {"xy": 0.4, "theta": 90},   # 快
        ]
        self.speed_index = 1  # 默认中速

        # 启动监听器（后台线程）
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        try:
            char = key.char.lower()
            if char == 'w':
                self.keys["forward"] = True
            elif char == 's':
                self.keys["backward"] = True
            elif char == 'a':
                self.keys["left"] = True
            elif char == 'd':
                self.keys["right"] = True
            elif char == 'q':
                self.keys["rotate_left"] = True
            elif char == 'e':
                self.keys["rotate_right"] = True
            elif char == ']':
                self.keys["speed_up"] = True
            elif char == '[':
                self.keys["speed_down"] = True
        except AttributeError:
            pass  # 忽略特殊键

    def on_release(self, key):
        try:
            char = key.char.lower()
            if char == 'w':
                self.keys["forward"] = False
            elif char == 's':
                self.keys["backward"] = False
            elif char == 'a':
                self.keys["left"] = False
            elif char == 'd':
                self.keys["right"] = False
            elif char == 'q':
                self.keys["rotate_left"] = False
            elif char == 'e':
                self.keys["rotate_right"] = False
            elif char == ']':
                self.keys["speed_up"] = False
            elif char == '[':
                self.keys["speed_down"] = False
            elif key == keyboard.Key.esc:
                return False  # 停止监听
        except AttributeError:
            pass

    def get_action(self):
        # 处理速度档位（只在按键按下时切换一次）
        if self.keys["speed_up"]:
            self.speed_index = min(self.speed_index + 1, len(self.speed_levels) - 1)
            self.keys["speed_up"] = False  # 防止持续触发
        if self.keys["speed_down"]:
            self.speed_index = max(self.speed_index - 1, 0)
            self.keys["speed_down"] = False

        speed = self.speed_levels[self.speed_index]
        xy_speed = speed["xy"]
        theta_speed = speed["theta"]  # 单位：deg/s

        x_cmd = 0.0
        y_cmd = 0.0
        theta_cmd = 0.0

        if self.keys["forward"]:
            x_cmd += xy_speed
        if self.keys["backward"]:
            x_cmd -= xy_speed
        if self.keys["left"]:
            y_cmd += xy_speed
        if self.keys["right"]:
            y_cmd -= xy_speed
        if self.keys["rotate_left"]:
            theta_cmd += theta_speed
        if self.keys["rotate_right"]:
            theta_cmd -= theta_speed

        # 注意：如果下游期望 theta.vel 是 rad/s，需转换：
        # theta_cmd = np.deg2rad(theta_cmd)

        return {
            "x.vel": x_cmd,
            "y.vel": y_cmd,
            "theta.vel": theta_cmd,
        }

    def stop(self):
        self.listener.stop()
