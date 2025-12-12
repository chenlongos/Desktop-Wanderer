class DirectionControl:
    def __init__(self):
        # 速度档位
        self.speed_levels = [
            {"xy": 0.1, "theta": 30},   # 慢
            {"xy": 0.25, "theta": 60},  # 中
            {"xy": 0.4, "theta": 90},   # 快
        ]
        self.speed_index = 1  # 默认中速

    def get_action(self, target, speed=None):
        if speed is None:
            speed = self.speed_index
        speed = self.speed_levels[speed]
        xy_speed = speed["xy"]
        theta_speed = speed["theta"]  # 单位：deg/s

        x_cmd = 0.0
        y_cmd = 0.0
        theta_cmd = 0.0

        if target == "forward":
            x_cmd += xy_speed
        if target == "backward":
            x_cmd -= xy_speed
        if target == "left":
            y_cmd += xy_speed
        if target == "right":
            y_cmd -= xy_speed
        if target == "rotate_left":
            theta_cmd += theta_speed
        if target == "rotate_right":
            theta_cmd -= theta_speed

        # 注意：如果下游期望 theta.vel 是 rad/s，需转换：
        # theta_cmd = np.deg2rad(theta_cmd)

        return {
            "x.vel": x_cmd,
            "y.vel": y_cmd,
            "theta.vel": theta_cmd,
        }
