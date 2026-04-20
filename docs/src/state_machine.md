# 状态机 (State Machine)

## RobotStatus 枚举定义

**文件**: `src/setup.py`

```python
class RobotStatus(Enum):
    SEARCH = "search"       # 找球模式
    PICK = "pick"          # 捡球模式
    FIND_BUCKET = "find_bucket"  # 找桶模式
    PUT_BALL = "put_ball"  # 放球模式
```

## 状态转换图

```
                    ┌─────────────────────────────────────────────────────┐
                    │                                                     │
                    ▼                                                     │
┌─────────┐    ┌─────────┐    ┌──────────────┐    ┌──────────┐    ┌─────────┐
│ SEARCH  │───▶│  PICK   │───▶│ FIND_BUCKET  │───▶│ PUT_BALL │───▶│ SEARCH  │
└─────────┘    └─────────┘    └──────────────┘    └──────────┘    └─────────┘
     ▲              │                 │                │              │
     │              │                 │                │              │
     └──────────────┴─────────────────┴────────────────┴──────────────┘
                              (reset_robot)
```

## 状态详细说明

### 1. SEARCH（找球模式）

**进入条件**：
- 系统启动后初始状态
- PUT_BALL 完成后返回

**行为**：
- 使用 `move_controller()` 控制底盘移动
- 调用 `yolo_infer()` 进行目标检测（找球）
- 检测到球并在视野中央稳定 10 帧后进入 PICK 状态

**目标检测区域**：
```
┌─────────────────────────────────┐
│                                 │
│    ┌───────────────────┐       │
│    │                   │       │
│    │   TARGET ZONE     │       │  ← 球需要在这个区域内
│    │                   │       │
│    └───────────────────┘       │
│                                 │
└─────────────────────────────────┘
```

### 2. PICK（捡球模式）

**进入条件**：
- SEARCH 状态下检测到球并稳定 10 帧

**行为**：
- 使用 `p_control_loop()` 执行 `CATCH_ACTION` 动作序列
- 逆运动学控制机械臂移动到球的位置
- 执行夹爪动作抓住球
- 动作序列完成后进入 FIND_BUCKET 状态

**动作序列**：

```python
CATCH_ACTION = [
    ("move_to", (0.0989, 0.125)),    # 移动到初始位置
    ("shoulder_pan", -12),            # 肩部旋转
    ("gripper", 60),                   # 张开夹爪
    ("wrist_flex", 80),               # 腕部弯曲
    ("move_to", (0.140, 0.1211)),     # 移动到球上方
    ("move_to", (0.140, -0.05)),      # 下降到球位置
    ("gap", 0),                        # 停顿
    ("gripper", -60),                  # 闭合夹爪（夹住球）
    ("gap", 0),                        # 停顿
    ("shoulder_pan", 12),              # 肩部归位
    ("move_to", (-0.1, 0.2)),          # 举起球
    ("wrist_flex", -20)                # 腕部配合
]
```

### 3. FIND_BUCKET（找桶模式）

**进入条件**：
- PICK 状态动作序列执行完成

**行为**：
- 检查夹爪是否仍然夹住球（`gripper_pos > 25`）
- 如果球丢失，返回 SEARCH 状态重新寻找
- 使用 `move_controller_for_bucket()` 控制底盘移动
- 调用 `get_black_bucket_local()` 或 `get_red_bucket_local()` 寻找桶
- 检测到桶并在视野中央稳定 10 帧后进入 PUT_BALL 状态

**找桶策略**：
1. 优先检测黑色桶
2. 如果找不到黑色桶，检测红色桶

### 4. PUT_BALL（放球模式）

**进入条件**：
- FIND_BUCKET 状态下检测到桶并稳定 10 帧

**行为**：
- 使用 `p_control_loop()` 执行 `PUT_ACTION` 动作序列
- 逆运动学控制机械臂移动到桶上方
- 执行夹爪动作释放球
- 动作序列完成后返回 SEARCH 状态

**动作序列**：

```python
PUT_ACTION = [
    ("shoulder_lift", 50),      # 抬起机械臂
    ("gap", 0),                  # 停顿
    ("gripper", 60),             # 张开夹爪（放球）
    ("gap", 0),                  # 停顿
    ("move_to", (-0.1, 0.2)),    # 收回机械臂
    ("gripper", -60),            # 闭合夹爪
]
```

## 状态转换逻辑

**文件**: `src/main.py`

### SEARCH → PICK

```python
# move_controller.py:53-56
if position > (TARGET_POSITION + 10):  # 球太大（太近）
    action = direction.get_action("backward", 1)
elif center_x < left or center_x > right:  # 球偏离中心
    action = direction.get_action("rotate_left/right")
else:
    _cycle_time += 1
    if _cycle_time > 10:  # 稳定10帧
        set_robot_status(RobotStatus.PICK)
```

### PICK → FIND_BUCKET

```python
# main.py:121-123
if CATCH_ACTION[command_step][0] == "move_to":
    if abs(current_x - target_x) < 0.002 and abs(current_y - target_y) < 0.002:
        command_step += 1
        if command_step == len(CATCH_ACTION):
            set_robot_status(RobotStatus.FIND_BUCKET)
```

### FIND_BUCKET → PUT_BALL

```python
# move_controller.py:98-102
else:
    _cycle_time += 1
    if _cycle_time > 10:
        set_robot_status(RobotStatus.PUT_BALL)
```

### FIND_BUCKET → SEARCH（球丢失）

```python
# main.py:80-83
gripper_pos = current_obs.get('arm_gripper.pos', 5)
is_gripper_holding = gripper_pos > 25
if not is_gripper_holding:
    set_robot_status(RobotStatus.SEARCH)
    reset_robot()
```

### PUT_BALL → SEARCH

```python
# main.py:136-146
if command_step == len(PUT_ACTION):
    set_robot_status(RobotStatus.SEARCH)
    reset_robot()
    command_step = 0
```

## 稳定性计数器机制

**文件**: `src/move_controller.py`

```python
_cycle_time = 0  # 全局计数器

def move_controller(...):
    if result:  # 检测到目标
        if 目标在视野中央:
            _cycle_time += 1
            if _cycle_time > 10:  # 稳定10帧
                触发状态转换
        else:
            _cycle_time = 0  # 目标偏离，重置计数器
    else:  # 未检测到目标
        _cycle_time = 0
```

**作用**：
- 防止误触发（目标一闪而过）
- 确保机器人已稳定对准目标
- 10 帧约等于 0.5 秒（20 FPS）

## 控制模式

**文件**: `src/setup.py`

```python
class RobotControlModel(Enum):
    ACT = "act"       # ACT 学习策略控制
    INVERSE = "inverse"  # 逆运动学控制
```

| 模式 | 说明 | 使用场景 |
|------|------|----------|
| `inverse` | 逆运动学控制机械臂 | 默认模式，精确位置控制 |
| `act` | ACT 策略控制 | 模仿学习模式 |

## 目标检测区域参数

**文件**: `src/main.py:46-53`

```python
height, width = 480, 640
_target_w = (min(height, width) // 3) - 10  # 约 150
_target_h = min(height, width) // 3          # 约 160
_left = max(0, (width - _target_w) // 2)     # 约 245
_top = max(0, (height - _target_h) // 2)    # 约 160
_right = min(width, _left + _target_w)       # 约 395
_bottom = min(height, _top + _target_h)      # 约 320
```

这些参数定义了摄像头画面中央的目标区域，用于判断球/桶是否在视野中央。
