import logging
import math
import time
import traceback
from time import sleep

from src.robot_setup import get_target_positions, get_pitch, set_pitch

JOINT_CALIBRATION = [
    ['arm_shoulder_pan', 6.0, 1.0],  # Joint 1: zero position offset, scale factor
    ['arm_shoulder_lift', 2.0, 0.97],  # Joint 2: zero position offset, scale factor
    ['arm_elbow_flex', 0.0, 1.05],  # Joint 3: zero position offset, scale factor
    ['arm_wrist_flex', 0.0, 0.94],  # Joint 4: zero position offset, scale factor
    ['arm_wrist_roll', 0.0, 0.5],  # Joint 5: zero position offset, scale factor
    ['arm_gripper', 0.0, 1.0],  # Joint 6: zero position offset, scale factor
]

# Joint control mapping
joint_controls = {
    'shoulder_pan': 'arm_shoulder_pan',  # Joint 1 decrease
    'wrist_roll': 'arm_wrist_roll',  # Joint 5 increase
    'gripper': 'arm_gripper',  # Joint 6 increase
}

# --- 新增：运动控制参数配置 ---
MOTION_PARAMS = {
    'max_step': 0.01,  # 最大步长 (距离远时，每帧移动 2cm) - "快"
    'min_step': 0.001,  # 最小步长 (精细调整时，每帧移动 1mm) - "准"
    'slow_dist': 0.05,  # 减速距离 (距离目标 5cm 以内开始减速)
    'dead_zone': 0.0005  # 死区 (误差小于 0.5mm 认为到达，停止计算)
}


def p_control_loop(cmd, current_x, current_y, current_obs, kp=0.5):
    """
    P control loop

    Args:
        cmd
        current_x: current x coordinate
        current_y: current y coordinate
        current_obs: current observation
        kp: proportional gain
    """

    # Initialize pitch control variables
    target_positions = get_target_positions()
    pitch = get_pitch()

    move_command_list = []
    joint_command_list = []
    wrist_command_list = []

    # 初始化位移增量
    delta_x = 0.0
    delta_y = 0.0

    try:
        cmd_name = cmd[0]
        if cmd_name == "gap":
            sleep(0.2)

        # ---------------------------------------------------------
        # 修改部分：实现分段与自适应移动 (X, Y 坐标控制)
        # ---------------------------------------------------------
        if cmd_name == 'move_to':
            target_x = cmd[1][0]
            target_y = cmd[1][1]

            # 1. 计算当前与目标的误差矢量
            error_x = target_x - current_x
            error_y = target_y - current_y

            # 2. 计算欧几里得距离 (总误差)
            distance = math.sqrt(error_x ** 2 + error_y ** 2)

            # 3. 自适应速度规划逻辑
            if distance < MOTION_PARAMS['dead_zone']:
                # 到达目标附近，停止移动
                step_size = 0.0
            elif distance > MOTION_PARAMS['slow_dist']:
                # 距离远：全速前进
                step_size = MOTION_PARAMS['max_step']
            else:
                # 距离近：线性减速 (P控制: Speed = dist * gain)
                # 计算比例：距离越近，速度越慢
                ratio = distance / MOTION_PARAMS['slow_dist']
                step_size = MOTION_PARAMS['max_step'] * ratio

                # 保证有一个最小速度，否则最后几毫米会永远走不到
                if step_size < MOTION_PARAMS['min_step']:
                    step_size = MOTION_PARAMS['min_step']

                # 防止超调：如果步长大于剩余距离，直接一步到位
                if step_size > distance:
                    step_size = distance

            # 4. 计算分量 (保持直线移动)
            if distance > 1e-6:  # 防止除以零
                delta_x = (error_x / distance) * step_size
                delta_y = (error_y / distance) * step_size
            else:
                delta_x = 0
                delta_y = 0

            # 更新当前坐标
            current_x += delta_x
            current_y += delta_y

            # Debug 日志 (可选，用于调试参数)
            # logging.debug(f"Dist: {distance:.4f}, Step: {step_size:.4f}, CurX: {current_x:.3f}, CurY: {current_y:.3f}")

        # ---------------------------------------------------------
        # 处理关节直接控制命令
        # ---------------------------------------------------------
        elif cmd_name in joint_controls:
            joint_command_list.append(cmd)
        else:
            wrist_command_list.append(cmd)

        # Pitch control
        if len(wrist_command_list) > 0:
            for key, value in wrist_command_list:
                if key == 'wrist_flex':
                    pitch = value
            set_pitch(pitch)

        # 处理关节更新
        if len(joint_command_list) > 0:
            for key, value in joint_command_list:
                joint_name = joint_controls[key]
                if joint_name in target_positions:
                    current_target = target_positions[joint_name]
                    new_target = int(current_target + value)
                    target_positions[joint_name] = new_target
                    logging.debug(f"Update target position {joint_name}: {current_target} -> {new_target}")

        # ---------------------------------------------------------
        # 更新逆运动学解 (只有在 X 或 Y 发生变化时才计算)
        # ---------------------------------------------------------
        if abs(delta_x) > 0 or abs(delta_y) > 0:
            # Calculate target angles for joint2 and joint3
            joint2_target, joint3_target = inverse_kinematics(current_x, current_y)
            target_positions['arm_shoulder_lift'] = joint2_target
            target_positions['arm_elbow_flex'] = joint3_target
            logging.debug(
                f"Update x coordinate: {current_x:.4f}, Update y coordinate: {current_y:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}")

        # Apply pitch adjustment to wrist_flex
        # Calculate wrist_flex target position based on shoulder_lift and elbow_flex
        if 'arm_shoulder_lift' in target_positions and 'arm_elbow_flex' in target_positions:
            target_positions['arm_wrist_flex'] = - target_positions['arm_shoulder_lift'] - target_positions[
                'arm_elbow_flex'] + pitch

        # Extract current joint positions
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                # Apply calibration coefficients
                calibrated_value = apply_joint_calibration(motor_name, value)
                current_positions[motor_name] = calibrated_value

        # P control calculation (关节层面的 P 控制)
        robot_action = {}
        for joint_name, target_pos in target_positions.items():
            if joint_name in current_positions:
                current_pos = current_positions[joint_name]
                error = target_pos - current_pos

                # P control: output = Kp * error
                control_output = kp * error

                # Convert control output to position command
                new_position = current_pos + control_output
                robot_action[f"{joint_name}.pos"] = new_position

        if robot_action:
            return robot_action, current_x, current_y

    except Exception as e:
        logging.error(f"P control loop error: {e}")
        traceback.print_exc()
        return {}, current_x, current_y


def return_to_start_position(robot, current_obs, start_positions, kp=0.5, control_freq=50):
    """
    Use P control to return to start position

    Args:
        robot: robot instance
        current_obs: start position
        start_positions: start joint position dictionary
        kp: proportional gain
        control_freq: control frequency (Hz)
    """
    print("Returning to start position...")

    control_period = 1.0 / control_freq
    max_steps = int(5.0 * control_freq)  # Maximum 5 seconds

    for step in range(max_steps):
        # Get current robot state
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                current_positions[motor_name] = value  # Don't apply calibration coefficients

        # P control calculation
        robot_action = {}
        total_error = 0
        for joint_name, target_pos in start_positions.items():
            if joint_name in current_positions:
                current_pos = current_positions[joint_name]
                error = target_pos - current_pos
                total_error += abs(error)

                # P control: output = Kp * error
                control_output = kp * error

                # Convert control output to position command
                new_position = current_pos + control_output
                robot_action[f"{joint_name}.pos"] = new_position

        base_action = {
            "x.vel": 0,
            "y.vel": 0,
            "theta.vel": 0,
        }
        # Send action to robot
        if robot_action:
            robot.send_action({**robot_action, **base_action})

        # Check if reached start position
        if total_error < 2.0:  # If total error is less than 2 degrees, consider reached
            print("Returned to start position")
            break

        time.sleep(control_period)

    print("Return to start position completed")


def apply_joint_calibration(joint_name, raw_position):
    """
    Apply joint calibration coefficients

    Args:
        joint_name: joint name
        raw_position: raw position value

    Returns:
        calibrated_position: calibrated position value
    """
    for joint_cal in JOINT_CALIBRATION:
        if joint_cal[0] == joint_name:
            offset = joint_cal[1]  # zero position offset
            scale = joint_cal[2]  # scale factor
            calibrated_position = (raw_position - offset) * scale
            return calibrated_position
    return raw_position  # if no calibration coefficient found, return original value


def inverse_kinematics(x, y, l1=0.1159, l2=0.1350):
    """
    Calculate inverse kinematics for a 2-link robotic arm, considering joint offsets

    Parameters:
        x: End effector x coordinate
        y: End effector y coordinate
        l1: Upper arm length (default 0.1159 m)
        l2: Lower arm length (default 0.1350 m)

    Returns:
        joint2, joint3: Joint angles in radians as defined in the URDF file
    """
    # Calculate joint2 and joint3 offsets in theta1 and theta2
    theta1_offset = math.atan2(0.028, 0.11257)  # theta1 offset when joint2=0
    theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset  # theta2 offset when joint3=0

    # Calculate distance from origin to target point
    r = math.sqrt(x ** 2 + y ** 2)
    r_max = l1 + l2  # Maximum reachable distance

    # If target point is beyond maximum workspace, scale it to the boundary
    if r > r_max:
        scale_factor = r_max / r
        x *= scale_factor
        y *= scale_factor
        r = r_max

    # If target point is less than minimum workspace (|l1-l2|), scale it
    r_min = abs(l1 - l2)
    if r < r_min and r > 0:
        scale_factor = r_min / r
        x *= scale_factor
        y *= scale_factor
        r = r_min

    # Use law of cosines to calculate theta2
    cos_theta2 = -(r ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)

    # Calculate theta2 (elbow angle)
    theta2 = math.pi - math.acos(cos_theta2)

    # Calculate theta1 (shoulder angle)
    beta = math.atan2(y, x)
    gamma = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    theta1 = beta + gamma

    # Convert theta1 and theta2 to joint2 and joint3 angles
    joint2 = theta1 + theta1_offset
    joint3 = theta2 + theta2_offset

    # Ensure angles are within URDF limits
    joint2 = max(-0.1, min(3.45, joint2))
    joint3 = max(-0.2, min(math.pi, joint3))

    # Convert from radians to degrees
    joint2_deg = math.degrees(joint2)
    joint3_deg = math.degrees(joint3)

    joint2_deg = 90 - joint2_deg
    joint3_deg = joint3_deg - 90

    return joint2_deg, joint3_deg


def move_to_zero_position(robot, duration=3.0, kp=0.5):
    """
    Use P control to slowly move robot to zero position

    Args:
        robot: robot instance
        duration: time to move to zero position (seconds)
        kp: proportional gain
    """
    print("Using P control to slowly move robot to zero position...")

    # Get current robot state
    current_obs = robot.get_observation()

    # Extract current joint positions
    current_positions = {}
    for key, value in current_obs.items():
        if key.endswith('.pos'):
            motor_name = key.removesuffix('.pos')
            current_positions[motor_name] = value

    # Zero position targets
    zero_positions = {
        'arm_shoulder_pan': 0.0,
        'arm_shoulder_lift': 0.0,
        'arm_elbow_flex': 0.0,
        'arm_wrist_flex': 0.0,
        'arm_wrist_roll': 0.0,
        'arm_gripper': 0.0
    }

    # Calculate control steps
    control_freq = 50  # 50Hz control frequency
    total_steps = int(duration * control_freq)
    step_time = 1.0 / control_freq

    print(
        f"Will use P control to move to zero position in {duration} seconds, control frequency: {control_freq}Hz, proportional gain: {kp}")

    for step in range(total_steps):
        # Get current robot state
        current_obs = robot.get_observation()
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                # Apply calibration coefficients
                calibrated_value = apply_joint_calibration(motor_name, value)
                current_positions[motor_name] = calibrated_value

        # P control calculation
        robot_action = {}
        for joint_name, target_pos in zero_positions.items():
            if joint_name in current_positions:
                current_pos = current_positions[joint_name]
                error = target_pos - current_pos

                # P control: output = Kp * error
                control_output = kp * error

                # Convert control output to position command
                new_position = current_pos + control_output
                robot_action[f"{joint_name}.pos"] = new_position
        base_action = {
            "x.vel": 0.0,
            "y.vel": 0.0,
            "theta.vel": 0.0,
        }
        # Send action to robot
        if robot_action:
            robot.send_action({**robot_action, **base_action})
        time.sleep(step_time)

    print("Robot has moved to zero position")
