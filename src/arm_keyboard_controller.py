import math
import time
import traceback

JOINT_CALIBRATION = [
    ['arm_shoulder_pan', 6.0, 1.0],  # Joint 1: zero position offset, scale factor
    ['arm_shoulder_lift', 2.0, 0.97],  # Joint 2: zero position offset, scale factor
    ['arm_elbow_flex', 0.0, 1.05],  # Joint 3: zero position offset, scale factor
    ['arm_wrist_flex', 0.0, 0.94],  # Joint 4: zero position offset, scale factor
    ['arm_wrist_roll', 0.0, 0.5],  # Joint 5: zero position offset, scale factor
    ['arm_gripper', 0.0, 1.0],  # Joint 6: zero position offset, scale factor
]

target_positions = {
    'arm_shoulder_pan': 0.0,
    'arm_shoulder_lift': 0.0,
    'arm_elbow_flex': 0.0,
    'arm_wrist_flex': 0.0,
    'arm_wrist_roll': 0.0,
    'arm_gripper': 0.0
}

# Joint control mapping
joint_controls = {
    'q': ('arm_shoulder_pan', -1),  # Joint 1 decrease
    'a': ('arm_shoulder_pan', 1),  # Joint 1 increase
    't': ('arm_wrist_roll', -1),  # Joint 5 decrease
    'g': ('arm_wrist_roll', 1),  # Joint 5 increase
    'y': ('arm_gripper', -1),  # Joint 6 decrease
    'h': ('arm_gripper', 1),  # Joint 6 increase
}

# x,y coordinate control
xy_controls = {
    'w': ('x', -0.004),  # x decrease
    's': ('x', 0.004),  # x increase
    'e': ('y', -0.004),  # y decrease
    'd': ('y', 0.004),  # y increase
}

pitch = 0.0  # Initial pitch adjustment

def p_control_loop(robot, cmd, current_x, current_y, kp=0.5, control_freq=50):
    global target_positions, pitch
    """
    P control loop

    Args:
        robot: robot instance
        target_positions: target joint position dictionary
        start_positions: start joint position dictionary
        current_x: current x coordinate
        current_y: current y coordinate
        kp: proportional gain
        control_freq: control frequency (Hz)
    """

    # Initialize pitch control variables
    pitch_step = 1  # Pitch adjustment step size

    move_command_list = []
    joint_command_list = []
    wrist_command_list = []

    try:
        cmd_name = cmd[0]
        if cmd_name == 'move_to':
            cmd_x = cmd[1][0]
            cmd_y = cmd[1][1]

            if cmd_x > current_x:
                key = 's'
                move_command_list.append(key)
            if cmd_x < current_x:
                key = 'w'
                move_command_list.append(key)
            if cmd_y > current_y:
                key = 'd'
                move_command_list.append(key)
            if cmd_y < current_y:
                key = 'e'
                move_command_list.append(key)
        elif cmd_name == 'open_gripper':
            step = cmd[1]
            key = 'h'
            for _ in range(step):
                joint_command_list.append(key)
        elif cmd_name == 'close_gripper':
            step = cmd[1]
            key = 'y'
            for _ in range(step):
                joint_command_list.append(key)
        elif cmd_name == 'wrist_flex':
            step = cmd[1]
            if step > 0:
                key = 'r'
            else:
                key = 'f'
            for _ in range(abs(step)):
                wrist_command_list.append(key)
        elif cmd_name == 'shoulder_pan':
            step = cmd[1]
            if step > 0:
                key = 'a'
            else:
                key = 'q'
            for _ in range(abs(step)):
                joint_command_list.append(key)

        # Pitch control
        if len(wrist_command_list) > 0:
            for key in wrist_command_list:
                if key == 'r':
                    pitch += pitch_step
                    print(f"Increase pitch adjustment: {pitch:.3f}")
                elif key == 'f':
                    pitch -= pitch_step
                    print(f"Decrease pitch adjustment: {pitch:.3f}")

        if len(joint_command_list) > 0:
            for key in joint_command_list:
                joint_name, delta = joint_controls[key]
                if joint_name in target_positions:
                    current_target = target_positions[joint_name]
                    new_target = int(current_target + delta)
                    target_positions[joint_name] = new_target
                    print(f"Update target position {joint_name}: {current_target} -> {new_target}")

        if len(move_command_list) > 0:
            for key in move_command_list:
                coord, delta = xy_controls[key]
                if coord == 'x':
                    current_x += delta
                elif coord == 'y':
                    current_y += delta

            # Calculate target angles for joint2 and joint3
            joint2_target, joint3_target = inverse_kinematics(current_x, current_y)
            target_positions['arm_shoulder_lift'] = joint2_target
            target_positions['arm_elbow_flex'] = joint3_target
            print(
                f"Update x coordinate: {current_x:.4f}, Update y coordinate: {current_y:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}")


        # Apply pitch adjustment to wrist_flex
        # Calculate wrist_flex target position based on shoulder_lift and elbow_flex
        if 'arm_shoulder_lift' in target_positions and 'arm_elbow_flex' in target_positions:
            target_positions['arm_wrist_flex'] = - target_positions['arm_shoulder_lift'] - target_positions[
                'arm_elbow_flex'] + pitch
            # Show current pitch value (display every 100 steps to avoid screen flooding)
        if hasattr(p_control_loop, 'step_counter'):
            p_control_loop.step_counter += 1
        else:
            p_control_loop.step_counter = 0
        if p_control_loop.step_counter % 100 == 0:
            print(
                f"Current pitch adjustment: {pitch:.3f}, wrist_flex target: {target_positions['arm_wrist_flex']:.3f}")

        # Get current robot state
        current_obs = robot.get_observation()

        # Extract current joint positions
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                # Apply calibration coefficients
                calibrated_value = apply_joint_calibration(motor_name, value)
                current_positions[motor_name] = calibrated_value

        # P control calculation
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
        print(f"P control loop error: {e}")
        traceback.print_exc()


def return_to_start_position(robot, start_positions, kp=0.5, control_freq=50):
    """
    Use P control to return to start position

    Args:
        robot: robot instance
        start_positions: start joint position dictionary
        kp: proportional gain
        control_freq: control frequency (Hz)
    """
    print("Returning to start position...")

    control_period = 1.0 / control_freq
    max_steps = int(5.0 * control_freq)  # Maximum 5 seconds

    for step in range(max_steps):
        # Get current robot state
        current_obs = robot.get_observation()
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
    global target_positions
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
