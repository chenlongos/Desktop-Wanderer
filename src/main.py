import sys
import os

from src.arm_act_controller import arm_controller
from src.arm_inverse_controller import p_control_loop, return_to_start_position
from src.robot_setup import init_robot, get_robot, get_direction, reset_robot, get_target_positions
from src.setup import init_app, get_left, get_top, get_right, get_bottom, get_log_level, get_robot_status, \
    RobotStatus, get_control_mode, RobotControlModel, set_robot_status, get_hardware_mode
from src.utils import busy_wait
from src.move_controller import move_controller, get_empty_move_action

sys.path.append(os.path.dirname(__file__))
import time
import logging
import cv2

from src.yolov.process import yolo_infer

logger = logging.getLogger(__name__)
logging.basicConfig(level=getattr(logging, get_log_level()))

FPS = 50

CATCH_ACTION = [("shoulder_pan", -8), ("wrist_flex", 48), ("gripper", 60), ("move_to", (0.1089, -0.075)),
                ("gripper", -40),
                ("shoulder_pan", 8), ("move_to", (0.0, 0.13)), ("gripper", 50)]


# CATCH_ACTION = [("shoulder_pan", -8),("wrist_flex", 48), ("open_gripper", 50), ("move_to", (0.1089, -0.06))]


def main():
    init_app()
    init_robot()
    robot = get_robot()
    direction = get_direction()
    robot.connect()

    print("Reading initial joint angles...")
    start_obs = robot.get_observation()
    start_positions = {}
    for key, value in start_obs.items():
        if key.endswith('.pos'):
            motor_name = key.removesuffix('.pos')
            start_positions[motor_name] = int(value)  # Don't apply calibration coefficients

    print("Initial joint angles:")
    for joint_name, position in start_positions.items():
        print(f"  {joint_name}: {position}°")

    return_to_start_position(robot, get_target_positions(), 0.2, FPS)
    # x0, y0 = 0.1629, 0.1131
    x0, y0 = 0.0069, 0.0970
    current_x, current_y = x0, y0
    command_step = 0
    try:
        while True:
            t0 = time.perf_counter()

            obs = robot.get_observation()
            frame = obs["front"]
            result = yolo_infer(frame)

            if get_hardware_mode() == 'normal':
                for box in result:
                    x, y, w, h = box.x, box.y, box.w, box.h
                    center_x = x + w // 2
                    center_y = y + h // 2
                    pt1, pt2 = (x, y), (x + w, y + h)
                    cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
                    cv2.rectangle(frame, (get_left(), get_top()), (get_right(), get_bottom()), color=(255, 255, 0),
                                  thickness=2)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                cv2.imshow("frame", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            arm_action = {}
            move_action = get_empty_move_action(direction)

            if get_robot_status() == RobotStatus.CATCH:
                if get_control_mode() == RobotControlModel.ACT:
                    arm_action = arm_controller(robot)
                else:
                    arm_action, current_x, current_y = p_control_loop(robot, CATCH_ACTION[command_step],
                                                                      current_x,
                                                                      current_y, kp=0.5)
                    if CATCH_ACTION[command_step][0] == "move_to":
                        if abs(current_x - CATCH_ACTION[command_step][1][0]) < 0.005 and abs(
                                current_y - CATCH_ACTION[command_step][1][1]) < 0.005:
                            command_step += 1
                            if command_step == len(CATCH_ACTION):
                                set_robot_status(RobotStatus.SEARCH)
                                reset_robot()
                                command_step = 0
                    else:
                        command_step += 1
                        if command_step == len(CATCH_ACTION):
                            set_robot_status(RobotStatus.SEARCH)
                            reset_robot()
                            command_step = 0
            elif get_robot_status() == RobotStatus.SEARCH:
                move_action = move_controller(direction, result)

            _action_sent = robot.send_action({**arm_action, **move_action})
            busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    finally:
        robot.disconnect()


if __name__ == '__main__':
    main()
