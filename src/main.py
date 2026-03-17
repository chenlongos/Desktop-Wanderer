import os
import sys

from src.arm_act_controller import arm_controller
from src.arm_inverse_controller import p_control_loop, return_to_start_position
from src.move_controller import move_controller, get_empty_move_action, move_controller_for_bucket
from src.robot_setup import init_robot, get_robot, get_direction, reset_robot, get_target_positions
from src.setup import init_app, get_left, get_top, get_right, get_bottom, get_log_level, get_robot_status, \
    RobotStatus, get_control_mode, RobotControlModel, set_robot_status, get_hardware_mode, get_fps
from src.utils import busy_wait
from src.yolov import yolo_infer, get_black_bucket_local

sys.path.append(os.path.dirname(__file__))
import time
import logging
import cv2


logger = logging.getLogger(__name__)
logging.basicConfig(level=getattr(logging, get_log_level()))

# 夹球的动作序列
CATCH_ACTION = [
                ("move_to", (0.0989, 0.125)),
                ("shoulder_pan", -12), # 对应1号舵机
                ("gripper", 60), # 夹爪打开
                ("wrist_flex", 80),  # 腕部舵机转动角度q
                ("move_to", (0.17, 0.1211)), # 机械臂坐标移动指令，x移动到范围为 0.22 - -0.22
                ("move_to", (0.17, -0.02)), # 机械臂移动到球的位置， y移动范围为 0.22 - -0.15
                ("gap", 0), # 停顿指令
                ("gripper", -60), # 夹爪关闭
                ("gap", 0), # 停顿指令
                ("shoulder_pan", 12), # 1号舵机归位
                ("move_to", (0.00, 0.2)), # 把球举起
                ("wrist_flex", -20)   # 腕部配合移动
                ]

PUT_ACTION = [
    ("move_to", (0.10, 0.2)), # 向前伸
    ("gripper", 60),
    ("gap", 0), # 停顿指令
    ("move_to", (0.00, 0.2)), # 回收
    ("wrist_flex", 80)
]

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
            start_positions[motor_name] = int(value)

    print("Initial joint angles:")
    for joint_name, position in start_positions.items():
        print(f"  {joint_name}: {position}°")

    return_to_start_position(robot, start_obs, get_target_positions(), 0.9, get_fps()) # 机械臂回到预设位置
    x0, y0 = 0.0989, 0.125 # 当前位置的xy坐标
    current_x, current_y = x0, y0
    command_step = 0
    try:
        while True:
            t0 = time.perf_counter()

            current_obs = robot.get_observation()
            frame = current_obs["front"]
            if get_robot_status() == RobotStatus.FIND_BUCKET:
                gripper_pos = current_obs.get('arm_gripper.pos', 5)
                is_gripper_holding = gripper_pos > 25

                if not is_gripper_holding:
                    set_robot_status(RobotStatus.SEARCH)
                    reset_robot()
                    continue

                result = get_black_bucket_local(frame) # 找桶的算法
            elif get_robot_status() == RobotStatus.SEARCH:
                result = yolo_infer(frame) # 找球的算法

            if get_hardware_mode() == 'normal': # 摄像头视角显示，
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

            if get_robot_status() == RobotStatus.PICK:
                if get_control_mode() == RobotControlModel.ACT:
                    arm_action = arm_controller(robot)
                else:
                    arm_action, current_x, current_y = p_control_loop(CATCH_ACTION[command_step],
                                                                      current_x,
                                                                      current_y, current_obs, kp=0.8)
                    if CATCH_ACTION[command_step][0] == "move_to":
                        if abs(current_x - CATCH_ACTION[command_step][1][0]) < 0.002 and abs(
                                current_y - CATCH_ACTION[command_step][1][1]) < 0.002:
                            command_step += 1
                            if command_step == len(CATCH_ACTION):
                                set_robot_status(RobotStatus.FIND_BUCKET)
                                command_step = 0
                    else:
                        command_step += 1
                        if command_step == len(CATCH_ACTION):
                            set_robot_status(RobotStatus.FIND_BUCKET)
                            command_step = 0
            elif get_robot_status() == RobotStatus.PUT_BALL:
                arm_action, current_x, current_y = p_control_loop(PUT_ACTION[command_step],
                                                                  current_x,
                                                                  current_y, current_obs, kp=0.9)
                if PUT_ACTION[command_step][0] == "move_to":
                    if abs(current_x - PUT_ACTION[command_step][1][0]) < 0.002 and abs(
                            current_y - PUT_ACTION[command_step][1][1]) < 0.002:
                        command_step += 1
                        if command_step == len(PUT_ACTION):
                            set_robot_status(RobotStatus.SEARCH)
                            reset_robot()
                            command_step = 0
                else:
                    command_step += 1
                    if command_step == len(PUT_ACTION):
                        set_robot_status(RobotStatus.SEARCH)
                        reset_robot()
                        command_step = 0
            elif get_robot_status() == RobotStatus.SEARCH:
                move_action = move_controller(direction, result)
            elif get_robot_status() == RobotStatus.FIND_BUCKET:
                move_action = move_controller_for_bucket(direction, result)

            _action_sent = robot.send_action({**arm_action, **move_action})
            if get_robot_status() != RobotStatus.PICK:
                busy_wait(max(1.0 / get_fps() - (time.perf_counter() - t0), 0.0))
    finally:
        robot.disconnect()


if __name__ == '__main__':
    main()
