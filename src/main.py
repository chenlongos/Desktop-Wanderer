from operator import contains
import os
import sys

from src.arm_act_controller import arm_controller
from src.arm_inverse_controller import p_control_loop, return_to_start_position
from src.move_controller import CENTER_GRAB_TOLERANCE_PX, GRAB_GOAL_CX, move_controller, get_empty_move_action, move_controller_for_bucket, \
    CENTER_FIND_TOLERANCE_PX, CENTER_SLOWDOWN_PX, FIND_GOAL_CX
from src.robot_setup import init_robot, get_robot, get_direction, reset_robot, get_target_positions
from src.setup import init_app, get_left, get_top, get_right, get_bottom, get_log_level, get_robot_status, \
    RobotStatus, get_control_mode, RobotControlModel, set_robot_status, get_fps
from src.utils import busy_wait
from src.yolov import yolo_infer, get_black_bucket_local, get_red_bucket_local
from src.stream_server import start_stream_server, update_frame, is_running, is_quit

sys.path.append(os.path.dirname(__file__))
import time
import logging
import cv2


logger = logging.getLogger(__name__)
logging.basicConfig(level=getattr(logging, get_log_level()))

# 夹球的动作序列
CATCH_ACTION = [
                [
                    # ("move_to", (0.0989, 0.125)),
                    ("shoulder_pan_abs", -12), # 对应1号舵机
                    ("gripper_abs", 70), # 夹爪打开
                    ("wrist_flex", 95),  # 腕部舵机转动角度q
                    # ("move_to", (0.140, 0.1211)), # 机械臂坐标移动指令，x移动到范围为 0.22 - -0.22
                    ("move_to", (0.140, -0.060)), # 机械臂移动到球的位置， y移动范围为 0.22 - -0.15
                ],
                ("gap", 0), # 停顿指令
                ("gripper", -60), # 夹爪关闭
                ("gap", 0), # 停顿指令
                [
                    ("shoulder_pan_abs", 0), # 1号舵机归位
                    ("move_to", (-0.1, 0.2)), # 把球举起
                    ("wrist_flex", -20)   # 腕部配合移动
                ]
                ]

PUT_ACTION = [
    ("shoulder_lift", 50),
    ("gap", 0),  # 停顿指令
    ("gripper", 60),
    ("gap", 0), # 停顿指令
    [("gripper_abs", 10), ("move_to", (-0.1, 0.2))],
]

def main():
    init_app()
    init_robot()
    robot = get_robot()
    direction = get_direction()
    robot.connect()
    start_stream_server()

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

    return_to_start_position(robot, start_obs, get_target_positions(), 0.2, get_fps())  # 机械臂回到预设位置
    x0, y0 = 0.0989, 0.125 # 当前位置的xy坐标
    current_x, current_y = x0, y0
    command_step = 0

    # 通过网页按钮控制：按住运行，松开停止
    print("Open the control panel in your browser to start.")
    was_running = False
    try:
        while not is_quit():
            t0 = time.perf_counter()

            # 始终读取摄像头并推流
            current_obs = robot.get_observation()
            frame = current_obs["front"]

            # no_move = not is_running()
            no_move = False
            if no_move:
                # 未按下按钮时，只推流不执行动作
                if was_running:
                    robot.send_action(get_empty_move_action(direction))
                    was_running = False
                update_frame(frame)
                busy_wait(max(1.0 / get_fps() - (time.perf_counter() - t0), 0.0))
                continue

            was_running = True

            if get_robot_status() == RobotStatus.FIND_BUCKET:
                gripper_pos = current_obs.get('arm_gripper.pos', 5)
                is_gripper_holding = gripper_pos > 25

                if not is_gripper_holding:
                    set_robot_status(RobotStatus.SEARCH)
                    reset_robot()
                    continue

                # result = get_black_bucket_local(frame) # 找桶的算法
                # if len(result) == 0:
                result = get_red_bucket_local(frame)
            elif get_robot_status() == RobotStatus.SEARCH:
                result = yolo_infer(frame) # 找球的算法

            # 摄像头视角显示，通过MJPEG流
            for box in result:
                x, y, w, h = box.x, box.y, box.w, box.h
                center_x = x + w // 2
                center_y = y + h // 2
                pt1, pt2 = (x, y), (x + w, y + h)
                cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
                # CENTER_TOLERANCE_PX 线（绿色，内侧）
                cv2.line(frame, (FIND_GOAL_CX - CENTER_FIND_TOLERANCE_PX, 0), (FIND_GOAL_CX - CENTER_FIND_TOLERANCE_PX, 480), (0, 255, 0), 1)
                cv2.line(frame, (FIND_GOAL_CX + CENTER_FIND_TOLERANCE_PX, 0), (FIND_GOAL_CX + CENTER_FIND_TOLERANCE_PX, 480), (0, 255, 0), 1)
                # CENTER_SLOWDOWN_PX 线（黄色，外侧）
                cv2.line(frame, (FIND_GOAL_CX - CENTER_SLOWDOWN_PX, 0), (FIND_GOAL_CX - CENTER_SLOWDOWN_PX, 480), (0, 255, 255), 1)
                cv2.line(frame, (FIND_GOAL_CX + CENTER_SLOWDOWN_PX, 0), (FIND_GOAL_CX + CENTER_SLOWDOWN_PX, 480), (0, 255, 255), 1)

                cv2.line(frame, (GRAB_GOAL_CX - CENTER_GRAB_TOLERANCE_PX, 0), (GRAB_GOAL_CX - CENTER_GRAB_TOLERANCE_PX, 480), (255, 0, 255), 1)
                cv2.line(frame, (GRAB_GOAL_CX + CENTER_GRAB_TOLERANCE_PX, 0), (GRAB_GOAL_CX + CENTER_GRAB_TOLERANCE_PX, 480), (255, 0, 255), 1)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                diameter_px = max(w, h)
                distance_cm = 2892.91 / diameter_px + 0.27
                cv2.putText(frame, f"{diameter_px}px {distance_cm:.1f}cm", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            update_frame(frame)

            arm_action = {}
            move_action = get_empty_move_action(direction)

            if get_robot_status() == RobotStatus.PICK:
                if get_control_mode() == RobotControlModel.ACT:
                    arm_action = arm_controller(robot)
                else:
                    arm_action, current_x, current_y = p_control_loop(CATCH_ACTION[command_step],
                                                                      current_x,
                                                                      current_y, current_obs, kp=0.8)
                    step = CATCH_ACTION[command_step]
                    # 判断当前步骤是否完成
                    has_move_to = False
                    if isinstance(step, list):
                        # 列表命令：检查最后一个 move_to 是否到达
                        for c in step:
                            if c[0] == "move_to":
                                has_move_to = True
                                move_target = c[1]
                        if has_move_to:
                            if abs(current_x - move_target[0]) < 0.002 and abs(current_y - move_target[1]) < 0.002:
                                command_step += 1
                        else:
                            command_step += 1
                    elif step[0] == "move_to":
                        has_move_to = True
                        if abs(current_x - step[1][0]) < 0.002 and abs(current_y - step[1][1]) < 0.002:
                            command_step += 1
                    else:
                        command_step += 1
                        
                    if command_step == len(CATCH_ACTION):
                        set_robot_status(RobotStatus.FIND_BUCKET)
                        command_step = 0
            elif get_robot_status() == RobotStatus.PUT_BALL:
                arm_action, current_x, current_y = p_control_loop(PUT_ACTION[command_step],
                                                                  current_x,
                                                                  current_y, current_obs, kp=0.8)
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
                move_action = move_controller(direction, result, frame)
            elif get_robot_status() == RobotStatus.FIND_BUCKET:
                move_action = move_controller_for_bucket(direction, result)

            robot.send_action({**arm_action, **move_action})
            if get_robot_status() != RobotStatus.PICK:
                busy_wait(max(1.0 / get_fps() - (time.perf_counter() - t0), 0.0))
    finally:
        robot.disconnect()


if __name__ == '__main__':
    main()
