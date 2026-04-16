import time
from src.robot_setup import get_robot
from src.arm_inverse_controller import p_control_loop, get_target_positions, return_to_start_position
from src.setup import get_fps

def print_current_positions(observation):
    print("当前机械臂关节位置:")
    for key, value in observation.items():
        if key.endswith('.pos'):
            motor_name = key.removesuffix('.pos')
            print(f"  {motor_name}: {int(value)}°")
def manual_control_loop():
    """
    手动控制循环，允许用户输入指定动作并让机械臂执行
    动作名直接定义在all_unique_actions中，参数由用户输入
    """
    robot = get_robot()
    start_obs = robot.get_observation()
    print_current_positions(start_obs)

    return_to_start_position(robot, start_obs, get_target_positions(), 0.9, get_fps())
    print_current_positions(robot.get_observation())

    
    # 直接定义所有可用的动作类型
    all_unique_actions = [
        "shoulder_pan", 
        "shoulder_lift",
        "gap",
        "wrist_flex",
        "wrist_roll",
        "gripper",
        "move_to"
    ]
    
    # 获取当前坐标
    current_obs = robot.get_observation()
    current_x, current_y = 0.0989, 0.125  # 初始坐标
    
    print("可用的动作列表:")
    for i, action in enumerate(all_unique_actions):
        print(f"{i+1}. {action}")
    print("输入 'quit' 退出程序")
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n请输入要执行的动作编号或名称: ").strip()
            
            if user_input.lower() == 'quit':
                print("退出手动控制模式")
                break
            
            # 解析用户输入
            selected_action = None
            
            # 检查是否输入的是编号
            if user_input.isdigit():
                idx = int(user_input) - 1
                if 0 <= idx < len(all_unique_actions):
                    selected_action = all_unique_actions[idx]
            else:
                # 检查是否输入的是动作名称
                if user_input in all_unique_actions:
                    selected_action = user_input
            
            if selected_action is None:
                print(f"无效的输入: {user_input}")
                print("请从以下选项中选择:")
                for i, action in enumerate(all_unique_actions):
                    print(f"{i+1}. {action}")
                continue
            
            # 根据动作类型获取用户输入的参数
            action_detail = None
            
            if selected_action == "move_to":
                # 要求用户输入x和y坐标
                try:
                    x_input = float(input("请输入x坐标 (例如: 0.140): "))
                    y_input = float(input("请输入y坐标 (例如: -0.05): "))
                    action_detail = (selected_action, (x_input, y_input))
                except ValueError:
                    print("输入的坐标不是有效数字，请重新输入")
                    continue
                    
            elif selected_action in ["shoulder_pan", "shoulder_lift", "wrist_flex", "gripper", "wrist_roll"]:
                # 要求用户输入关节角度值
                try:
                    value_input = float(input(f"请输入{selected_action}的值 (例如: 60): "))
                    action_detail = (selected_action, value_input)
                except ValueError:
                    print("输入的角度值不是有效数字，请重新输入")
                    continue
                    
            elif selected_action == "gap":
                # gap动作不需要额外参数，固定为0
                action_detail = (selected_action, 0)
            
            if action_detail:
                print(f"执行动作: {action_detail}")
                
                # 根据动作类型执行相应的操作
                if action_detail[0] == "move_to":
                    # 执行移动动作
                    target_x, target_y = action_detail[1]
                    print(f"移动到坐标: ({target_x}, {target_y})")
                    
                    # 循环执行直到达到目标位置
                    while True:
                        # 使用p_control_loop执行移动
                        robot_action, current_x, current_y = p_control_loop(
                            action_detail, 
                            current_x, 
                            current_y, 
                            robot.get_observation(), 
                            kp=0.8
                        )
                        
                        # 发送动作到机器人
                        base_action = {
                            "x.vel": 0.0,
                            "y.vel": 0.0,
                            "theta.vel": 0.0,
                        }
                        robot.send_action({**robot_action, **base_action})
                        
                        # 检查是否达到目标位置
                        if abs(current_x - target_x) < 0.002 and abs(current_y - target_y) < 0.002:
                            print("到达目标位置")
                            break
                        
                        # 短暂停顿，控制执行频率
                        time.sleep(1.0 / get_fps())
                    
                elif action_detail[0] in ["shoulder_pan", "shoulder_lift", "wrist_flex", "gripper", "wrist_roll"]:
                    # 执行关节控制动作
                    joint_value = action_detail[1]
                    print(f"执行关节控制: {action_detail[0]} 设置为 {joint_value}")
                    
                    # 创建关节控制动作
                    robot_action, current_x, current_y = p_control_loop(
                        action_detail, 
                        current_x, 
                        current_y, 
                        robot.get_observation(), 
                        kp=0.8
                    )
                    
                    # 发送动作到机器人
                    base_action = {
                        "x.vel": 0.0,
                        "y.vel": 0.0,
                        "theta.vel": 0.0,
                    }
                    robot.send_action({**robot_action, **base_action})
                    
                elif action_detail[0] == "gap":
                    # 执行停顿动作
                    print("执行停顿...")
                    time.sleep(0.3)
                
                print(f"动作 '{action_detail[0]}' 执行完成")
            else:
                print(f"无法构建动作 '{selected_action}' 的详细信息")
        
        except KeyboardInterrupt:
            print("\n程序被用户中断")
            break
        except Exception as e:
            print(f"执行动作时发生错误: {e}")
            continue

if __name__ == "__main__":
    # 初始化机器人连接
    from src.robot_setup import init_robot
    from src.setup import init_app
    init_app()
    init_robot()
    
    robot = get_robot()
    robot.connect()
    
    try:
        manual_control_loop()
    finally:
        robot.disconnect()