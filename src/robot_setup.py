from src.lekiwi import LeKiwiConfig, DirectionControl
from src.lekiwi.lekiwi import LeKiwi
from src.setup import get_port

_start_positions = {
    'arm_shoulder_pan': 0.0,
    'arm_shoulder_lift': -31.70,
    'arm_elbow_flex': 27.69,
    'arm_wrist_flex': 88.00,
    'arm_wrist_roll': 0.0,
    'arm_gripper': 0.0
}

_is_robot_initialized = False
_robot = None
_direction = None

_target_positions = {
    'arm_shoulder_pan': 0.0,
    'arm_shoulder_lift': -31.70,
    'arm_elbow_flex': 27.69,
    'arm_wrist_flex': 88.00,
    'arm_wrist_roll': 0.0,
    'arm_gripper': 0.0
}
_pitch = 0.0


def init_robot():
    global _is_robot_initialized, _robot, _direction, _target_positions, _pitch
    if _is_robot_initialized:
        return
    cfg = LeKiwiConfig(port=get_port())
    _robot = LeKiwi(cfg)
    _direction = DirectionControl()

    _is_robot_initialized = True


def get_robot():
    if not _is_robot_initialized:
        init_robot()
    return _robot


def get_direction():
    if not _is_robot_initialized:
        init_robot()
    return _direction


def get_target_positions():
    if not _is_robot_initialized:
        init_robot()
    return _target_positions


def reset_target_positions():
    global _target_positions
    if not _is_robot_initialized:
        init_robot()
    _target_positions = {
        'arm_shoulder_pan': 0.0,
        'arm_shoulder_lift': -31.70,
        'arm_elbow_flex': 27.69,
        'arm_wrist_flex': 82.00,
        'arm_wrist_roll': 0.0,
        'arm_gripper': 0.0
    }

def get_pitch():
    if not _is_robot_initialized:
        init_robot()
    return _pitch

def set_pitch(pitch):
    global _pitch
    if not _is_robot_initialized:
        init_robot()
    _pitch = pitch

def reset_pitch():
    global _pitch
    if not _is_robot_initialized:
        init_robot()
    _pitch = 0.0

def reset_robot():
    global _is_robot_initialized, _target_positions, _pitch
    if not _is_robot_initialized:
        init_robot()
    reset_pitch()
    reset_target_positions()