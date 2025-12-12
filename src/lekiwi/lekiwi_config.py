from lerobot.cameras import CameraConfig, Cv2Rotation, ColorMode
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots import RobotConfig
from dataclasses import dataclass, field

def lekiwi_cameras_config() -> dict[str, CameraConfig]:
    return {
        "front": OpenCVCameraConfig(
            index_or_path=0, fps=30, width=640, height=480, color_mode=ColorMode.BGR
        ),
    }

@RobotConfig.register_subclass("lekiwi")
@dataclass
class LeKiwiConfig(RobotConfig):
    port: str = "/dev/ttyACM0"  # port to connect to the bus

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=lekiwi_cameras_config)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False
