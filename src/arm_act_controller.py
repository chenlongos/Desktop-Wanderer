from pathlib import Path

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action

from src.lekiwi.lekiwi import LeKiwi
from src.state import get_control_mode, RobotControlModel

if get_control_mode() == RobotControlModel.ACT:
    device = torch.device("mps")
    model_id = "src/policy/train/catch_ball_test/checkpoints/last/pretrained_model"
    model = ACTPolicy.from_pretrained(model_id)

    dataset_id = "src/policy/data"
    path = Path(dataset_id)
    dataset_metadata = LeRobotDatasetMetadata("", root=path)
    preprocess, postprocess = make_pre_post_processors(model.config, dataset_stats=dataset_metadata.stats)

def arm_controller(robot: LeKiwi):
    obs = robot.get_observation()
    obs_frame = build_inference_frame(
        observation=obs, ds_features=dataset_metadata.features, device=device
    )

    obs = preprocess(obs_frame)

    action = model.select_action(obs)
    action = postprocess(action)

    action = make_robot_action(action, dataset_metadata.features)
    return action
