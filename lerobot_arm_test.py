from pathlib import Path
import time
from time import sleep

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.utils.robot_utils import busy_wait

from src.lekiwi import LeKiwiConfig
from src.lekiwi.lekiwi import LeKiwi

device = torch.device("cpu")  # or "cuda" or "cpu"
model_id = "src/policy/train/catch_ball_test/checkpoints/last/pretrained_model"
model = ACTPolicy.from_pretrained(model_id)

dataset_id = "src/policy/data"
path = Path(dataset_id)
dataset_metadata = LeRobotDatasetMetadata("", root=path)
preprocess, postprocess = make_pre_post_processors(model.config, dataset_stats=dataset_metadata.stats)

cfg = LeKiwiConfig(port="/dev/tty.usbmodem5A7C1220171")
robot = LeKiwi(cfg)
robot.connect()
fps = 30

while True:
    start_loop_t = time.perf_counter()

    obs = robot.get_observation()
    obs_frame = build_inference_frame(
        observation=obs, ds_features=dataset_metadata.features, device=device
    )

    obs = preprocess(obs_frame)

    action = model.select_action(obs)
    action = postprocess(action)

    action = make_robot_action(action, dataset_metadata.features)
    move_action = {
        "x.vel": 0.0,
        "y.vel": 0.0,
        "theta.vel": 0.0,
    }

    robot.send_action({**action, **move_action})

    dt_s = time.perf_counter() - start_loop_t
    busy_wait(1 / fps - dt_s)
