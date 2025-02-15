import argparse
import os
import pickle
import numpy as np

import torch
from rock_env import RockEnv
from rsl_rl.runners import OnPolicyRunner

print("before genesis import")

import genesis as gs
print("sdhsdhskdhsd")

exp_name = "go2-walking"
timestamp = "2025-01-25_19-16-04"
checkpoint = 500

env_cfg = {
    "num_actions": 1, # angle of pendulum
    # joint/link names
    "default_joint_angles": {  # [rad]
        "Revolute_1": 0.0,
    },
    "dof_names": [
        "Revolute_1",
    ],
    # PD
    # motor voltage = kp*(desired - current) + kd*(0 - current_vel)
    "kp": 100,
    "kd": 100,
    # termination
    "termination_if_roll_greater_than": 80,  # degree
    "termination_if_pitch_greater_than": 80,
    # base pose
    "base_init_pos": [0.0, 0.0, 0.01],
    "base_init_quat": [1.0, 0.0, 0.0, 0.0],
    "episode_length_s": 20.0,
    "simulate_action_latency": False,
    "clip_actions": 100.0,
}
obs_cfg = {
    "num_obs": 3,
    "obs_scales": {
        "lin_vel": 2.0,
    },
}
reward_cfg = {
    "reward_scales": {
        "lin_vel": 2.0,
    },

}
command_cfg = {
    "num_commands": 1,

}


def main():
    # gs.init(logging_level='warning')
    gs.init(logging_level='info')

    print("sdhsdhskdhsd")

    env = RockEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )
    print("sdhs2222dhskdhsd")

    
    
    obs, _ = env.reset()
    with torch.no_grad():

        print("sdhsdhskdhs3232323d")

        
        while True:
            print("sdhsd543545hskdhsd")
            actions = torch.zeros((1, 1))
            obs, _, rews, dones, infos = env.step(actions)

if __name__ == "__main__":
    main()