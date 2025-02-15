from rock_env import RockEnv
import os
import pickle
import numpy as np

import torch
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

def get_cfgs():
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
        "kp": 20.0,
        "kd": 0.5,
        # termination
        "termination_if_roll_greater_than": 45,  # degree
        "termination_if_pitch_greater_than": 45,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "simulate_action_latency": False,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "go_fast": 1.0,
            "tracking_lin_vel": 0.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [2, 2],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


exp_name = "rock_test1"
timestamp = "2025-01-25_19-16-04"
checkpoint = 500

def main():
    