import argparse
import os
import pickle
import numpy as np

import torch
from rock_env import RockEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


import json

with open("data.json", "w") as f:
    json.dump(all_cfg, f, indent=4)  # indent=4 makes it more readable




def rock_eval(log_dir:str, env_cfg_file=None, checkpoint=-1, show_viewer=True, do_record=True, do_log=True):
    # gs.init(logging_level='warning')
    gs.init(logging_level='info')


    env = RockEnv(
        num_envs=1,
        show_viewer=show_viewer,
    )

    # runner = OnPolicyRunner(env, train_cfg, log_dir, device=env.device)
    # runner.load(f"{log_dir}/model_{checkpoint}.pt")
    # policy = runner.get_inference_policy(device=env.device)
    
    
    obs, _ = env.reset()
    with torch.no_grad():

        while True:
            # actions = policy(obs)
            actions = torch.zeros((1,1))
            obs, _, rews, dones, infos = env.step(actions)



if __name__ == "__main__":

    exp_name = "go2-walking_2025-01-25_19-16-04"
    
    rock_eval(exp_name)