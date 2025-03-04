import argparse
import os
import pickle
import numpy as np

import torch
from rock_env import RockEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


import json





def rock_eval(run_name:str, env_cfg=None, checkpoint=-1, show_viewer=False, do_record=True, do_log=False):
    gs.init(logging_level='warning')
    # gs.init(logging_level='info')
    torch.no_grad()

    log_dir = f"{RockEnv.SIM_DIR}/runs/{run_name}"

    if env_cfg is None:
        with open(f"{log_dir}/env_cfg.json", "r") as f:
            env_cfg = json.load(f)

    with open(f"{log_dir}/train_cfg.json", "r") as f:
        train_cfg = json.load(f)


    env = RockEnv(
        num_envs=1,
        env_cfg=env_cfg,
        add_camera=do_record,
        show_viewer=show_viewer,
    )

    if checkpoint == -1:
        # Find highest checkpoint number
        model_files = [f for f in os.listdir(f"{log_dir}/models") if f.startswith("model_") and f.endswith(".pt")]
        if not model_files:
            raise FileNotFoundError(f"No model checkpoint files found in {log_dir}/models")
            
        checkpoint_numbers = [int(f.split("_")[1].split(".")[0]) for f in model_files]
        checkpoint = max(checkpoint_numbers)

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=env.device)
    runner.load(f"{log_dir}/models/model_{checkpoint}.pt")
    policy = runner.get_inference_policy(device=env.device)
    
    
    obs, _ = env.reset()

    if do_record:
        env.cam.start_recording()

    for i in range(1000):
        actions = policy(obs)
        # actions = torch.ones((1,1))
        obs, _, rews, dones, infos = env.step(actions)

        if do_record:
            if i % 10 == 0:
                print(i)
            env.cam.render()

    env.cam.stop_recording(f"{log_dir}/eval_ckpt{checkpoint}.mp4", fps=30)
            


if __name__ == "__main__":

    exp_name = "balo1_2025-03-01_18-26-20"
    
    rock_eval(exp_name, checkpoint=7250)