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

    log_dir = f"{RockEnv.SIM_DIR}/penaltysweep/{run_name}"

    if env_cfg is None:
        with open(f"{log_dir}/env_cfg.json", "r") as f:
            env_cfg = json.load(f)

    with open(f"{log_dir}/train_cfg.json", "r") as f:
        train_cfg = json.load(f)

    env_cfg['resampling_time_s'] = 3
    env_cfg['episode_length_s'] = 999


    env = RockEnv(
        num_envs=1,
        env_cfg=env_cfg,
        add_camera=do_record,
        show_viewer=show_viewer,
    )
    # env.get_robot().set_friction(1e-2)
    # env.get_robot().set_mass_shift(
    #     mass_shift = 0.1 * torch.rand(env.scene.n_envs, env.get_robot().n_links),
    #     link_indices=np.arange(0, env.get_robot().n_links),
    # )
    # env.get_robot().set_COM_shift(
    #     com_shift = 0.1 * torch.rand(env.scene.n_envs, env.get_robot().n_links, 3),
    #     link_indices=np.arange(0, env.get_robot().n_links),
    # )


    if checkpoint == -1:
        # Find highest checkpoint number
        model_files = [f for f in os.listdir(f"{log_dir}/models") if f.startswith("model_") and f.endswith(".pt")]
        if not model_files:
            raise FileNotFoundError(f"No model checkpoint files found in {log_dir}/models")
            
        checkpoint_numbers = [int(f.split("_")[1].split(".")[0]) for f in model_files]
        checkpoint = max(checkpoint_numbers)

    output_filename = f"{log_dir}/eval_ckpt{checkpoint}"


    runner = OnPolicyRunner(env, train_cfg, log_dir, device=env.device)
    runner.load(f"{log_dir}/models/model_{checkpoint}.pt")
    policy = runner.get_inference_policy(device=env.device)
    
    
    obs, _ = env.reset()

    if do_record:
        env.cam.start_recording()

    if do_log:
        log_file = open(output_filename+".csv", "w")
        log_file.write("i")
        for n in range(7):
            log_file.write(f",pos{n}")
        for n in range(7):
            log_file.write(f",vel{n}")
        log_file.write(",action")
        for n in range(env.cfg["num_obs_per_step"]):
            log_file.write(f",obs{n}")
        log_file.write("\n")

    with torch.no_grad():
        for i in range(2000): # doubled number of steps from 1000 to 2000
            actions = policy(obs)
            # actions = torch.ones((1,1))

            if i%400 > 0 and i%400 < 50:
                actions = actions*0


            obs, _, rews, dones, infos = env.step(actions)


            robot_pos = env.get_robot().get_pos()[0].flatten().cpu().numpy()
            robot_vel = env.get_robot().get_vel()[0].flatten().cpu().numpy()

            # Apply low pass filter to robot velocity
            if i == 0:
                # Initialize filtered velocity on first iteration
                filtered_vel = robot_vel
            else:
                # Low pass filter with alpha=0.1 (smaller alpha = more smoothing)
                alpha = 0.1
                filtered_vel = alpha * robot_vel + (1 - alpha) * filtered_vel
            robot_vel = filtered_vel

            dof_vel = env.dof_vel[0].flatten().cpu().numpy()
            action = actions[0].flatten().cpu().numpy()

            if do_record:
                
                offset_x = 1.0  # centered horizontally
                offset_y = -1.0 
                offset_z = 1  
                camera_pos = (float(robot_pos[0] + offset_x), float(robot_pos[1] + offset_y), offset_z)
                # print(camera_pos, tuple(float(x) for x in robot_pos))
                env.cam.set_pose(pos=camera_pos, lookat=(robot_pos[0], robot_pos[1], 0))

                robot_vel_plane = np.array([robot_vel[0], robot_vel[1], 0])
                
                env.scene.draw_debug_arrow(pos=robot_pos, vec=env.commands[0].cpu()*0.3, color=(1,0,0,0.5))
                env.scene.draw_debug_arrow(pos=robot_pos, vec=robot_vel_plane*0.3, color=(0,1,0,0.5))
                env.cam.render()
                env.scene.clear_debug_objects()

            if do_log:
                log_file.write(f"{i}")
                for dof_pos in env.get_robot().get_dofs_position()[0].flatten().cpu().numpy():
                    log_file.write(f",{dof_pos}")
                for dof_vel in env.get_robot().get_dofs_velocity()[0].flatten().cpu().numpy():
                    log_file.write(f",{dof_vel}")
                log_file.write(f",{action[0]}")
                obs_last = obs[0].reshape((env.cfg["num_obs_per_step"], env.cfg["num_obs_hist"]))[:,0].flatten().cpu().numpy()
                for obs_single in obs_last:
                    log_file.write(f",{obs_single}")
                log_file.write("\n")




            if i % 10 == 0:
                print(i)
                print(float(dof_vel), float(action), env._reward_regularize())

        env.cam.stop_recording(output_filename+".mp4", fps=int(0.5 * 1/env.control_dt)) 
            


if __name__ == "__main__":

    exp_name = "intui2torquerand_s1m0.7r10r1_2025-03-27_14-44-41"
    
    rock_eval(exp_name, checkpoint=-1, do_log=True)