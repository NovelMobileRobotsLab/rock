import traceback
from rock_env import RockEnv
import json
import os
import itertools
import copy

from rsl_rl.runners import OnPolicyRunner
import genesis as gs
from datetime import datetime


# Define parameter sweeps (feel free to modify these values)
gamma_values = [0.97, 0.99, 0.995, 0.999]  # Discount factor values
lambda_values = [0.8, 0.9, 0.95, 0.99]  # GAE lambda values
num_steps_values = [24, 48, 64, 96]  # Rollout length values

# Default parameters
# learning_iterations = 450
learning_iterations = 1000
seed = 1
num_envs = 4096

# Base training configuration
base_train_cfg = {
    "algorithm": {
        "clip_param": 0.2,
        "desired_kl": 0.01,
        "entropy_coef": 0.01,
        "gamma": 0.995,  # Default value, will be overridden in parameter sweep
        "lam": 0.90,     # Default value, will be overridden in parameter sweep
        "learning_rate": 0.001,
        "max_grad_norm": 1.0,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "schedule": "adaptive",
        "use_clipped_value_loss": True,
        "value_loss_coef": 1.0,
    },
    # "algorithm": {
    #     "clip_param": 0.2,              # More flexibility in updates
    #     "desired_kl": 0.01,             # Keep for adaptive if retained
    #     "entropy_coef": 0.1,           # Boost exploration
    #     "gamma": 0.999,                  # Longer horizon for smaller dt
    #     "lam": 0.99,                    # Better advantage estimation
    #     "learning_rate": 0.001,        # Slower, stable learning
    #     "max_grad_norm": 1.0,           # Unchanged, stability cap
    #     "num_learning_epochs": 5,       # Unchanged, sufficient passes
    #     "num_mini_batches": 2,          # Larger updates for stability
    #     "schedule": "fixed",         # Test "fixed" if issues persist
    #     "use_clipped_value_loss": True, # Unchanged, helps stability
    #     "value_loss_coef": 0.1          # Balance value and policy
    # },
    "init_member_classes": {},
    "policy": {
        "activation": "relu",
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "init_noise_std": 1.0,
    },
    "runner": {
        "algorithm_class_name": "PPO",
        "checkpoint": -1,
        "experiment_name": "",  # Will be set in parameter sweep
        "load_run": -1,
        "log_interval": 1,
        "num_steps_per_env": 64,  # Default value, will be overridden in parameter sweep
        "policy_class_name": "ActorCritic",
        "record_interval": -1,
        "resume": False,
        "resume_path": None,
        "run_name": "",
        "runner_class_name": "runner_class_name",
        "save_interval": 50,
    },
    "runner_class_name": "OnPolicyRunner",
    "seed": seed,
}

def run_training(params_dict):
    """Run a single training session with the specified parameters.
    
    Args:
        params_dict: Dictionary containing parameters to modify. Each key should be in the format
                     "config:section:parameter" where config is either "train" or "env",
                     section is the section within the config (e.g., "algorithm" or "runner"),
                     and parameter is the name of the parameter to modify.
                     For env_cfg parameters, just use "env:parameter_name".
    """
    # Create a copy of the base config and get default env config
    train_cfg = copy.deepcopy(base_train_cfg)
    env_cfg = copy.deepcopy(RockEnv.env_cfg)
    
    # Build experiment name based on parameter keys and values
    exp_parts = ["CTE"]
    param_str = ""
    
    # Apply parameter changes and build experiment name
    for param_key, param_value in params_dict.items():
        parts = param_key.split(":")
        
        # Get first letter of the last part (parameter name)
        param_name = parts[-1]
        first_letter = param_name[0]
        
        # Add to experiment name
        param_str += f"{first_letter}{param_value}"
        
        # Apply parameter based on the config type
        if parts[0] == "train":
            if len(parts) == 3:  # train:section:parameter
                section, parameter = parts[1], parts[2]
                train_cfg[section][parameter] = param_value
            else:  # Direct train config parameter
                train_cfg[parts[1]] = param_value
        elif parts[0] == "env":
            if len(parts) == 2:  # env:parameter
                env_cfg[parts[1]] = param_value
    
    # Create experiment name with parameter values
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # exp_name = f"intui2torque_{param_str}_{timestamp}"
    exp_name = f"sincosproj_{param_str}_{timestamp}"
    train_cfg["runner"]["experiment_name"] = exp_name
    
    # Print parameter summary
    print(f"Starting run {exp_name} with parameters:")
    for key, value in params_dict.items():
        print(f"  {key} = {value}")
    
    run_dir = f"{RockEnv.SIM_DIR}/projruns/{exp_name}"
    os.makedirs(run_dir, exist_ok=True)

    # save environment config to file
    with open(f"{run_dir}/env_cfg.json", "w") as f:
        json.dump(env_cfg, f, indent=4)

    # save training config to file
    with open(f"{run_dir}/train_cfg.json", "w") as f:
        json.dump(train_cfg, f, indent=4)

    # copy the urdf at env_cfg into run_dir
    os.system(f"cp {RockEnv.SIM_DIR}/../{env_cfg['urdf_path']} {run_dir}")
    print(f"cp {RockEnv.SIM_DIR}/../{env_cfg['urdf_path']} {run_dir}")

    # copy the rock_env.py into run_dir
    os.system(f"cp {RockEnv.SIM_DIR}/rock_env.py {run_dir}")
    print(f"cp {RockEnv.SIM_DIR}/rock_env.py {run_dir}")

    gs.init(logging_level="warning")
    env = RockEnv(num_envs, env_cfg, add_camera=True)
    
    runner = OnPolicyRunner(env, train_cfg, f"{run_dir}/models", device=env.device)
    # last_run = 'novelest_s1r3_2025-04-03_11-57-27'
    # ckpt = 500
    # runner.load(f'{RockEnv.SIM_DIR}/penaltysweep/{last_run}/models/model_{ckpt}.pt', load_optimizer=False)
    # runner.current_learning_iteration = ckpt

    try:
        runner.learn(learning_iterations, init_at_random_ep_len=True)
    except Exception as e:
        # Write error to file
        with open(f"{run_dir}/error.txt", "w") as f:
            f.write(f"Error during training:\n{str(e)}\n\n")
            f.write("Traceback:\n")
            f.write(traceback.format_exc())
        print(f"Error occurred during training. See {run_dir}/error.txt for details.")

    # Cleanup to free memory
    del env
    del runner
    import gc
    gc.collect()
    
    # Properly close Genesis
    if hasattr(gs, "destroy"):
        gs.destroy()


if __name__ == "__main__":


    # for seed in [3,4,5,6,7]:
    #     for resampling_time_s in [1,3,5]:
    #         params = {
    #             "seed": seed,
    #             "env:resampling_time_s": resampling_time_s,
    #         }
    #         run_training(params)

    params = {
        "seed": 5,
        "env:resampling_time_s": 3,
    }
    run_training(params)


    # for seed in [2]:
    #     for misalignment_penalty in [0.5]:
    #         for action_rate_penalty in [5]:
    #             for regularization_rew in [1]:
    #                 params = {
    #                     "seed": seed,
    #                     "env:misalignment_penalty": misalignment_penalty,
    #                     "env:reward_scales:regularize": regularization_rew,
    #                     "env:reward_scales:action_rate": action_rate_penalty,
    #                     "env:resampling_time_s": 1,
    #                 }
    #                 run_training(params)


    # for seed in [1,2]:
    #     for misalignment_penalty in [0.2, 0.5, 0.8]:
    #         for regularize in [2, 5, 10]:
    #             for resampling_time_s in [1,3]:
    #                 params = {
    #                     "seed": seed,
    #                     "env:misalignment_penalty": misalignment_penalty,
    #                     "env:reward_scales:regularize": regularize,
    #                     "env:resampling_time_s": resampling_time_s,
    #                 }
    #                 run_training(params)
    
    # Example 2: Sweeping across gamma and lambda values only
    """
    for gamma in gamma_values:
        for lam in lambda_values:
            params = {
                "train:algorithm:gamma": gamma,
                "train:algorithm:lam": lam,
                "train:runner:num_steps_per_env": 64  # Fixed value
            }
            run_training(params)
    """
    
    # Example 3: Full parameter sweep
    
    # for gamma, lam, num_steps in itertools.product(gamma_values, lambda_values, num_steps_values):
    #     params = {
    #         "train:algorithm:gamma": gamma,
    #         "train:algorithm:lam": lam,
    #         "train:runner:num_steps_per_env": num_steps
    #     }
    #     run_training(params)
    
    
    # Example 4: Environment parameter sweep
    """
    for kp in [80, 100, 120]:
        for kd in [80, 100, 120]:
            params = {
                "env:kp": kp,
                "env:kd": kd,
                "train:algorithm:gamma": 0.99  # Fixed training parameters
            }
            run_training(params)
    """