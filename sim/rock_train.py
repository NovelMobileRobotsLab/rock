from rock_env import RockEnv
import json
import os

from rsl_rl.runners import OnPolicyRunner
import genesis as gs



exp_name = "rock_test1"
learning_iterations = 1000
seed = 1
num_envs = 4096

train_cfg = {
    "algorithm": {
        "clip_param": 0.2,
        "desired_kl": 0.01,
        "entropy_coef": 0.01,
        "gamma": 0.99,
        "lam": 0.95,
        "learning_rate": 0.001,
        "max_grad_norm": 1.0,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "schedule": "adaptive",
        "use_clipped_value_loss": True,
        "value_loss_coef": 1.0,
    },
    "init_member_classes": {},
    "policy": {
        "activation": "elu",
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "init_noise_std": 1.0,
    },
    "runner": {
        "algorithm_class_name": "PPO",
        "checkpoint": -1,
        "experiment_name": exp_name,
        "load_run": -1,
        "log_interval": 1,
        "num_steps_per_env": 24,
        "policy_class_name": "ActorCritic",
        "record_interval": -1,
        "resume": False,
        "resume_path": None,
        "run_name": "",
        "runner_class_name": "runner_class_name",
        "save_interval": 100,
    },
    "runner_class_name": "OnPolicyRunner",
    "seed": seed,
}



if __name__ == "__main__":

    run_dir = f"{RockEnv.SIM_DIR}/runs/{exp_name}"
    os.makedirs(run_dir, exist_ok=True)

    # default environment config
    env_cfg = RockEnv.env_cfg

    # # load environment config from file
    # with open(f"{run_dir}/env_cfg.json", "r") as f:
    #     env_cfg = json.load(f)

    env_cfg['kp'] = 99

    # save environment config to file
    with open(f"{run_dir}/env_cfg.json", "w") as f:
        json.dump(env_cfg, f, indent=4)

    # save training config to file
    with open(f"{run_dir}/train_cfg.json", "w") as f:
        json.dump(train_cfg, f, indent=4)


    print(f"Starting run {exp_name}")
    gs.init(logging_level="warning")
    env = RockEnv(num_envs, env_cfg, add_camera=True)
    
    runner = OnPolicyRunner(env, train_cfg, f"{run_dir}/models", device=env.device)

    os.system(f"tensorboard --logdir {run_dir}")

    runner.learn(learning_iterations)