import torch
import math
import genesis as gs
import numpy as np
import os
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from genesis.engine.entities.rigid_entity import RigidEntity

class RockEnv:
    SIM_DIR = os.path.dirname(os.path.abspath(__file__))

    env_cfg = {
        "num_commands": 1,
        "num_actions": 1, # angle of pendulum

        "num_obs_per_step": 8,
        "num_obs_hist": 3,  # number of previous observations to include

        "reward_scales": {
            "alive": 1.0,
            "regularize": 0.1,
        },

        # joint names, initial position
        "dofs": {  # [rad]
            "Revolute_1": 0.0,
        },

        # PD
        # motor voltage = kp*(desired - current) + kd*(0 - current_vel)
        "kp": 100,
        "kd": 100,

        # termination
        "termination_if_roll_greater_than": 60,  # degree
        "termination_if_pitch_greater_than": 60,

        # base pose
        "base_init_pos": [0.0, 0.0, 0.020],
        "base_init_quat": [1., 0., 0.07, 0.],

        "dt": 0.001,
        "substeps": 2,
        "episode_length_s": 3.0,
        "max_torque": 0.6,
        "max_motor_speed": 276, # 2640 rpm 

        "friction": 1.0,
    }
    
    def __init__(self, num_envs:int, env_cfg=None, show_viewer=False, add_camera=False, viewer_timescale=0.5, device="cuda"):
        self.device = torch.device(device)

        self.num_envs = num_envs

        if env_cfg is not None:
            self.cfg = env_cfg
        else:
            self.cfg = RockEnv.env_cfg

        #shortcuts for commonly used configs
        self.num_actions = self.cfg["num_actions"]
        self.num_commands = self.cfg["num_commands"]
        self.reward_scales = self.cfg["reward_scales"]
        self.num_obs = self.cfg["num_obs_per_step"] * self.cfg["num_obs_hist"]
        self.num_privileged_obs = self.num_obs

        self.dt = self.cfg["dt"]
        self.max_episode_length = math.ceil(self.cfg["episode_length_s"] / self.dt)

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_self_collision=False,
                enable_joint_limit=False,
                contact_resolve_time=self.dt*3, #mujoco recommends at least 2 times dt
            ),

            show_viewer=show_viewer,
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(viewer_timescale / self.dt),
                camera_pos=(1.0, 0.0, 1.0),
                camera_lookat=(0.0, 0.0, 0.1),
                camera_fov=30,
            ),
            vis_options=gs.options.VisOptions(
                n_rendered_envs=1,
                show_world_frame=False,
            ),
        )

        # add plane
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add get_robot
        self.base_init_pos = torch.tensor(self.cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=f'{RockEnv.SIM_DIR}/../onshape/pmrock/pmrock.urdf',
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        if add_camera:
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(0.5, 0.5, 0.5),
                lookat=(0, 0, 0.1),
                fov=30,
                GUI=False,
            )

        self.scene.build(n_envs=num_envs)

        
        self.get_robot().set_friction(self.cfg['friction'])
        

        # names to indices
        self.motor_dofs = [self.get_robot().get_joint(name).dof_idx_local for name in self.cfg["dofs"].keys()]

        # PD control parameters
        self.get_robot().set_dofs_kp([self.cfg["kp"]], self.motor_dofs)
        self.get_robot().set_dofs_kv([self.cfg["kd"]], self.motor_dofs)
        self.get_robot().set_dofs_force_range([-100], [100], self.motor_dofs)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_stacked = torch.zeros((self.num_envs, self.cfg["num_obs_per_step"], self.cfg["num_obs_hist"]), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.cfg["dofs"][name] for name in self.cfg["dofs"].keys()],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging

    # casts self.get_robot to RigidEntity
    def get_robot(self) -> RigidEntity:
        return self.robot


    def step(self, actions):
        self.actions = actions

        motor_speed = self.get_robot().get_dofs_velocity(self.motor_dofs)

        torques = 100*torch.clip(self.actions, -self.cfg["max_torque"], self.cfg["max_torque"])

        torques = torques - self.cfg["max_torque"] * (motor_speed / self.cfg["max_motor_speed"])
        self.get_robot().control_dofs_force(torques, self.motor_dofs) 
        
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.get_robot().get_pos()
        self.base_quat[:] = self.get_robot().get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.get_robot().get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.get_robot().get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.get_robot().get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.get_robot().get_dofs_velocity(self.motor_dofs)

        # resample commands
        # envs_idx = (
        #     (self.episode_length_buf % int(self.cfg["resampling_time_s"] / self.dt) == 0)
        #     .nonzero(as_tuple=False)
        #     .flatten()
        # )
        # self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_stacked = torch.roll(self.obs_stacked, 1, dims=-1)     # shift obs 1 index later in hist dimension,
        self.obs_stacked[:,:,0] =  torch.cat(
            [
                # self.actions, # 1
                self.base_euler / 60, # 3
                # self.base_lin_vel * 1e3,  # 3
                self.base_ang_vel,  # 3
                self.get_robot().get_dofs_velocity(self.motor_dofs) / 50,  # 1
                self.get_robot().get_dofs_position(self.motor_dofs) / 10,  # 1
            ],
            axis=-1
        )

    
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        return self.obs_stacked.flatten(1), None, self.rew_buf, self.reset_buf, self.extras
    
    def get_observations(self):
        return self.obs_stacked.flatten(1)
    def get_privileged_observations(self):
        return self.obs_stacked.flatten(1)

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.get_robot().set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.get_robot().set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.get_robot().set_quat(self.base_quat[envs_idx] + 0.05*torch.rand_like(self.base_quat[envs_idx]), zero_velocity=True, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.get_robot().zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self.obs_stacked[envs_idx, :, :] = 0


        # self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_stacked.flatten(1), None # rsl-rl expects 2 outputs, only first is used

    def _reward_alive(self):
        return 1
    
    def _reward_regularize(self):
        return torch.exp(-4 * (self.actions.sum(-1) / 0.6)**2)
    

if __name__ == "__main__":
    gs.init(logging_level="warning")

    print("Building env")
    env = RockEnv(num_envs=1, add_camera=True)

    print("Starting simulation")
    env.cam.start_recording()
    for i in range(1000):

        
        obs, _, rews, dones, infos = env.step(1*torch.ones((1,1), device=env.device))

        if i % 24 == 0:
            print(obs)
            env.cam.render()

    env.cam.stop_recording(f"{RockEnv.SIM_DIR}/test2.mp4", fps=30)