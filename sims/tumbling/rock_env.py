import torch
import math
import genesis as gs
import numpy as np
import os
from genesis.utils.geom import xyz_to_quat, quat_to_xyz, axis_angle_to_quat, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from genesis.engine.entities.rigid_entity import RigidEntity

PI = torch.pi

class RockEnv:
    SIM_DIR = os.path.dirname(os.path.abspath(__file__))

    env_cfg = {
        # "urdf_path": "onshape/pmrock/pmrock.urdf",
        "urdf_path": "../onshape/balo2/balo_2.urdf", # fixed file path for new sim folders

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
        # "termination_if_roll_greater_than": 45,  # degree
        # "termination_if_pitch_greater_than": 45,
        "termination_if_angle_greater_than": 90, #degree

        # base pose
        "base_init_pos": [0.0, 0.0, 0.08],
        # "base_init_pos": [0.0, 0.0, 0.1],
        "base_init_quat": [ 0, 0.7071068, 0, 0.7071068 ],#[1., 0., 0., 0.], #rotate rock 90 so it is on its side

        "dt": 0.001,
        "substeps": 2,
        "episode_length_s": 3.0,
        "max_torque": 0.6,
        "max_motor_speed": 276, # 2640 rpm 

        "initial_motor_speed": 15,
        "initial_shell_speed": 10,
        # "initial_motor_speed": 0,
        # "initial_shell_speed": 0,
        "lean_angle": 5,

        "friction": 2,
        "gravity": [0, 0, -9.81],#[0, 0, -1], normal gravity conditions
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
            sim_options=gs.options.SimOptions(
                dt=self.dt, 
                substeps=2, 
                gravity=self.cfg["gravity"],
            ),
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
                file=f'{RockEnv.SIM_DIR}/../{self.cfg["urdf_path"]}',
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        if add_camera:
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(0.5, 0.5, 0.5), #diag
                # pos=(0.0, 0.0, 0.5), #on top
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
            self.reward_functions[name] = getattr(self, "_reward_" + name) # finds all the functions starting with _reward
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

        # motor_speed = self.get_robot().get_dofs_velocity(self.motor_dofs)

        # torques = 100*torch.clip(self.actions, -self.cfg["max_torque"], self.cfg["max_torque"])

        # torques = torques - self.cfg["max_torque"] * (motor_speed / self.cfg["max_motor_speed"])
        # self.get_robot().control_dofs_force(torques, self.motor_dofs) 
        # self.get_robot().control_dofs_velocity([[10]], self.motor_dofs)

        target_speed = torch.clip(self.actions + self.cfg["initial_motor_speed"], -self.cfg["max_motor_speed"], self.cfg["max_motor_speed"])
        self.get_robot().control_dofs_velocity(target_speed, self.motor_dofs)

        # self.get_robot().control_dofs_velocity(self.cfg["initial_motor_speed"]*torch.ones((1,1)), self.motor_dofs)
        

        for i in range(10):
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
        # self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.cfg["termination_if_pitch_greater_than"]
        # self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.cfg["termination_if_roll_greater_than"]
        #dot product of projected gravity and [0,0,-1] = cos of angle
        self.reset_buf |= torch.abs(self.projected_gravity[:, 2]) < np.cos(np.deg2rad(self.cfg["termination_if_angle_greater_than"]))

        #get unit vector corresponding to base_euler
        self.base_euler_unit = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_euler_unit[:, 0] = torch.sin(self.base_euler[:, 1]) * torch.cos(self.base_euler[:, 0])
        self.base_euler_unit[:, 1] = torch.sin(self.base_euler[:, 1]) * torch.sin(self.base_euler[:, 0])
        self.base_euler_unit[:, 2] = torch.cos(self.base_euler[:, 1])


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
        
        lean_axis_yaw = 2*PI * torch.rand(1, device=self.device)
        lean_axis = torch.tensor([torch.sin(lean_axis_yaw), torch.cos(lean_axis_yaw), 0], device=self.device)
        lean_angle = torch.deg2rad(torch.tensor(self.cfg["lean_angle"], device=self.device))
        
        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        # self.base_quat[envs_idx] = self.base_init_quat
        self.base_quat[envs_idx] = axis_angle_to_quat(lean_angle, lean_axis)
        self.get_robot().set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.get_robot().set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        # self.get_robot().set_quat(self.base_quat[envs_idx] + 0.05*torch.rand_like(self.base_quat[envs_idx]), zero_velocity=True, envs_idx=envs_idx)
        # self.base_lin_vel[envs_idx] = 0
        # self.base_ang_vel[envs_idx] = 0

        # [x, y, z, rx, ry, rz] #global frame
        # n x dn/dt
            # theta = torch.deg2rad(torch.tensor(10, device=self.device))
            # n_vec = transform_by_quat(vert_spin, self.base_quat[:])
            # phi = torch.atan2(n_vec[:,1], n_vec[:,0])
            # # print(phi)
            # joystick_angvel = torch.tensor([
            #     -omega*torch.sin(theta)*torch.cos(theta)*torch.cos(phi),
            #     -omega*torch.sin(theta)*torch.cos(theta)*torch.sin(phi),
            #     omega*(torch.sin(theta))**2,
            # ], device=self.device).reshape(1,3)
        spin = self.cfg["initial_shell_speed"]

        #vert_spin is a tensor with shape (envs_idx, 3)
        #copy it to all envs
        vert_spin = torch.tensor([[0.0, 0.0, spin]], device=self.device).expand((len(envs_idx), 3))
        joystick_angvel = vert_spin - transform_by_quat(vert_spin, self.base_quat[envs_idx])
        radius_vec = 0.0776 * torch.nn.functional.normalize(transform_by_quat(vert_spin, self.base_quat[envs_idx]), p=2.0, dim = 1)
        joystick_linvel = torch.cross(vert_spin, radius_vec) # v x r
        self.get_robot().set_dofs_velocity(joystick_linvel, dofs_idx_local=[0,1,2], envs_idx=envs_idx)
        self.get_robot().set_dofs_velocity(joystick_angvel, dofs_idx_local=[3,4,5], envs_idx=envs_idx)

        # reset dofs
        self.dof_pos[envs_idx] = -lean_axis_yaw - PI/2
        self.dof_vel[envs_idx] = self.cfg["initial_motor_speed"]
        self.get_robot().set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=False,
            envs_idx=envs_idx,
        )
        self.get_robot().set_dofs_velocity(
            velocity=self.dof_vel[envs_idx],
            dofs_idx_local=self.motor_dofs,
            envs_idx=envs_idx,
        )        

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
    
    # linear velocity reward function referenced from genesis locomotion example !!!!!!
    # def _reward_tracking_lin_vel(self):
    #     # Tracking of linear velocity along x and y axis ... seems to look at linear displacement
    #     lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
    #     return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"]) 

    def _reward_tracking_lin_vel_x(self): # not entirely sure if this will make it go in a straight line along x axis or not
        # provide positive reinforcement for linear velocity along x axis
        lin_vel_error = torch.sum(torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0]), dim=1)
        #reward would be based on how closely the actual linear velocity direction aligns with the desired direction
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])
    
    def _reward_tracking_lin_vel_y(self): 
        # provide positive reinforcement for linear velocity along y axis
        lin_vel_error = torch.sum(torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])
    
    def _reward_action_rate(self):
        # Penalize changes in actions, encourages going towards the same action that provides the best reward
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    # def _reward_tracking(self):
    #     #tracking is cmd[0]
    #     self.get_robot().get_vel()
    
    def _reward_regularize(self):
        return torch.exp(-4 * (self.actions.sum(-1) / 0.6)**2)
    

if __name__ == "__main__":
    gs.init(logging_level="warning")

    print("Building env")
    env = RockEnv(num_envs=2, add_camera=True)
    # env = RockEnv(num_envs=1, show_viewer=True)

    print("Starting simulation")
    env.reset()
    env.cam.start_recording()
    for i in range(500):
    # i=0
    # while True:
    #     i+=1

        
        obs, _, rews, dones, infos = env.step(100*torch.ones((env.num_envs,1), device=env.device))

        if i % 10 == 0:
            # print(obs)
            print(i)
        env.cam.render()

        # if i % 100 == 0:
        #     env.reset()

    env.cam.stop_recording(f"{RockEnv.SIM_DIR}/test3.mp4", fps=30)