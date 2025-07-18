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
        # "urdf_path": "../onshape/balo/balo.urdf", # fixed file path for new sim folders
        "urdf_path": "../onshape/intuition2/intuition2.urdf", 

        "num_commands": 3,
        "num_actions": 1, # angle of pendulum

        "num_obs_per_step": 16,
        "num_obs_hist": 3,  # number of previous observations to include

        "reward_scales": {
            "regularize": 5,
            "direction": 100.0,
            # "tracking_lin_vel": 1.0,
            "action_rate": 1,
        },
        "misalignment_penalty": 0.5,

        "tracking_sigma": 0.1,

        # joint names, initial position
        "dofs": {  # [rad]
            "Revolute_1": 0.0,
        },


        # base pose
        "base_init_pos": [0.0, 0.0, 0.08],
        # "base_init_pos": [0.0, 0.0, 0.1],
        "base_init_quat": [0.7071068 ,0, 0.7071068, 0],#[1., 0., 0., 0.], #rotate rock 90 so it is on its side


        "resampling_time_s": 3.0,

        "episode_length_s": 10.0,
        "max_torque": 0.2, #Nm
        "max_motor_speed": 276, # 2640 rpm 
        
        "control_dt": 0.01,
        "sim_steps_per_control": 10,
        "friction": 1,
        "gravity": [0, 0, -9.81], # normal gravity conditions
        "substeps": 2,

        "friction_range": [0.10, 1.00],
        "com_shift_scale": 0.01, #m
        "mass_shift_scale": 0.010, #kg
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

        self.max_episode_length = math.ceil(self.cfg["episode_length_s"] / self.cfg["control_dt"])
        self.control_dt = self.cfg["control_dt"]
        self.sim_dt = self.cfg["control_dt"] / self.cfg["sim_steps_per_control"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.sim_dt, 
                substeps=self.cfg["substeps"], 
                gravity=self.cfg["gravity"],
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.sim_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_self_collision=False,
                enable_joint_limit=False,
                contact_resolve_time=self.sim_dt*3, #mujoco recommends at least 2 times dt
            ),

            show_viewer=show_viewer,
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(viewer_timescale / self.control_dt),
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
                pos=(2, 2, 2), #diag
                # pos=(0.0, 0.0, 0.5), #on top
                lookat=(0, 0, 0.1),
                fov=30,
                GUI=False,
            )
            self.cam_top = self.scene.add_camera(
                res=(1920,1080),
                pos=(2, 2, 2), #diag
                # pos=(0.0, 0.0, 0.5), #on top
                lookat=(0, 0, 0.1),
                fov=30,
                GUI=False,
            )

        self.scene.build(n_envs=num_envs)

        
        self.get_robot().set_friction(self.cfg['friction'])
        
        # names to indices
        self.motor_dofs = [self.get_robot().get_joint(name).dof_idx_local for name in self.cfg["dofs"].keys()]

        # DOF parameters
        # self.get_robot().set_dofs_kp([self.cfg["kp"]], self.motor_dofs)
        # self.get_robot().set_dofs_kv([self.cfg["kd"]], self.motor_dofs)
        print("armature before", self.get_robot().get_dofs_armature())
        self.get_robot().set_dofs_armature((0,), dofs_idx_local=self.motor_dofs)
        print("armature after", self.get_robot().get_dofs_armature())

        print("damping before", self.get_robot().get_dofs_damping())
        self.get_robot().set_dofs_damping((0,), dofs_idx_local=self.motor_dofs)
        print("damping after", self.get_robot().get_dofs_damping())
        self.get_robot().set_dofs_force_range([-self.cfg["max_torque"]], [self.cfg["max_torque"]], self.motor_dofs)

        #domain randomization
        self.get_robot().set_friction_ratio(
            friction_ratio=self.cfg["friction_range"][0] + (self.cfg["friction_range"][1] - self.cfg["friction_range"][0]) * torch.rand(self.scene.n_envs, self.get_robot().n_links),
            link_indices=np.arange(0, self.get_robot().n_links),
        )
        self.get_robot().set_mass_shift(
            mass_shift = self.cfg["mass_shift_scale"] * torch.randn(self.scene.n_envs, self.get_robot().n_links),
            link_indices=np.arange(0, self.get_robot().n_links),
        )
        self.get_robot().set_COM_shift(
            com_shift = self.cfg["com_shift_scale"] * torch.randn(self.scene.n_envs, self.get_robot().n_links, 3),
            link_indices=np.arange(0, self.get_robot().n_links),
        )

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.control_dt
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
        self.last_actions = self.actions
        self.actions = torch.clip(actions, -1, 1)
        motor_speed = self.get_robot().get_dofs_velocity(self.motor_dofs)
        torques = self.cfg["max_torque"]*(self.actions - motor_speed / self.cfg["max_motor_speed"])
        self.get_robot().control_dofs_force(torques, self.motor_dofs);
        
        # target_speed = torch.clip(self.actions, -1, 1) * self.cfg["max_motor_speed"]
        # self.get_robot().control_dofs_velocity(target_speed, self.motor_dofs)


        
        for i in range(self.cfg["sim_steps_per_control"]):
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
        envs_idx_to_reset = (
            (self.episode_length_buf % int(self.cfg["resampling_time_s"] / self.control_dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self.resample_commands(envs_idx_to_reset)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        # self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.cfg["termination_if_pitch_greater_than"]
        # self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.cfg["termination_if_roll_greater_than"]
        #dot product of projected gravity and [0,0,-1] = cos of angle
        # self.reset_buf |= torch.abs(self.projected_gravity[:, 2]) < np.cos(np.deg2rad(self.cfg["termination_if_angle_greater_than"]))

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
                self.actions, # 1
                # self.base_euler / 60, # 3
                # self.base_lin_vel * 1e3,  # 3
                # self.base_ang_vel,  # 3
                self.get_robot().get_quat(), # 4
                # self.get_robot().get_pos(), # 3
                self.get_robot().get_ang() / 6,  # 3
                self.get_robot().get_vel() * 10,  # 3
                self.get_robot().get_dofs_velocity(self.motor_dofs) / 50,  # 1
                self.get_robot().get_dofs_position(self.motor_dofs) / 10,  # 1
                self.commands,  # 3
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
        
        # lean_axis_yaw = 2*PI * torch.rand(1, device=self.device)
        # lean_axis = torch.tensor([torch.sin(lean_axis_yaw), torch.cos(lean_axis_yaw), 0], device=self.device)
        # lean_angle = torch.deg2rad(torch.tensor(self.cfg["lean_angle"], device=self.device))

        # Generate random lean axis angle in radians (yaw around z-axis)
        lean_axis_yaw = 2*PI * torch.rand(len(envs_idx), device=self.device)
        # Create lean axis vector from yaw angle
        lean_axis = torch.stack([
            torch.sin(lean_axis_yaw),
            torch.cos(lean_axis_yaw),
            torch.zeros_like(lean_axis_yaw)
        ], dim=-1)
        
        # Random lean angle between 0 and 90 degrees
        lean_angle = torch.deg2rad(90 * torch.rand(len(envs_idx), device=self.device))
        
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


        # self.get_robot().set_dofs_velocity(joystick_linvel, dofs_idx_local=[0,1,2], envs_idx=envs_idx) # body linear velocity
        # self.get_robot().set_dofs_velocity(joystick_angvel, dofs_idx_local=[3,4,5], envs_idx=envs_idx) # body angular velocity

        # reset dofs
        self.dof_pos[envs_idx] = torch.rand(len(envs_idx), 1, device=self.device) * 2 * PI
        # self.dof_vel[envs_idx] = torch.rand(len(envs_idx), 1, device=self.device) * self.cfg["max_motor_speed"]
        self.get_robot().set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=False,
            envs_idx=envs_idx,
        )
        # self.get_robot().set_dofs_velocity(
        #     velocity=self.dof_vel[envs_idx],
        #     dofs_idx_local=self.motor_dofs,
        #     envs_idx=envs_idx,
        # )        

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

        self.resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_stacked.flatten(1), None # rsl-rl expects 2 outputs, only first is used
    

    def resample_commands(self, envs_idx=None):
        if envs_idx is None:
            envs_idx = torch.arange(self.num_envs, device=self.device)
        direction = torch.rand(len(envs_idx), device=self.device) * 2 * PI
        self.commands[envs_idx, 0] = torch.cos(direction) # x-axis component
        self.commands[envs_idx, 1] = torch.sin(direction) # y-axis component
        self.commands[envs_idx, 2] = 0 # z-axis component


    ''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ REWARD FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''

    # Reward for going in the direction of the command: project the current base linear velocity onto the command unit vector
    def _reward_direction(self):
        global_lin_vel = self.get_robot().get_vel()
        aligned_vel = torch.sum(self.commands[:, :2] * global_lin_vel[:, :2], dim=1)

        # Rotate velocity 90 degrees in xy plane
        rotated_vel = torch.zeros_like(global_lin_vel[:, :2])
        rotated_vel[:, 0] = -global_lin_vel[:, 1]  # x = -y 
        rotated_vel[:, 1] = global_lin_vel[:, 0]   # y = x
        misaligned_vel = torch.abs(torch.sum(self.commands[:, :2] * rotated_vel[:, :2], dim=1))

        # misaligned_vel = torch.abs(global_lin_vel - aligned_vel)
        aligned_vel = torch.clip(aligned_vel, min=-torch.inf, max=1)
        
        return aligned_vel - self.cfg["misalignment_penalty"]*misaligned_vel
    
    
    # Reward having actions close to 0
    def _reward_regularize(self):
        return torch.exp(-4 * self.actions.sum(-1)**2)
    
    # # linear velocity reward function referenced from genesis locomotion example !!!!!!
    # def _reward_tracking_lin_vel(self):
    #     # Tracking of linear velocity along x and y axis ... seems to look at linear displacement
    #     lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
    #     return torch.exp(-lin_vel_error / self.cfg["tracking_sigma"]) 
    

    # def _reward_tracking_lin_vel_x(self): # not entirely sure if this will make it go in a straight line along x axis or not
    #     # provide positive reinforcement for linear velocity along x axis
    #     lin_vel_error = torch.sum(torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0]), dim=1)
    #     #reward would be based on how closely the actual linear velocity direction aligns with the desired direction
    #     return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])
    
    # def _reward_tracking_lin_vel_y(self): 
    #     # provide positive reinforcement for linear velocity along y axis
    #     lin_vel_error = torch.sum(torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1]), dim=1)
    #     return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])
    
    def _reward_action_rate(self):
        # Penalize changes in actions, encourages going towards the same action that provides the best reward
        return -torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    # def _reward_tracking_ang_vel(self):
    #     # Tracking of angular velocity commands (yaw)
    #     ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
    #     return torch.exp(-ang_vel_error / self.cfg["tracking_sigma"])

    # def _reward_tracking(self):
    #     #tracking is cmd[0]
    #     self.get_robot().get_vel()
    
    

''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TESTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''


if __name__ == "__main__":
    np.set_printoptions(precision=3, linewidth=200)
    gs.init(logging_level="warning")

    print("Building env")
    env = RockEnv(num_envs=2, add_camera=True)

    # env = RockEnv(num_envs=1, show_viewer=True)

    print("Starting simulation")
    env.reset()
    env.cam.start_recording()
    # env.cam_top.start_recording()

    robot = env.get_robot()


    robot.set_dofs_kp((1,), dofs_idx_local=env.motor_dofs, envs_idx=range(2))
    robot.set_dofs_kv((1,), dofs_idx_local=env.motor_dofs, envs_idx=range(2))

    print("armature before", robot.get_dofs_armature())
    robot.set_dofs_armature((0,), dofs_idx_local=env.motor_dofs, envs_idx=range(2))
    print("armature after", robot.get_dofs_armature())

    print("damping before", robot.get_dofs_damping())
    robot.set_dofs_damping((0,), dofs_idx_local=env.motor_dofs, envs_idx=range(2))
    print("damping after", robot.get_dofs_damping())



    for i in range(300):
        
        obs, _, rews, dones, infos = env.step( 1 *torch.ones((env.num_envs,1), device=env.device))

        if i % 10 == 0:
            print(i)
            # print(obs[0])
        print("control force")
        print(robot.get_dofs_control_force()[0].flatten().cpu().numpy())
        # print(robot.get_dofs_kp().cpu().numpy())
        # print(robot.get_dofs_kv().cpu().numpy())
        # print(robot.get_dofs_position()[0].flatten().cpu().numpy()[-1])
        print("link inertial mass\n", robot.get_links_inertial_mass())
        print("link invweight\n", robot.get_links_invweight())

        motorjoints = robot.joints[-1]
        # for joint in joints:
        #     print(joint)

        massmats = robot.solver.mass_mat_inv.to_numpy()[:, :, 0]
        # print("massmats", massmats.shape)
        # print(np.diag(massmats))
        if i == 0:
            massmat_sum = massmats
        else:
            massmat_sum += massmats
        if i == 299: 
            print("massmat_avg", massmat_sum/300) 

        ''' normal masses
        massmats (7, 7)
[   3.162    4.182    4.324 1589.795 1485.914 1471.903    9.887]
massmat_avg [[ 3.897e+00 -5.845e-02  2.898e-01 -2.927e+00 -1.110e+00 -7.266e+00  3.119e-03]
 [-5.845e-02  3.711e+00 -5.736e-02  4.824e-01  2.894e+00 -1.461e+01 -1.435e-02]
 [ 2.898e-01 -5.736e-02  3.899e+00  1.207e+01  3.574e+00 -1.079e+01 -1.326e-02]
 [-2.927e+00  4.824e-01  1.207e+01  1.536e+03 -4.144e+01 -1.454e-01 -8.945e-02]
 [-1.110e+00  2.894e+00  3.574e+00 -4.144e+01  1.489e+03 -2.070e+01  3.030e+00]
 [-7.266e+00 -1.461e+01 -1.079e+01 -1.454e-01 -2.070e+01  1.529e+03 -3.311e-02]
 [ 3.119e-03 -1.435e-02 -1.326e-02 -8.945e-02  3.030e+00 -3.311e-02  9.887e+00]]
 
 big pendulum a d interias:
 massmats (7, 7)
[2.747 1.313 3.210 1.396e+03 1.246e+03 1.395, 9.882]
massmat_avg [[ 2.435e+00  6.354e-02 -1.810e-01  4.711e+00  9.624e-01 -1.062e+01 -4.193e-03]
 [ 6.354e-02  2.609e+00  1.442e-02 -4.349e+00 -4.664e+00 -2.062e+00  2.015e-02]
 [-1.810e-01  1.442e-02  2.061e+00  2.647e+00  9.177e+00  5.152e+00 -4.051e-02]
 [ 4.711e+00 -4.349e+00  2.647e+00  1.402e+03 -3.601e+01 -1.911e+00 -8.967e-02]
 [ 9.624e-01 -4.664e+00  9.177e+00 -3.601e+01  1.246e+03 -1.777e+01  4.150e+00]
 [-1.062e+01 -2.062e+00  5.152e+00 -1.911e+00 -1.777e+01  1.390e+03 -3.346e-02]
 [-4.193e-03  2.015e-02 -4.051e-02 -8.967e-02  4.150e+00 -3.346e-02  9.882e+00]]


 big mass pendulum only:
 massmats (7, 7)
[   2.846    2.354    2.335 1667.438 1283.549 1186.894    9.883]
massmat_avg [[ 1.747e+00  1.281e-01  5.434e-01 -9.319e-01  2.576e+00 -4.552e+00 -1.317e-02]
 [ 1.281e-01  2.961e+00 -3.023e-01  6.369e+00 -2.617e+01  5.103e+00  1.222e-01]
 [ 5.434e-01 -3.023e-01  2.533e+00  2.490e+00  2.041e+01 -1.824e+00 -9.324e-02]
 [-9.319e-01  6.369e+00  2.490e+00  1.432e+03 -3.277e+01  1.111e-01 -1.104e-01]
 [ 2.576e+00 -2.617e+01  2.041e+01 -3.277e+01  1.289e+03 -2.072e+01  3.950e+00]
 [-4.552e+00  5.103e+00 -1.824e+00  1.111e-01 -2.072e+01  1.428e+03 -2.338e-02]
 [-1.317e-02  1.222e-01 -9.324e-02 -1.104e-01  3.950e+00 -2.338e-02  9.883e+00]]
 
 '''
        

        robot_pos = robot.get_pos()[0].flatten().cpu().numpy()
        robot_vel = robot.get_vel()[0].flatten().cpu().numpy()

        
        env.scene.draw_debug_arrow(pos=robot_pos, vec=env.commands[0].cpu()*0.3, color=(1,0,0,0.5))

        offset_x = 0.0  # centered horizontally
        offset_y = -1.0 
        offset_z = 0.5  
        camera_pos = (float(robot_pos[0] + offset_x), float(robot_pos[1] + offset_y), float(robot_pos[2] + offset_z))
        # print(camera_pos, tuple(float(x) for x in robot_pos))
        env.cam.set_pose(pos=camera_pos, lookat=tuple(float(x) for x in robot_pos))
        env.cam.render()

        env.scene.clear_debug_objects()
        # env.cam_top.render()

        # if i % 20 == 0:
        #     env.reset()

    env.cam.stop_recording(f"{RockEnv.SIM_DIR}/testfollow2.mp4", fps=30)
    # env.cam_top.stop_recording(f"{RockEnv.SIM_DIR}/testfollow_top.mp4", fps=30)
