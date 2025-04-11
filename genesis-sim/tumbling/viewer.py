from rock_env import *
from pynput import keyboard
import time




if __name__ == "__main__":
    running = False
    angvel = [0.0, 0.0, 0.0]

    def on_press(key):
        global running, angvel
        if key == keyboard.Key.space:
            running = not running
        elif key == keyboard.Key.up:
            angvel[0] += 0.1
            print("angvel", angvel)
        elif key == keyboard.Key.down:
            angvel[0] -= 0.1
            print("angvel", angvel)
        elif key == keyboard.Key.left:
            angvel[1] += 0.1
            print("angvel", angvel)
        elif key == keyboard.Key.right:
            angvel[1] -= 0.1
            print("angvel", angvel)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()



    np.set_printoptions(precision=3, linewidth=200)
    gs.init(logging_level="warning")

    print("Building env")
    # env = RockEnv(num_envs=128, show_viewer=True)
    cfg = RockEnv.env_cfg
    # cfg['gravity'] = [0.0, 0.0, 0.0]

    env = RockEnv(num_envs=3, show_viewer=True, env_cfg=cfg, viewer_timescale=4)

    print("Starting simulation")
    env.reset()

    robot = env.get_robot()

    # robot.set_quat(torch.tensor([[0.9, 0.415, 0.0, 0.0]]))
    robot.set_quat(torch.tensor([[1, 0, 0, 0]]).repeat((env.num_envs, 1)))

    robot.set_dofs_kp([1000], dofs_idx_local=env.motor_dofs)

    i = 0
    while True:
        if not running:
            time.sleep(0.01)
            continue #pause
        i += 1

        env.scene.clear_debug_objects()

        robot_pos = robot.get_pos()[0].flatten().cpu().numpy()
        robot_vel = robot.get_vel()[0].flatten().cpu().numpy()
        robot_quat = robot.get_quat()[0].flatten().cpu().numpy()

        dof_pos = robot.get_dofs_velocity()[0].flatten().cpu().numpy()
        def quatmult(q1, q2):
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            w = w1*w2 - x1*x2 - y1*y2 - z1*z2
            x = w1*x2 + x1*w2 + y1*z2 - z1*y2
            y = w1*y2 - x1*z2 + y1*w2 + z1*x2
            z = w1*z2 + x1*y2 - y1*x2 + z1*w2
            return np.array([w, x, y, z])
        
        q_offset = np.array([0.7071, 0.0, 0.0, 0.7071])
        q_imu = quatmult(robot_quat, q_offset)
        

        print(f"quat: {robot_quat}")
        print(f"rot_robot: {gs.quat_to_xyz(robot_quat)}")
        print(f"q_imu: {q_imu}")
        print(f"rot_imu: {gs.quat_to_xyz(q_imu)}")





        # robot.set_pos(torch.tensor([[0.0, 0.0, 0.1]]), zero_velocity=False)
        # robot.set_dofs_velocity(torch.tensor([angvel,])*50, dofs_idx_local=[3,4,5]) # body angular velocity
        
        torque = 1*(env.proj_angle - robot.get_dofs_position(env.motor_dofs))
        obs, _, rews, dones, infos = env.step( torque)

        if i % 10 == 0:
            print(i)

        motorjoints = robot.joints[-1]
        # for joint in joints:
        #     print(joint)

        env.commands = torch.zeros((env.num_envs, 3), device=env.device)
        env.commands[:,0] = 1.0


        
        env.scene.draw_debug_arrow(pos=robot_pos, vec=env.commands[0].cpu()*0.3, color=(1,0,0,0.5))




        






        # lin_acc_global = np.zeros(3)
        # for i_d in range(3):
        #     lin_acc_global[i_d] = robot.solver.dofs_state[i_d,0].acc
        # print("lin_acc_global")
        # print(lin_acc_global)



        # #create 4x4 transformatino matrix from root position and orientation
        # # Create 4x4 transformation matrix from position and quaternion
        # T_base = np.eye(4)
        # # Convert quaternion to 3x3 rotation matrix
        # qw, qx, qy, qz = robot_quat
        # T_base[:3,:3] = np.array([
        #     [1-2*qy*qy-2*qz*qz, 2*qx*qy-2*qz*qw, 2*qx*qz+2*qy*qw],
        #     [2*qx*qy+2*qz*qw, 1-2*qx*qx-2*qz*qz, 2*qy*qz-2*qx*qw],
        #     [2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*qx*qx-2*qy*qy]
        # ])
        # # Set translation
        # T_base[:3,3] = robot_pos

        # #vector from base to IMU
        # p_imu = np.array([-0.02989, 0.06366, -0.0032])
        
        # # Create 4x4 transformation matrix from IMU position and rotation
        # T_imu_translate = np.eye(4)
        # T_imu_translate[:3,3] = p_imu   # Set translation vector


        # x_rot_imu = np.array([
        #     [1, 0, 0],
        #     [0, np.cos(PI), -np.sin(PI)],
        #     [0, np.sin(PI), np.cos(PI)],
        # ])
        # z_rot_imu = np.array([
        #     [np.cos(PI/2), -np.sin(PI/2), 0],
        #     [np.sin(PI/2), np.cos(PI/2), 0],
        #     [0, 0, 1]
        # ])
        # R_imu = x_rot_imu @ z_rot_imu
        # T_imu_rotate = np.eye(4)
        # T_imu_rotate[:3,:3] = R_imu  # Set rotation block
        

        # T_imu = T_base @ T_imu_translate @ T_imu_rotate

        # r_imu_base = np.linalg.inv(T_base) @ T_imu
        # r_imu_base = r_imu_base[:3, 3]  # Translation component


        # env.scene.draw_debug_frame(T=T_base, origin_size=0.005, axis_radius=0.002, axis_length=0.1)
        # env.scene.draw_debug_frame(T=T_imu, origin_size=0.005, axis_radius=0.002, axis_length=0.1)
        # # env.scene.draw_debug_arrow(pos=robot_pos, vec=env.commands[0].cpu()*0.3, color=(1,0,0,0.5))
        # env.scene.draw_debug_arrow(pos=robot_pos, vec=lin_acc_global*0.05, color=(1,0,0,0.5))

        # offset_x = 0.0  # centered horizontally
        # offset_y = -1.0 
        # offset_z = 0.5  
        # camera_pos = (float(robot_pos[0] + offset_x), float(robot_pos[1] + offset_y), float(robot_pos[2] + offset_z))
        # # print(camera_pos, tuple(float(x) for x in robot_pos))
        # # env.cam.set_pose(pos=camera_pos, lookat=tuple(float(x) for x in robot_pos))
        # # env.cam.render()

        # env.scene.clear_debug_objects()
