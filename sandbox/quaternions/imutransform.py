import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

# Quaternion utilities
def quat_conjugate(q):
    w, x, y, z = q
    return [w, -x, -y, -z]

def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ]

def quat_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])

def normalize_quat(q):
    q = np.array(q)
    return (q / np.linalg.norm(q)).tolist()

# Static known transform: imu → body
q_imu_to_body = normalize_quat([0, 0.707, 0.707, 0])  # 90 deg rotation around X-Y diagonal
# q_imu_to_body_inv = quat_conjugate(q_imu_to_body)
q_imu_to_body_inv = (q_imu_to_body)

# Set up plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1.2, 1.2])
ax.set_ylim([-1.2, 1.2])
ax.set_zlim([-1.2, 1.2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=30, azim=45)
ax.set_title("IMU Frame (dashed) vs Body Frame (solid)")

# Base frame vectors
origin = np.array([0, 0, 0])
axes = np.eye(3)

# Draw axes for IMU and Body (placeholder lines)
body_lines = [ax.plot([0,0], [0,0], [0,0], 'r-')[0],  # x
              ax.plot([0,0], [0,0], [0,0], 'g-')[0],  # y
              ax.plot([0,0], [0,0], [0,0], 'b-')[0]]  # z

imu_lines = [ax.plot([0,0], [0,0], [0,0], 'r--', linewidth=3)[0],  # x
             ax.plot([0,0], [0,0], [0,0], 'g--', linewidth=3)[0],  # y
             ax.plot([0,0], [0,0], [0,0], 'b--', linewidth=3)[0]]  # z


# Sliders for IMU orientation
axcolor = 'lightgoldenrodyellow'
slider_axs = [plt.axes([0.2, 0.02 + 0.05*i, 0.65, 0.03], facecolor=axcolor) for i in range(3)]
sliders = [
    Slider(slider_axs[0], 'Roll (°)', -180, 180, valinit=0),
    Slider(slider_axs[1], 'Pitch (°)', -180, 180, valinit=0),
    Slider(slider_axs[2], 'Yaw (°)', -180, 180, valinit=0)
]

def euler_to_quat(roll, pitch, yaw):
    """ Convert roll-pitch-yaw in degrees to quaternion (w,x,y,z) """
    r, p, y = np.radians([roll, pitch, yaw])
    cr = np.cos(r/2)
    sr = np.sin(r/2)
    cp = np.cos(p/2)
    sp = np.sin(p/2)
    cy = np.cos(y/2)
    sy = np.sin(y/2)

    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return [w, x, y, z]

def update(val=None):
    # Get current slider values
    roll = sliders[0].val
    pitch = sliders[1].val
    yaw = sliders[2].val

    q_imu = normalize_quat(euler_to_quat(roll, pitch, yaw))
    q_body = quat_multiply(q_imu, q_imu_to_body_inv)

    R_imu = quat_to_rotation_matrix(q_imu)
    R_body = quat_to_rotation_matrix(q_body)

    # Update IMU lines (dashed)
    for i in range(3):
        vec = R_imu[:, i]
        imu_lines[i].set_data([0, vec[0]], [0, vec[1]])
        imu_lines[i].set_3d_properties([0, vec[2]])

    # Update Body lines (solid)
    for i in range(3):
        vec = R_body[:, i]*1.5
        body_lines[i].set_data([0, vec[0]], [0, vec[1]])
        body_lines[i].set_3d_properties([0, vec[2]])

    print(f"q_imu  (IMU → global): \n{np.round(quat_to_rotation_matrix(q_imu), 4)}")
    print(f"q_body (Body → global): \n{np.round(quat_to_rotation_matrix(q_body), 4)}")
    print("—" * 50)
    fig.canvas.draw_idle()

# Initial draw
update()

for slider in sliders:
    slider.on_changed(update)

plt.show()
