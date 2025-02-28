import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from numpy import sin, cos

# Quaternion multiplication (for rotation composition)
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

# Normalize a quaternion
def normalize_quaternion(q):
    return q / np.linalg.norm(q)

# Quaternion conjugate
def quaternion_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

# Quaternion from axis-angle
def quaternion_from_axis_angle(axis, angle):
    axis = axis / np.linalg.norm(axis)
    half_angle = angle / 2
    s = np.sin(half_angle)
    return np.array([np.cos(half_angle), axis[0] * s, axis[1] * s, axis[2] * s])

# Rotate a vector by a quaternion
def rotate_vector(q, v):
    qv = np.array([0, v[0], v[1], v[2]])
    q_conj = quaternion_conjugate(q)
    q_rot = quaternion_multiply(q, quaternion_multiply(qv, q_conj))
    return q_rot[1:4]

# Quaternion derivative given angular velocity
def quaternion_derivative(q, omega):
    # omega is angular velocity [ωx, ωy, ωz]
    # q_dot = (1/2) * q * ω (pure quaternion [0, ωx, ωy, ωz])
    omega_quat = np.array([0, omega[0], omega[1], omega[2]])
    q_dot = 0.5 * quaternion_multiply(omega_quat, q)
    return q_dot

def quaternion_to_rotation_matrix(q):
    return np.array([
        [1 - 2 * q[2]**2 - 2 * q[3]**2, 2 * q[1] * q[2] - 2 * q[0] * q[3], 2 * q[1] * q[3] + 2 * q[0] * q[2]],
        [2 * q[1] * q[2] + 2 * q[0] * q[3], 1 - 2 * q[1]**2 - 2 * q[3]**2, 2 * q[2] * q[3] - 2 * q[0] * q[1]],
        [2 * q[1] * q[3] - 2 * q[0] * q[2], 2 * q[2] * q[3] + 2 * q[0] * q[1], 1 - 2 * q[1]**2 - 2 * q[2]**2]
    ])

# Define the body's initial reference vectors (e.g., axes of a triad)
body_vectors = np.array([
    [1, 0, 0],  # x-axis (red)
    [0, 1, 0],  # y-axis (green)
    [0, 0, 1]   # z-axis (blue)
])

# Simulation parameters
dt = 0.02  # Time step (seconds)
time_steps = 1000  # Number of steps
tilt_angle = np.deg2rad(10)
q = quaternion_from_axis_angle(np.array([1.0, 0.0, 0.0]), tilt_angle)
orientations = []  # Store rotated vectors for animation



# Integrate orientation over time
for t in np.arange(0, time_steps * dt, dt):
    # Get angular velocity at current time

    omega=10
    theta=tilt_angle
    #phi is the global yaw angle
    R = quaternion_to_rotation_matrix(q)
    z_axis = R @ np.array([0, 0, 1])
    phi = np.arctan2(z_axis[1], z_axis[0])
    global_angvel = [
        -omega*sin(theta)*cos(theta)*cos(phi),
        -omega*sin(theta)*cos(theta)*sin(phi),
        omega*(sin(theta))**2,
    ]
    
    # Compute quaternion derivative
    q_dot = quaternion_derivative(q, global_angvel)
    
    # Update quaternion (Euler integration)
    q = q + q_dot * dt
    q = normalize_quaternion(q)  # Ensure unit quaternion
    
    # Rotate body vectors
    rotated_vectors = np.array([rotate_vector(q, v) for v in body_vectors])
    orientations.append(rotated_vectors)

# Convert to array for animation
orientations = np.array(orientations)



# Set up 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-1.5, 1.5])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Initialize quiver plots for body axes
colors = ['r', 'g', 'b']
quivers = [ax.quiver(0, 0, 0, 1, 0, 0, color=c) for c in colors]

#draw origin axes using quiver, no arrow head
ax.quiver(0, 0, 0, 1, 0, 0, color='k', arrow_length_ratio=0)
ax.quiver(0, 0, 0, 0, 1, 0, color='k', arrow_length_ratio=0)
ax.quiver(0, 0, 0, 0, 0, 1, color='k', arrow_length_ratio=0)

# Animation update function
def update(frame):
    rotated = orientations[frame]
    for i, quiver in enumerate(quivers):
        quiver.remove()
        quivers[i] = ax.quiver(0, 0, 0, rotated[i, 0], rotated[i, 1], rotated[i, 2], color=colors[i])
        ax.set_title(f'Time: {frame * dt:.2f} seconds')
    return quivers

# Create animation
ani = animation.FuncAnimation(fig, update, frames=time_steps, interval=1000*dt, blit=False)

plt.show()