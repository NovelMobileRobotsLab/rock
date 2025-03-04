import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Quaternion multiplication
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
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-10 else q

# Quaternion conjugate
def quaternion_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

# Rotate a vector by a quaternion
def rotate_vector(q, v):
    qv = np.array([0, v[0], v[1], v[2]])
    q_conj = quaternion_conjugate(q)
    q_rot = quaternion_multiply(q, quaternion_multiply(qv, q_conj))
    return q_rot[1:4]

# Quaternion from axis-angle
def quaternion_from_axis_angle(axis, angle):
    axis = axis / np.linalg.norm(axis)
    half_angle = angle / 2
    s = np.sin(half_angle)
    return np.array([np.cos(half_angle), axis[0] * s, axis[1] * s, axis[2] * s])

# Quaternion derivative for body-frame ω (helper function)
def quaternion_derivative_body(q, omega_body):
    omega_quat = np.array([0, omega_body[0], omega_body[1], omega_body[2]])
    q_dot = 0.5 * quaternion_multiply(omega_quat, q)  # Body-frame: ω * q
    return q_dot

# Convert quaternion to rotation matrix
def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    return R

# Define the body's initial reference vectors (e.g., axes of a triad)
body_vectors = np.array([
    [1, 0, 0],  # x-axis (red)
    [0, 1, 0],  # y-axis (green)
    [0, 0, 1]   # z-axis (blue)
])

# Simulation parameters
dt = 0.02  # Time step (seconds)
time_steps = 500  # Number of steps

# Initial quaternion: tilted 45 degrees around x-axis
tilt_angle = np.deg2rad(90)
q = quaternion_from_axis_angle(np.array([1.0, 0.0, 0.0]), tilt_angle)
orientations = []  # Store rotated vectors for animation

# Angular velocity function in global frame
def get_angular_velocity_global(t):
    # Constant rotation around global z-axis
    return np.array([0.0, 0.0, 0.5])
    # Alternative: Decaying oscillation
    # amplitude = np.exp(-t / 5.0)
    # return amplitude * np.array([0.5 * np.sin(t), 0.5 * np.cos(t), 0.3])

# Integrate orientation over time
for t in np.arange(0, time_steps * dt, dt):
    # Get angular velocity in global frame
    omega_global = get_angular_velocity_global(t)
    
    # Convert current orientation to rotation matrix
    R = quaternion_to_rotation_matrix(q)
    
    # Transform global ω to body frame: ω_body = R^T * ω_global
    omega_body = R.T @ omega_global
    
    # Compute quaternion derivative using body-frame ω
    q_dot = quaternion_derivative_body(q, omega_body)
    
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

# Animation update function
def update(frame):
    rotated = orientations[frame]
    for i, quiver in enumerate(quivers):
        quiver.remove()
        quivers[i] = ax.quiver(0, 0, 0, rotated[i, 0], rotated[i, 1], rotated[i, 2], color=colors[i])
    return quivers

# Create animation
ani = animation.FuncAnimation(fig, update, frames=time_steps, interval=1000*dt, blit=False)

plt.show()