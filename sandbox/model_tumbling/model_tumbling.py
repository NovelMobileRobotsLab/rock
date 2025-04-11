import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    """
    a = unit_vector(vec1)
    b = unit_vector(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    
    if s == 0:
        return np.eye(3)
    
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def create_circle_points(normal, zero_angle, radius=1.0, num_points=100):
    # Create a circle in the x-y plane
    theta = np.linspace(0, 2*np.pi, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros_like(theta)
    
    circle_points = np.vstack((x, y, z)).T
    
    # Create rotation matrix from [0,0,1] to the desired normal
    rot_matrix = rotation_matrix_from_vectors(np.array([0, 0, 1]), normal)
    
    # Rotate the circle to align with the normal vector
    rotated_points = np.dot(circle_points, rot_matrix.T)
    
    # Apply the zero angle rotation around the normal axis
    # First create a rotation matrix for the zero angle
    z_axis = np.array([0, 0, 1])
    zero_rot_matrix = np.array([
        [np.cos(zero_angle), -np.sin(zero_angle), 0],
        [np.sin(zero_angle), np.cos(zero_angle), 0],
        [0, 0, 1]
    ])
    
    # Apply the rotation in the original x-y plane before the normal rotation
    circle_points_zero = np.dot(circle_points, zero_rot_matrix.T)
    
    # Then apply the normal rotation
    rotated_points = np.dot(circle_points_zero, rot_matrix.T)
    
    return rotated_points

def get_point_at_angle(normal, zero_angle, angle, radius=1.0):
    # Get all points on the circle
    circle_points = create_circle_points(normal, zero_angle, radius, num_points=360)
    
    # The angle corresponds to the index in the circle points
    index = int(angle * 180 / np.pi) % 360
    
    return circle_points[index]

def get_angle_for_projection_direction(normal, zero_angle, target_direction, radius=1.0):
    """
    Calculate the angle needed on the circle to achieve a desired projection direction in the x-y plane.
    
    Args:
        normal: np.array([x, y, z]) - normal vector of the circle
        zero_angle: float - angle offset for the circle's zero position
        target_direction: np.array([x, y]) - desired direction in x-y plane (will be normalized)
        radius: float - radius of the circle
    
    Returns:
        float: angle (in radians) needed to achieve the projection direction
    """
    # Normalize the target direction
    target_direction = np.array([target_direction[0], target_direction[1], 0])
    target_direction = target_direction / np.linalg.norm(target_direction)
    
    # Get a set of test points around the circle
    test_angles = np.linspace(0, 2*np.pi, 360)
    points = np.array([get_point_at_angle(normal, zero_angle, angle, radius) for angle in test_angles])
    
    # Get the projection directions for each point
    # We only care about x-y components since we're projecting to x-y plane
    proj_directions = points[:, :2]  # Take only x and y components
    # Normalize each direction
    norms = np.linalg.norm(proj_directions, axis=1)
    valid_points = norms > 1e-10  # Avoid division by zero
    proj_directions[valid_points] /= norms[valid_points, np.newaxis]
    
    # Calculate dot product with target direction
    # This will be 1 when directions match exactly
    dot_products = np.dot(proj_directions, target_direction[:2])
    
    # Find the angle that gives the closest match
    best_idx = np.argmax(dot_products)
    return test_angles[best_idx]

# Create the figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Set equal aspect ratio
ax.set_box_aspect([1, 1, 1])


# Initial values
initial_normal = np.array([0, 0, 1])  # z-axis
initial_zero_angle = 0
initial_angle = 0
radius = 1.0

# Create the initial circle
circle_points = create_circle_points(initial_normal, initial_zero_angle, radius)
circle_line, = ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], 'b-')

# Add the "zero" position marker
zero_point = get_point_at_angle(initial_normal, initial_zero_angle, 0, radius)
zero_marker, = ax.plot([0, zero_point[0]], [0, zero_point[1]], [0, zero_point[2]], 'g-', linewidth=2)
# zero_point_marker, = ax.plot([zero_point[0]], [zero_point[1]], [zero_point[2]], 'go', markersize=8)

# Add the angle position marker
angle_point = get_point_at_angle(initial_normal, initial_zero_angle, initial_angle, radius)
angle_marker, = ax.plot([0, angle_point[0]], [0, angle_point[1]], [0, angle_point[2]], 'r-', linewidth=2)
angle_point_marker, = ax.plot([angle_point[0]], [angle_point[1]], [angle_point[2]], 'ro', markersize=8)

# Add projection onto x-y plane
projection_line, = ax.plot([angle_point[0], angle_point[0]], 
                         [angle_point[1], angle_point[1]], 
                         [angle_point[2], -0.9], 'k:', linewidth=1)
projection_point, = ax.plot([angle_point[0]], [angle_point[1]], [-0.9], 'mo', markersize=6)

# Plot the normal vector
normal_line, = ax.plot([0, initial_normal[0]], [0, initial_normal[1]], [0, initial_normal[2]], 'k-', linewidth=2)

# Set the axis limits
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-1.5, 1.5])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('1DOF Swinging Mass Visualization')

# Draw unit circle on xy plane
theta_xy = np.linspace(0, 2*np.pi, 100)
xy_circle_x = np.cos(theta_xy)
xy_circle_y = np.sin(theta_xy)
xy_circle_z = np.zeros_like(theta_xy)-0.9
ax.plot(xy_circle_x, xy_circle_y, xy_circle_z, 'k--', alpha=0.3)


# Add a cube to show orientation
r = [-1, 1]
x, y = np.meshgrid(r, r)
# Plot the faces of the cube
ax.plot_surface(x, y, np.ones_like(x)*-1, alpha=0.1, color='gray')
ax.plot_surface(x, y, np.ones_like(x), alpha=0.1, color='gray')
ax.plot_surface(x, np.ones_like(x)*-1, y, alpha=0.1, color='gray')
ax.plot_surface(x, np.ones_like(x), y, alpha=0.1, color='gray')
ax.plot_surface(np.ones_like(x)*-1, x, y, alpha=0.1, color='gray')
ax.plot_surface(np.ones_like(x), x, y, alpha=0.1, color='gray')

# draw a plane at z=-1
z = np.ones_like(x)*-1
ax.plot_surface(x, y, z, alpha=0.5, color='gray')

# Add sliders
plt.subplots_adjust(bottom=0.5)

# Normal vector theta slider (elevation angle from z-axis)
ax_theta = plt.axes([0.25, 0.25, 0.5, 0.03])
theta_slider = Slider(
    ax=ax_theta,
    label='Normal θ (elevation)',
    valmin=0,
    valmax=np.pi,
    valinit=0,
    valstep=0.01,
)

# Normal vector phi slider (azimuth angle)
ax_phi = plt.axes([0.25, 0.2, 0.5, 0.03])
phi_slider = Slider(
    ax=ax_phi,
    label='Normal φ (azimuth)',
    valmin=0,
    valmax=2*np.pi,
    valinit=0,
    valstep=0.01,
)

# Zero position angle slider
ax_zero = plt.axes([0.25, 0.15, 0.5, 0.03])
zero_slider = Slider(
    ax=ax_zero,
    label='Zero Position',
    valmin=0,
    valmax=2*np.pi,
    valinit=initial_zero_angle,
    valstep=0.01,
)

# Target direction angle slider
ax_target = plt.axes([0.25, 0.05, 0.5, 0.03])
target_slider = Slider(
    ax=ax_target,
    label='Target Direction Angle',
    valmin=0,
    valmax=2*np.pi,
    valinit=0,
    valstep=0.01,
)

# Add target direction visualization
target_dir = np.array([1, 0, 0])  # Initial direction along x-axis
target_line, = ax.plot([0, target_dir[0]], [0, target_dir[1]], [-0.9, -0.9], 'y-', linewidth=2)

# Update function for the sliders
def update(val):
    # Get slider values
    theta = theta_slider.val
    phi = phi_slider.val
    zero_angle = zero_slider.val
    target_angle = target_slider.val
    
    # Calculate the normal vector from spherical coordinates
    normal = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    
    # Update target direction visualization
    target_dir = np.array([np.cos(target_angle), np.sin(target_angle), 0])
    target_line.set_data([0, target_dir[0]], [0, target_dir[1]])
    target_line.set_3d_properties([-0.9, -0.9])
    
    # Calculate the angle needed for desired projection
    calculated_angle = get_angle_for_projection_direction(normal, zero_angle, target_dir[:2])
    print(calculated_angle)

    rot_matrix = rotation_matrix_from_vectors(np.array([0, 0, 1]), normal)
    unit_u = np.dot(np.array([1,0,0]), rot_matrix.T)
    unit_v = np.dot(np.array([0,1,0]), rot_matrix.T)
    unit_w = np.dot(np.array([0,0,1]), rot_matrix.T)
    analytical_angle = np.arctan2(-unit_u[1]*target_dir[0] + unit_u[0]*target_dir[1], unit_v[1]*target_dir[0]-unit_v[0]*target_dir[1])
    if unit_w[2] < 0:
        analytical_angle += np.pi
    analytical_angle = (analytical_angle - zero_angle)%(2*np.pi)
    print(analytical_angle)
    print("~~~~~~~~~~~")



    
    # Update the circle
    circle_points = create_circle_points(normal, zero_angle, radius)
    circle_line.set_data(circle_points[:, 0], circle_points[:, 1])
    circle_line.set_3d_properties(circle_points[:, 2])
    
    # Update the normal vector line
    normal_line.set_data([0, normal[0]], [0, normal[1]])
    normal_line.set_3d_properties([0, normal[2]])
    
    # Update the zero position marker
    zero_point = get_point_at_angle(normal, zero_angle, 0, radius)
    zero_marker.set_data([0, zero_point[0]], [0, zero_point[1]])
    zero_marker.set_3d_properties([0, zero_point[2]])
    # zero_point_marker.set_data([zero_point[0]], [zero_point[1]])
    # zero_point_marker.set_3d_properties([zero_point[2]])
    
    # Update the angle position marker using calculated angle
    angle_point = get_point_at_angle(normal, zero_angle, analytical_angle, radius)
    angle_marker.set_data([0, angle_point[0]], [0, angle_point[1]])
    angle_marker.set_3d_properties([0, angle_point[2]])
    angle_point_marker.set_data([angle_point[0]], [angle_point[1]])
    angle_point_marker.set_3d_properties([angle_point[2]])
    
    # Update the projection
    projection_line.set_data([angle_point[0], angle_point[0]], [angle_point[1], angle_point[1]])
    projection_line.set_3d_properties([angle_point[2], -0.9])
    projection_point.set_data([angle_point[0]], [angle_point[1]])
    projection_point.set_3d_properties([-0.9])
    
    fig.canvas.draw_idle()

# Connect the update function to all sliders
theta_slider.on_changed(update)
phi_slider.on_changed(update)
zero_slider.on_changed(update)
target_slider.on_changed(update)
# Remove the angle slider since we're now calculating it
# ax_angle.remove()

# Add a reset button
reset_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
reset_button = plt.Button(reset_ax, 'Reset')

def reset(event):
    theta_slider.reset()
    phi_slider.reset()
    zero_slider.reset()
    target_slider.reset()

reset_button.on_clicked(reset)

plt.show()
