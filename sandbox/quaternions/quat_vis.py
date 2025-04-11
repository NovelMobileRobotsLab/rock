
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

# Function to convert quaternion to rotation matrix
def quaternion_to_rotation_matrix(q):
    # Normalize quaternion
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    
    # Create rotation matrix from quaternion
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    
    return R

# Set up the figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Initial quaternion [w, x, y, z]
q_initial = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion

# Create initial coordinate vectors (standard basis)
origin = np.array([0, 0, 0])
x_axis = np.array([1, 0, 0])
y_axis = np.array([0, 1, 0])
z_axis = np.array([0, 0, 1])

# Initialize quiver objects as global variables
quiver_x = None
quiver_y = None
quiver_z = None

# Add text showing quaternion values
q_text = ax.text2D(0.05, 0.95, f"q = [{q_initial[0]:.2f}, {q_initial[1]:.2f}, {q_initial[2]:.2f}, {q_initial[3]:.2f}]", 
                   transform=ax.transAxes)

# Flag to prevent recursive updates
updating = False

# Function to update the plot with a new quaternion
def update_plot(q):
    global quiver_x, quiver_y, quiver_z
    
    # Normalize the quaternion
    q_norm = np.linalg.norm(q)
    if q_norm > 0:
        q = q / q_norm
    
    # Calculate rotation matrix and rotated axes
    R = quaternion_to_rotation_matrix(q)
    rotated_x = R @ x_axis
    rotated_y = R @ y_axis
    rotated_z = R @ z_axis
    
    # Remove old quivers if they exist
    if quiver_x:
        quiver_x.remove()
    if quiver_y:
        quiver_y.remove()
    if quiver_z:
        quiver_z.remove()
    
    # Create updated quiver plots
    quiver_x = ax.quiver(origin[0], origin[1], origin[2], rotated_x[0], rotated_x[1], rotated_x[2], 
                         color='red', linewidth=2, arrow_length_ratio=0.15)
    quiver_y = ax.quiver(origin[0], origin[1], origin[2], rotated_y[0], rotated_y[1], rotated_y[2], 
                         color='green', linewidth=2, arrow_length_ratio=0.15)
    quiver_z = ax.quiver(origin[0], origin[1], origin[2], rotated_z[0], rotated_z[1], rotated_z[2], 
                         color='blue', linewidth=2, arrow_length_ratio=0.15)
    
    # Update quaternion text display
    q_text.set_text(f"q = [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]")
    
    fig.canvas.draw_idle()
    return q

# Initial plot
q_initial = update_plot(q_initial)

# Setting equal aspect ratio and labels
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-1.5, 1.5])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Quaternion Orientation Visualizer')

# Add reference coordinate frame at origin (thinner lines)
ax.quiver(origin[0], origin[1], origin[2], 1, 0, 0, color='darkred', alpha=0.3, linewidth=1, arrow_length_ratio=0.1)
ax.quiver(origin[0], origin[1], origin[2], 0, 1, 0, color='darkgreen', alpha=0.3, linewidth=1, arrow_length_ratio=0.1)
ax.quiver(origin[0], origin[1], origin[2], 0, 0, 1, color='darkblue', alpha=0.3, linewidth=1, arrow_length_ratio=0.1)

# Add some grid lines
ax.grid(True)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Add text labels for the axes
ax.text(1.1, 0, 0, "X", color='darkred')
ax.text(0, 1.1, 0, "Y", color='darkgreen')
ax.text(0, 0, 1.1, "Z", color='darkblue')

# Add sliders for quaternion components
plt.subplots_adjust(bottom=0.35)  # Make room for sliders

# Slider positions and properties
slider_w = plt.axes([0.15, 0.25, 0.65, 0.03])
slider_x = plt.axes([0.15, 0.20, 0.65, 0.03])
slider_y = plt.axes([0.15, 0.15, 0.65, 0.03])
slider_z = plt.axes([0.15, 0.10, 0.65, 0.03])

# Create the sliders with initial values
s_w = Slider(slider_w, 'w', -1.0, 1.0, valinit=q_initial[0])
s_x = Slider(slider_x, 'i', -1.0, 1.0, valinit=q_initial[1])
s_y = Slider(slider_y, 'j', -1.0, 1.0, valinit=q_initial[2])
s_z = Slider(slider_z, 'k', -1.0, 1.0, valinit=q_initial[3])

# Update function for sliders - fixed to avoid recursion
def update(val=None):
    # Get values from sliders
    q = np.array([s_w.val, s_x.val, s_y.val, s_z.val])
    update_plot(q)

# Connect update function to sliders
s_w.on_changed(update)
s_x.on_changed(update)
s_y.on_changed(update)
s_z.on_changed(update)

# Add normalization button
norm_button_ax = plt.axes([0.4, 0.02, 0.2, 0.04])
norm_button = plt.Button(norm_button_ax, 'Normalize')

def normalize_quaternion(event):
    q = np.array([s_w.val, s_x.val, s_y.val, s_z.val])
    q_norm = np.linalg.norm(q)
    
    if q_norm > 0:
        q = q / q_norm
        
        # Temporarily disconnect callbacks to avoid recursive updates
        s_w.disconnect(update)
        s_x.disconnect(update)
        s_y.disconnect(update)
        s_z.disconnect(update)
        
        # Set new values
        s_w.set_val(q[0])
        s_x.set_val(q[1])
        s_y.set_val(q[2])
        s_z.set_val(q[3])
        
        # Reconnect callbacks
        s_w.on_changed(update)
        s_x.on_changed(update)
        s_y.on_changed(update)
        s_z.on_changed(update)
        
        # Update the plot
        update_plot(q)

norm_button.on_clicked(normalize_quaternion)

# Reset button
reset_button_ax = plt.axes([0.15, 0.02, 0.2, 0.04])
reset_button = plt.Button(reset_button_ax, 'Reset')

def reset_quaternion(event):
    # Temporarily disconnect callbacks to avoid recursive updates
    s_w.disconnect(update)
    s_x.disconnect(update)
    s_y.disconnect(update)
    s_z.disconnect(update)
    
    # Set new values
    s_w.set_val(1.0)
    s_x.set_val(0.0)
    s_y.set_val(0.0)
    s_z.set_val(0.0)
    
    # Reconnect callbacks
    s_w.on_changed(update)
    s_x.on_changed(update)
    s_y.on_changed(update)
    s_z.on_changed(update)
    
    # Update the plot
    update_plot(np.array([1.0, 0.0, 0.0, 0.0]))

reset_button.on_clicked(reset_quaternion)

plt.show()