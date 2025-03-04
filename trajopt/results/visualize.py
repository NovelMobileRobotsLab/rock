from numpy import sin, cos
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import imageio
import os

[h, r, m, g, h2,  Isx, Isy, Isz,  Iax, Iay, Iaz] = [0.1, 0.05, 0.5, 9.81, -0.01,  9E-4, 9E-4, 25E-4,  4E-9, 9E-9, 9E-9]

def get_pts(q):
    yaw, pit, rol, q4 = q
    pts = np.zeros((6, 3))

    # 124 operations
    x0 = sin(rol)
    x1 = sin(yaw)
    x2 = x0*x1
    x3 = h*x2
    x4 = sin(pit)
    x5 = cos(rol)
    x6 = cos(yaw)
    x7 = x5*x6
    x8 = x4*x7
    x9 = h*x8
    x10 = x3 + x9
    x11 = x0*x6
    x12 = h*x11
    x13 = x1*x5
    x14 = x13*x4
    x15 = h*x14
    x16 = -x12 + x15
    x17 = cos(pit)
    x18 = x17*x5
    x19 = h*x18
    x20 = r*x17
    x21 = 0.29999999999999999*x20
    x22 = r*x4
    x23 = 1.0/(m + 1)
    x24 = h2*m
    x25 = cos(q4)
    x26 = x20*x25
    x27 = x26*x6
    x28 = sin(q4)
    x29 = r*x28
    x30 = m*x29
    x31 = m*x22*x28
    x32 = x1*x26
    x33 = h2*x18
    x34 = x22*x25
    x35 = x0*x17*x29
    pts[0,0] = 0
    pts[0,1] = 0
    pts[0,2] = 0
    pts[1,0] = x10
    pts[1,1] = x16
    pts[1,2] = x19
    pts[2,0] = x10 + x21*x6
    pts[2,1] = x1*x21 + x16
    pts[2,2] = x19 - 0.29999999999999999*x22
    pts[3,0] = x10
    pts[3,1] = x16
    pts[3,2] = x19
    pts[4,0] = x23*(m*x27 + m*x3 + m*x9 + x10 + x11*x31 - x13*x30 + x2*x24 + x24*x8)
    pts[4,1] = x23*(-m*x12 + m*x15 + m*x32 - x11*x24 + x14*x24 + x16 + x2*x31 + x30*x7)
    pts[4,2] = x23*(m*x19 + m*x33 - m*x34 + m*x35 + x19)
    pts[5,0] = h2*(x2 + x8) + x10 + x27 + x29*(x11*x4 - x13)
    pts[5,1] = h2*(-x11 + x14) + x16 + x29*(x2*x4 + x7) + x32
    pts[5,2] = x19 + x33 - x34 + x35

    return pts

def draw_link(start, end):
    """Draw a cylindrical link between two points"""
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return
    
    direction = direction / length
    
    # Create a rotation matrix to align cylinder with direction
    up = np.array([0., 0., 1.])
    if np.allclose(direction, up) or np.allclose(direction, -up):
        rotation_axis = np.array([1., 0., 0.])
    else:
        rotation_axis = np.cross(up, direction)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    angle = np.arccos(np.dot(up, direction))
    
    glPushMatrix()
    glTranslatef(start[0], start[1], start[2])
    glRotatef(angle * 180 / np.pi, rotation_axis[0], rotation_axis[1], rotation_axis[2])
    
    # Draw rectangular cylinder
    radius = 0.05  # Adjust for desired thickness
    slices = 16
    quad = gluNewQuadric()
    gluCylinder(quad, radius, radius, length, slices, 1)

    # Draw end caps in green
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, (0.0, 1.0, 0.0, 1.0))
    
    # Draw cap at start
    gluDisk(quad, 0, radius, slices, 1)
    
    # Draw cap at end
    glTranslatef(0, 0, length)
    gluDisk(quad, 0, radius, slices, 1)
    
    # Reset material color
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, (0.5, 0.5, 1.0, 1.0))
    
    glPopMatrix()

def main():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load trajectory data
    q = np.load(f'{this_dir}/q.npy')
    q_dot = np.load(f'{this_dir}/q_dot.npy')
    
    num_steps, num_systems, _ = q.shape
    
    # Get points for system 0
    system_pts = np.zeros((num_steps, 6, 3))
    for t in range(num_steps):
        system_pts[t] = get_pts(q[t, 0])
    
    # Calculate bounds for camera setup
    x_min, x_max = system_pts[:,:,0].min(), system_pts[:,:,0].max()
    y_min, y_max = system_pts[:,:,1].min(), system_pts[:,:,1].max()
    z_min, z_max = system_pts[:,:,2].min(), system_pts[:,:,2].max()
    
    center = np.array([(x_min + x_max)/2, (y_min + y_max)/2, (z_min + z_max)/2])
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    
    # Initialize Pygame and OpenGL
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    
    # Setup camera
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)

    # Set background color to light gray
    glClearColor(0.9, 0.9, 0.9, 1.0)
    
    # Increase ambient light for brighter overall scene
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (0.5, 0.5, 0.5, 1.0))
    
    # Make directional light brighter
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
    glLightfv(GL_LIGHT0, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
    glLightfv(GL_LIGHT0, GL_POSITION, (0, 0, 10, 1))
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, (0.5, 0.5, 1.0, 1.0))
    
    camera_distance = max_range * 1
    # Set camera position on x=y line
    camera_x = 0.01
    camera_y = 0.01
    camera_z = 0.01
    
    # Calculate look-at vector (pointing to origin)
    look_at = np.array([0, 0, 0]) - np.array([camera_x, camera_y, camera_z])
    look_at = look_at / np.linalg.norm(look_at)
    
    # Calculate up vector (vertical in world space)
    up = np.array([0, 0, 1])
    
    # Set up view matrix
    gluLookAt(camera_x, camera_y, camera_z,  # Camera position
              0, 0, 0,                        # Look at origin
              0, 0, 1)                        # Up vector
    
    # Set up perspective projection with narrower field of view for more zoom
    glTranslatef(0, 0, 0)  # Move camera back slightly to keep scene in view
    # gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    # glTranslatef(-center[0], -center[1], -camera_distance)
    
    frames = []
    for frame in range(0, num_steps, 10):  # Skip frames for smoother animation
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Draw links
        pts = system_pts[frame] * 8
        for i in range(len(pts)-1):
            draw_link(pts[i], pts[i+1])
            
        # Draw points
        glPointSize(5.0)
        glBegin(GL_POINTS)
        for pt in pts:
            glVertex3fv(pt)
        glEnd()

        # Draw coordinate axes at origin
        glLineWidth(2.0)
        glBegin(GL_LINES)
        # X axis - Red
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, (1.0, 0.0, 0.0, 1.0))
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.2, 0.0, 0.0)
        # Y axis - Green  
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, (0.0, 1.0, 0.0, 1.0))
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.2, 0.0)
        # Z axis - Blue
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, (0.0, 0.0, 1.0, 1.0))
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.2)
        glEnd()
        
        # Reset material color
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, (0.5, 0.5, 1.0, 1.0))
        
        pygame.display.flip()
        
        # Capture frame
        data = glReadPixels(0, 0, display[0], display[1], GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(data, dtype=np.uint8).reshape(display[1], display[0], 3)
        image = np.flipud(image)  # Flip because OpenGL uses bottom-left origin
        frames.append(image)
        
        pygame.time.wait(10)
    
    # Save animation
    with imageio.get_writer(f'{this_dir}/trajectory_animation.mp4', fps=30) as writer:
        for frame in frames:
            writer.append_data(frame)
    
    pygame.quit()

if __name__ == "__main__":
    main()
