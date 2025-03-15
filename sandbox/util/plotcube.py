import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from util import *

def plot_cube(ax, cube_definition):
    cube_definition_array = [
        np.array(list(item))
        for item in cube_definition
    ]

    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0]
    ]

    points += [cube_definition_array[0] + vectors[0] + vectors[1]]
    points += [cube_definition_array[0] + vectors[0] + vectors[2]]
    points += [cube_definition_array[0] + vectors[1] + vectors[2]]
    points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]]
    ]

    

    faces = Poly3DCollection(edges, linewidths=1, edgecolors='k')
    faces.set_facecolor((0,0,1,0.1))

    ax.add_collection3d(faces)

    # Plot the points themselves to force the scaling of the axes
    ax.scatter(points[:,0], points[:,1], points[:,2], s=0)

    ax.set_box_aspect([1,1,1])

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# cube_definition = [
#     (0,0,0), (0,1,0), (1,0,0), (0,0,1)
# ]
# plot_cube(ax, cube_definition)
# plt.show()

def plot_link(ax, R, p, size, color=(0,0,1,0.5), long_ax='z'):
    '''
    Plot a rectangular prism of rotation R and position p on the axis.
    size specifies how wide the prism is in the x, y, and z directions in the base frame.
    '''

    #shift so that the long axis is the one that is in the z spot
    if long_ax == 'y':
        size = np.roll(size, 1)
    if long_ax == 'x':
        size = np.roll(size, 2)

    (wx, wy, wz) = size

    verts = []
    for dx in (-wx, wx):
        for dy in (-wy, wy):
            verts.append(np.array([dx, dy*np.sign(dx), 0])) #ugh the vertices need to be in the right order
    for i in range(4):
        verts.append(verts[i] + np.array([0, 0, wz]))
    verts = np.array(verts)

    for i in range(len(verts)):

        # shift x, y, and z back to normal
        if long_ax == 'y':
            verts[i] = np.roll(verts[i], -1)
        if long_ax == 'x':
            verts[i] = np.roll(verts[i], -2)

        verts[i] = R@verts[i] + p


    bottom = verts[0:4]
    top = verts[4:8]
    edges = [bottom, top]
    for i in range(4):
        edges.append([verts[i],verts[(1+i)%4],verts[(1+i)%4+4],verts[i+4]])

    faces = Poly3DCollection(edges, zsort='min', linewidths=1, edgecolors='k')
    faces.set_facecolor(color)

    ax.add_collection3d(faces)
    ax.set_box_aspect([1,1,1])

def plot_arrow(ax, start, dir, color='red', alpha=0.8, arrow_ratio=0.2):
    ax.quiver(
        start[0], start[1], start[2], # <-- starting point of vector
        dir[0], dir[1], dir[2], # <-- directions of vector
        color = color, alpha = alpha,
        arrow_length_ratio=arrow_ratio
    )

def plot_frame(ax, R, p, lengths=0.05):
    plot_arrow(ax, p, R@np.array([lengths, 0, 0]), color='red')
    plot_arrow(ax, p, R@np.array([0, lengths, 0]), color='green')
    plot_arrow(ax, p, R@np.array([0, 0, lengths]), color='blue')

def plot_HTM(ax, HTM, lengths=0.05):
    R, p = extract_R_p_from_transformation(HTM)
    plot_frame(ax, R, p, lengths)

def plot_origin(ax, lengths=0.05):
    plot_frame(ax, R=np.eye(3), p=np.zeros(3), lengths=lengths)

def plot_point(ax, pt):
    pt = np.array(pt).flatten()
    ax.scatter3D(pt[0], pt[1], pt[2])

def init_3d_plot(size=(8,8), cube_lim=0.25):
    fig = plt.figure(figsize=size)
    fig.tight_layout()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-cube_lim, cube_lim)
    ax.set_ylim(-cube_lim, cube_lim)
    ax.set_zlim(-cube_lim, cube_lim)
    plot_origin(ax)
    ax.scatter3D(0, 0, 0, c='orange')
    ax.set_box_aspect([1,1,1])
    return fig, ax

def draw_labels(ax, cube_lim=0.25):
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plot_origin(ax)
    ax.scatter3D(0, 0, 0, c='orange')
    ax.set_box_aspect([1,1,1])
    ax.set_xlim(-cube_lim, cube_lim)
    ax.set_ylim(-cube_lim, cube_lim)
    ax.set_zlim(-cube_lim, cube_lim)
