import numpy as np
from numpy import sin, cos, tan, pi, arccos, arcsin, sqrt, radians, degrees
from scipy.linalg import expm, logm
import matplotlib.pyplot as plt
from fractions import Fraction

np.set_printoptions(precision=3, suppress=True)

'''     JUST MATH       '''

def to_frac(v):
    fracs = []
    try:
        for val in v:
            fracs.append(Fraction(val).limit_denominator())
    except:
        return Fraction(v).limit_denominator()
    return fracs


def cot(theta):
    return 1/tan(theta)

def cross(a:np.ndarray,b:np.ndarray)->np.ndarray:
    ''' workaround for bug with np.cross() that makes code unreachable'''
    return np.cross(a,b)

def norm(v):
    '''get the norm/magnitude of a vector'''
    return np.sqrt(np.dot(v, v))

def normalize(v):
    '''get the unit vector pointing in the same direction as the vector'''
    v = np.array(v)
    n = norm(v)
    if n > 1e-8:
        return v / norm(v)
    else:
        return np.zeros_like(v)

def ang_betw(v1, v2, axis):
    '''get angle between two 3 vectors. The axis is required for knowing the order'''
    ang = arccos(np.dot(v1, v2)/(norm(v1)*norm(v2)))
    if np.dot(cross(v1,v2), axis) < 0:
        ang *= -1
    return ang

# a = ang_betw((0,1,1), (0,0,-1))
# print(a)

def signed_mod(num, mod):
    return (num + mod/2)%mod - mod/2


'''  ROBOTICS    '''

def skew(w):
    '''
    Get skew symmetric [ω̂] (3x3 matrix) of a rotation axis w (3 vector). \n
    exp([ω̂]θ) is a rotation matrix of θ radians around axis ω. \n
    ṗ = [ω̂]p \n
    p(θ) = exp([ω̂]θ)*p(0)
    '''
    w = np.array(w).flatten()
    skv = np.roll(np.roll(np.diag(w), 1, 1), -1, 0)
    return skv - skv.T

def unskew(W):
    '''
    Get axis w (3 vector) by extracting components out of a skew symmetric [ω̂] (3x3 matrix)
    '''
    w = np.empty(3)
    w[0] = W[2][1]
    w[1] = W[0][2]
    w[2] = W[1][0]
    return w

def get_rotation_from_axis_angle(axis, angle):
    '''
    Rodriques formula for rotations. Get the Rotation (3x3 matrix) by evaluating exp([ω̂]θ).
    axis: [ω̂] (3x3 skew matrix)
    angle: θ  (scalar angle)
    '''
    W = axis
    theta = angle
    R = np.eye(3) + sin(theta)*W + (1 - cos(theta))*(W@W)
    return R

def get_axis_angle_from_rotation(R):
    '''
    Get [ω̂] (3x3 skew matrix) and θ (scalar) from a rotation (3x3 matrix) by taking matrix log.
    '''
    epsilon = 1e-3
    if(np.linalg.norm(R - np.eye(3)) < epsilon):
        theta = 0 #axis W is undefined
        W = np.nan
    elif(np.abs(np.trace(R) + 1) < epsilon):
        theta = pi
        print("there are 6 solutions for W")
        w = 1/sqrt(2*(1+R[2,2])) * np.array([R[0,2], R[1,2], 1+R[2,2]])
        # w = 1/sqrt(2*(1+R[1,1])) * np.array([R[0,1], 1+R[1,1], R[2,1]])
        # w = 1/sqrt(2*(1+R[0,0])) * np.array([1+R[0,0], R[1,0], R[2,0]])
        W = skew(w)
    else:
        theta = arccos((np.trace(R) - 1) / 2)   #(3.54)
        W = (R - R.T)/ (2 * sin(theta))         #(3.53)
    axis = W
    angle = theta
    return axis, angle

# def get_rotation_from_direction(dir):
#     '''
#     PROBABLY DOESNT WORK RN
#     Get the 3x3 rotation matrix that transforms the base coordinates to the orientation
#     of a direction vector. The magnitude of the vector doesn't matter.
#     '''
#     up = np.array([0, 0, 1])
#     dir = normalize(dir)
    
#     xaxis = cross(up, dir)
#     xaxis = normalize(xaxis)

#     yaxis = cross(dir, xaxis);
#     yaxis = normalize(yaxis)

#     R = np.array([
#         xaxis,
#         yaxis,
#         dir
#     ])
#     return R

def rot_xyz(axis='z', theta=pi/2):
    '''
    Get the rotation matrix for a rotation of angle theta along the x, y, or z axes.
    '''
    assert axis in ['x', 'y', 'z']
    if(axis == 'x'):
        return np.array([
            [1, 0, 0],
            [0, cos(theta), -sin(theta)],
            [0, sin(theta), cos(theta)],
        ])
    if(axis == 'y'):
        return np.array([
            [cos(theta), 0, sin(theta)],
            [0, 1, 0],
            [-sin(theta), 0, cos(theta)],
        ])
    if(axis == 'z'):
        return np.array([
            [cos(theta), -sin(theta), 0],
            [sin(theta), cos(theta), 0],
            [0, 0, 1]
        ])
    
import sympy as sp
def rot_xyz_sympy(axis='z', theta=pi/2):
    '''
    Get the rotation matrix for a rotation of angle theta along the x, y, or z axes.
    '''
    assert axis in ['x', 'y', 'z']
    if(axis == 'x'):
        return np.array([
            [1, 0, 0],
            [0, sp.cos(theta), -sp.sin(theta)],
            [0, sp.sin(theta), sp.cos(theta)],
        ])
    if(axis == 'y'):
        return np.array([
            [sp.cos(theta), 0, sp.sin(theta)],
            [0, 1, 0],
            [-sp.sin(theta), 0, sp.cos(theta)],
        ])
    if(axis == 'z'):
        return np.array([
            [sp.cos(theta), -sp.sin(theta), 0],
            [sp.sin(theta), sp.cos(theta), 0],
            [0, 0, 1]
        ])




def format_transform(R, p):
    '''
    Format a transformation T (4x4 matrix) from rotation R (3x3 matrix) and position (3 vector).\n
    T = [R p]
        [0 1]
    (Modern Robotics 3.62)
    '''
    p = np.reshape(p, (3, 1))
    T = np.hstack((R, p))
    T = np.vstack((T, np.array([0, 0, 0, 1])))
    return T

def extract_R_p_from_transformation(T):
    '''
    Extract rotation R (3x3 matrix) and displacement p (3 vector) from a transformation (4x4 matrix)
    '''
    T = np.reshape(T, (4, 4))
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    return R, p

def transform_point(T, x):
    '''
    Evaluates Tx = Rx + p = T@[x,1], where T = [R, p], a rotation R and displacement p. Left multiplication means rotate then displace.
    Appending a 1 to the 3 vector puts it in homogenous coordinates so it is compatible with transformation matrices.
    Will truncate the last 1 and output the transformed 3 vector.
    T: 4x4 transformation matrix
    x: 3x3 vector position in space
    (Modern Robotics 3.65)
    '''
    x = np.array(x).flatten()
    x_homogeneous = np.concatenate((x, [1]))
    Tx_homogenous = T @ x_homogeneous
    return Tx_homogenous[0:-1]

def adjoint(T):
    '''
    Get [Adj_Tsb] (6x6 matrix) which converts twists in the b frame to twists in the s frame (as 6 vectors): twist_s = [Adj_Tsb] @ twist_b  
    (Modern Robotics 3.83) \n
    Also works for converting screws from one frame to another.
    '''
    T = np.array(T).reshape((4, 4))
    R, p = extract_R_p_from_transformation(T)
    P = skew(p)
    top_3_rows = np.hstack((R, np.zeros((3,3))))
    low_3_rows = np.hstack((P@R, R))
    Adj_Tsb = np.vstack((top_3_rows, low_3_rows))
    return Adj_Tsb

def format_twist_matrix(w, v):
    '''
    Format a twist [V] (4x4 matrix) from angular velocity w (3 vector) and velocity (3 vector).\n
    w is the axis of rotation scaled by angular velocity.
    w and v can be in body or space frame, and the screw will be in the corresponding frame.
    [V] = [[ω] v]
          [ 0  0]
    (Modern Robotics 3.81) \n
    Use the exact same formula for representing the screw axis, because a screw is a normalized twist, where 
    |ω| = 1 or (|ω|=0 and |v|=1)
    '''
    W = skew(w)
    v = np.reshape(v, (3, 1))
    Twist = np.hstack((W, v))
    Twist = np.vstack((Twist, np.array([0, 0, 0, 0])))
    return Twist

def extract_twist_vector_from_matrix(Twist):
    '''
    Extract the twist vector (6 vector) consisting of [ω v] where ω and v are angular and linear velocity (both 3 vectors).  
    '''
    Twist = np.reshape(Twist, (4, 4))
    W = Twist[0:3,0:3]
    w = unskew(W)
    v = Twist[0:3, 3]
    twist = np.concatenate((w, v))
    return twist


def get_twist_from_screw_qsh(thetadot, q, s, h):
    '''
    (Modern Robotics 3.3.2.2)
    Just as an angular velocity ω can be viewed as ω̂θ̇, where ω̂ is the unit rotation
    axis and θ̇ is the rate of rotation about that axis, a twist can be interpreted
    in terms of a screw axis S and a velocity θ̇ about the screw axis.
    thetadot: angular velocity (scalar)
    q: any point on the screw axis (3 vector)
    s: unit vector in direction of screw axis (3 vector)
    h: screw pitch which is linear velocity/angular velocity (scalar)
    '''
    s = s / np.linalg.norm(s) #normalize just in case
    w = s*thetadot #rotation axis scaled by angular velocity
    v = cross(-s*thetadot, q) + h*s*thetadot
    twist = np.concatenate((w, v))
    return twist


def get_transformation_from_screw_axis_angle(screw_axis, angle):
    '''
    Evaluate exp([S]θ) to get the transformation of following the screw axis for θ distance
    screw_axis: 6 vector consisting of [ω v], where ω is a unit vector or (ω=0 and |v|=1)
    angle: scalar 
    (Modern Robotics 3.88)
    '''
    screw_axis = np.array(screw_axis).flatten()
    W = skew(screw_axis[0:3])
    v = screw_axis[3:6]
    theta = angle
    R = get_rotation_from_axis_angle(W, theta)
    I = np.eye(3)
    topright = (I*theta + (1 - cos(theta))*W + (theta - sin(theta))*(W@W)) @ v
    topright = topright.reshape((3, 1))
    toprow = np.hstack((R, topright))
    lowrow = np.array([0, 0, 0, 1])
    T = np.vstack((toprow, lowrow))
    return T

def get_screw_axis_angle_from_transformation(T):
    '''
    Matrix log for a rigid motion: get the screw axis and angle for transformation T containing rotation R and displacement p
    (Modern Robotics eq 3.92)
    '''
    T = np.array(T).reshape((4,4))
    R, p = extract_R_p_from_transformation(T)
    W, theta = get_axis_angle_from_rotation(R)
    G_inv = (1/theta)*np.eye(3) - (1/2)*W + (1/theta - 1/2*cot(theta/2))*(W@W)
    v = G_inv @ p
    w = unskew(W)
    S = format_twist_matrix(w, v)
    return S, theta



def sphere_intersect(line_origin, line_dir, sphere_center, sphere_radius):
    '''
    https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
    '''
    o = np.array(line_origin)
    u = normalize(line_dir)
    c = np.array(sphere_center)
    r = sphere_radius

    determinant = np.dot(u, o-c)**2 - (np.dot(o-c,o-c) - r**2)
    if(determinant < 0):
        print(f"No solution found: determinant={determinant}")
        return 0
    else:
        dist_from_line_origin = -np.dot(u, o-c) + np.sqrt(determinant)
        intersect_pt = o + u*dist_from_line_origin
        return intersect_pt, dist_from_line_origin

'''uses law of cosines to get an angle C of a triangle that has known sides a, b, c'''
def get_angle_of_tri(a, b, c):
    return np.arccos((a**2 + b**2 - c**2)/(2*a*b))

def spherical_fwdk(thetas, order=['x','y','z']):
    '''
    Get the orientation after being transformed by a spherical wrist,
    where thetas are the angles for rotation in the xyz axes in the specified order
    from the base link to farther out in the chain.
    '''
    R = np.eye(3)
    for i in range(3):
        R = R @ rot_xyz(order[i], thetas[i])
    return R

def spherical_invk(R):
    '''
    Find the thetas 1-3 of a spherical wrist to create a desired orientation R
    First convert the yxz frame to zyx frame
    https://www.slideserve.com/marva/ch-3-inverse-kinematics-ch-4-velocity-kinematics
    '''
    shift = np.array([ #convert to whatever frame harvard uses
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ])
    R = shift@R

    s5 = np.sqrt(1 - R[2,2]**2)
    
    #first solution s5 > 0:
    theta4 = np.arctan2(R[0,2], R[1,2]) - pi/2
    theta5 = np.arctan2(R[2,2], s5)
    theta6 = np.arctan2(-R[2,0], R[2,1])
    sol1 = [theta4, theta5, theta6]
    sol1 = [-signed_mod(ang, 2*pi) for ang in sol1]

    #first solution s5 < 0:
    theta4 = np.arctan2(-R[0,2], -R[1,2]) - pi/2
    theta5 = np.arctan2(R[2,2], -s5)
    theta6 = np.arctan2(R[2,0], -R[2,1])
    sol2 = [theta4, theta5, theta6]
    sol2 = [-signed_mod(ang, 2*pi) for ang in sol2]


    return sol1, sol2

