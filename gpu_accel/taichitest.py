import taichi as ti
import numpy as np
import time

# Initialize Taichi with Metal backend
ti.init(arch=ti.metal)

# Parameters
n_systems = 1000
n_dof = 4
dt = 0.01

# Taichi fields
q = ti.Vector.field(n_dof, dtype=ti.f32, shape=n_systems)
q_dot = ti.Vector.field(n_dof, dtype=ti.f32, shape=n_systems)
q_ddot = ti.Vector.field(n_dof, dtype=ti.f32, shape=n_systems)
m = ti.field(dtype=ti.f32, shape=n_systems)
h = ti.field(dtype=ti.f32, shape=n_systems)
M = ti.Matrix.field(n_dof, n_dof, dtype=ti.f32, shape=n_systems)
C = ti.Matrix.field(n_dof, n_dof, dtype=ti.f32, shape=n_systems)
G = ti.Vector.field(n_dof, dtype=ti.f32, shape=n_systems)
tau = ti.Vector.field(n_dof, dtype=ti.f32, shape=n_systems)

@ti.kernel
def initialize():
    for i in range(n_systems):
        q[i] = ti.Vector([0.0, 0.0, 0.0, 0.0])
        q_dot[i] = ti.Vector([0.1, 0.1, 0.1, 0.1])
        m[i] = 1.0 + i * 0.001
        h[i] = 0.5 + i * 0.0005

@ti.func
def compute_dynamics(i: ti.i32, t: ti.i32):  # Add type annotations
    # Example: Construct M, C, G, tau based on q[i], q_dot[i], and parameters
    M[i] = ti.Matrix.identity(ti.f32, n_dof) * m[i]  # Mass matrix
    C[i] = ti.Matrix.zero(ti.f32, n_dof, n_dof)  # Zero Coriolis for simplicity
    G[i] = ti.Vector([0.0, 0.0, 0.0, 9.81 * m[i]])  # Gravity term
    # Avoid division by zero at t=0, shift t or handle explicitly
    tau[i] = ti.Vector([1.0, 0.0, 0.0, 0.0]) / ti.cast(t + 1, ti.f32)  # Shift t to avoid t=0
    
    # Compute acceleration: q_ddot = M^-1 * (tau - C * q_dot - G)
    M_inv = M[i].inverse()
    q_ddot[i] = M_inv @ (tau[i] - C[i] @ q_dot[i] - G[i])

@ti.kernel
def step(t: ti.i32):  # Add type annotation
    for i in range(n_systems):
        compute_dynamics(i, t)
        q_dot[i] = q_dot[i] + q_ddot[i] * dt
        q[i] = q[i] + q_dot[i] * dt

# Main simulation loop
start_time = time.time()
initialize()
steps = 1000
for t in range(steps):
    step(t)

# Retrieve results
q_np = q.to_numpy()
print("Final positions (first 5 systems):", q_np[:5])

# Timing
print(f"Simulation took {time.time() - start_time:.2f}s")