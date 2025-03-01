import taichi as ti
import numpy as np
import time

# Initialize Taichi with Metal backend
ti.init(arch=ti.metal)

# Parameters
n_systems = 1000
n_dof = 4
dt = 0.01
steps = 100

# Taichi fields
q = ti.Vector.field(n_dof, dtype=ti.f32, shape=n_systems, needs_grad=True)
q_dot = ti.Vector.field(n_dof, dtype=ti.f32, shape=n_systems, needs_grad=True)
q_ddot = ti.Vector.field(n_dof, dtype=ti.f32, shape=n_systems, needs_grad=True)  # Now differentiable
m = ti.field(dtype=ti.f32, shape=n_systems, needs_grad=True)
h = ti.field(dtype=ti.f32, shape=n_systems)
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

@ti.kernel
def initialize():
    for i in range(n_systems):
        q[i] = ti.Vector([0.0, 0.0, 0.0, 0.0])
        q_dot[i] = ti.Vector([0.1 * (i % 10), 0.2, 0.3, 0.1])
        m[i] = 1.0 + i * 0.001
        h[i] = 0.5 + i * 0.0005

@ti.func
def compute_dynamics(i: ti.i32, t: ti.i32):
    external_force = ti.Vector([ti.sin(t * dt), 0.0, 0.0, 0.0])
    q_ddot[i] = -m[i] * q[i] + q_dot[i] + external_force  # Original dynamics

@ti.kernel
def simulate():
    for t in range(steps):
        for i in range(n_systems):
            compute_dynamics(i, t)
            new_q_dot = q_dot[i] + q_ddot[i] * dt
            new_q = q[i] + new_q_dot * dt
            q_dot[i] = new_q_dot
            q[i] = new_q

@ti.kernel
def compute_loss():
    for i in range(n_systems):
        ti.atomic_add(loss[None], q[i].norm_sqr())  # Loss on final q

# Main simulation with gradients
start_time = time.time()
initialize()

# Clear gradients and initialize loss
q.grad.fill(0)
q_dot.grad.fill(0)
q_ddot.grad.fill(0)  # Clear q_ddot gradients
m.grad.fill(0)
loss[None] = 0.0
loss.grad[None] = 1.0

# Run forward and backward passes
simulate()
compute_loss()
compute_loss.grad()
simulate.grad()

# Retrieve results and gradients
q_np = q.to_numpy()
q_grad_np = q.grad.to_numpy()
q_dot_grad_np = q_dot.grad.to_numpy()
m_grad_np = m.grad.to_numpy()

print("Final positions (first 5 systems):", q_np[:5])
print("Gradient w.r.t. q (first 5 systems):", q_grad_np[:5])
print("Gradient w.r.t. initial q_dot (first 5 systems):", q_dot_grad_np[:5])
print("Gradient w.r.t. m (first 5 systems):", m_grad_np[:5])
print(f"Simulation took {time.time() - start_time:.2f}s")
print("Loss:", loss[None])