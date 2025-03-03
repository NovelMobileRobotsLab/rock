''' Gradients do work because global fields are parameterized by time.'''


import taichi as ti
import numpy as np
import time
import matplotlib.pyplot as plt

import os
this_dir = os.path.dirname(os.path.abspath(__file__))


# Initialize Taichi with Metal backend
# ti.init(arch=ti.gpu)
ti.init(arch=ti.cpu, cpu_max_num_threads=1)
# ti.init(arch=ti.cpu)

# Parameters
n_systems = 1000
n_dof = 4
dt = 0.01
steps = 1000

# Taichi fields
q = ti.Vector.field(n_dof, dtype=ti.f32, shape=(steps, n_systems), needs_grad=True)  # Add needs_grad for debugging
q_dot = ti.Vector.field(n_dof, dtype=ti.f32, shape=(steps, n_systems), needs_grad=True)
q_ddot = ti.Vector.field(n_dof, dtype=ti.f32, shape=(steps, n_systems), needs_grad=True)
m = ti.field(dtype=ti.f32, shape=n_systems, needs_grad=True)
M = ti.Matrix.field(n_dof, n_dof, dtype=ti.f32, shape=n_systems, needs_grad=True)
C = ti.Matrix.field(n_dof, n_dof, dtype=ti.f32, shape=n_systems, needs_grad=True)
G = ti.Vector.field(n_dof, dtype=ti.f32, shape=n_systems, needs_grad=True)
tau = ti.Vector.field(n_dof, dtype=ti.f32, shape=(steps, n_systems), needs_grad=True)
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

@ti.kernel
def initialize():
    for i in range(n_systems):
        for t in range(steps):
            q[t, i] = ti.Vector([0.1 * (i % 3), 0.1 * (i % 10), 0.1 * (i % 23), 0.1 * (i % 31)])
            q_dot[t, i] = ti.Vector([0.1 * (i % 10), 0.2, 0.3, 0.1])
            m[i] = 1.0 + i * 0.001

            

@ti.kernel
def simulate():
    for i in range(n_systems):
        for t in range(steps-1):
            ti.atomic_add(loss[None], q[t, i].norm_sqr())

            M[i] = ti.Matrix.identity(ti.f32, n_dof) * m[i]
            C[i] = ti.Matrix.zero(ti.f32, n_dof, n_dof)
            G[i] = ti.Vector([0.0, 0.0, 0.0, 0.0])

            tau[t+1,i] = -q[t, i]


            q_ddot[t+1, i] = M[i].inverse() @ (tau[t+1, i] - C[i] @ q_dot[t, i] - G[i])
            q_dot[t+1, i] = q_dot[t, i] + q_ddot[t, i] * dt
            q[t+1, i] = q[t, i] + q_dot[t, i] * dt

            

# Main simulation with gradients
start_time = time.time()
initialize()

# Clear gradients and initialize loss
q.grad.fill(0)  # For debugging
q_dot.grad.fill(0)
m.grad.fill(0)
loss[None] = 0.0
loss.grad[None] = 1.0


# Run forward and backward passes
simulate()
simulate.grad()

# Retrieve results and gradients
q_np = q.to_numpy()
q_grad_np = q.grad.to_numpy()  # For debugging
q_dot_grad_np = q_dot.grad.to_numpy()
m_grad_np = m.grad.to_numpy()


print("Final positions (first 5 systems):", q_np[-1,:5])
print("Gradient w.r.t. q (first 5 systems):", q_grad_np[0,:5])  # Debugging
print("Gradient w.r.t. initial q_dot (first 5 systems):", q_dot_grad_np[0,:5])
print("Gradient w.r.t. m (first 5 systems):", m_grad_np[:5])
print(f"Simulation took {time.time() - start_time:.2f}s")
print("Loss:", loss[None])
print("shape", q_np.shape)


# Create plots for the first 5 systems
plt.figure(figsize=(10, 8))
for dof in range(n_dof):
    plt.subplot(2, 2, dof + 1)
    for system in range(10):
        plt.plot(np.arange(steps) * dt, q_np[:, system, dof], label=f'System {system}')
    plt.title(f'DOF {dof} Trajectory')
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.legend()
plt.tight_layout()
plt.savefig(f'{this_dir}/trajectory3.png')
