import numpy as np
import time
import matplotlib.pyplot as plt

# Parameters
n_systems = 1000
n_dof = 4
dt = 0.01
steps = 1000

# NumPy arrays instead of Taichi fields
q = np.zeros((n_systems, n_dof))
q_dot = np.zeros((n_systems, n_dof))
q_ddot = np.zeros((n_systems, n_dof))
m = np.zeros(n_systems)
M = np.zeros((n_systems, n_dof, n_dof))
C = np.zeros((n_systems, n_dof, n_dof))
G = np.zeros((n_systems, n_dof))
tau = np.zeros((n_systems, n_dof))
loss = 0.0

# Store trajectory for each system at each timestep
q_history = np.zeros((steps, n_systems, n_dof))

def initialize():
    for i in range(n_systems):
        q[i] = np.array([0.1 * (i % 3), 0.1 * (i % 10), 0.1 * (i % 23), 0.1 * (i % 31)])
        q_dot[i] = np.array([0.1 * (i % 10), 0.2, 0.3, 0.1])
        m[i] = 1.0 + i * 0.001

def compute_dynamics(i, t):
    M[i] = np.eye(n_dof) * m[i]
    C[i] = np.zeros((n_dof, n_dof))
    G[i] = np.array([0.0, 0.0, 0.0, 9.81 * m[i]])
    tau[i] = np.array([1.0, 1.0, 1.0, 1.0])
    M_inv = np.linalg.inv(M[i])
    q_ddot[i] = M_inv @ (tau[i] - C[i] @ q_dot[i] - G[i])

def simulate():
    global loss
    for t in range(steps):
        for i in range(n_systems):
            compute_dynamics(i, t)
            new_q = q[i] + q_dot[i] * dt
            new_q_dot = q_dot[i] + q_ddot[i] * dt
            
            q_dot[i] = new_q_dot
            q[i] = new_q
            q_history[t, i] = q[i]
            loss += np.sum(q[i] ** 2)  # norm_sqr equivalent

# Main simulation
start_time = time.time()
initialize()

# Initialize arrays for gradients (though they won't be computed automatically like in Taichi)
q_grad = np.zeros_like(q)
q_dot_grad = np.zeros_like(q_dot)
m_grad = np.zeros_like(m)
loss = 0.0

# Run simulation
simulate()

# Note: The automatic gradient computation is removed since NumPy doesn't have automatic differentiation

print("Final positions (first 5 systems):", q[:5])
print("Gradient w.r.t. q (first 5 systems):", q_grad[:5])  # Will be all zeros
print("Gradient w.r.t. initial q_dot (first 5 systems):", q_dot_grad[:5])  # Will be all zeros
print("Gradient w.r.t. m (first 5 systems):", m_grad[:5])  # Will be all zeros
print(f"Simulation took {time.time() - start_time:.2f}s")
print("Loss:", loss)

# Create plots for the first 5 systems
plt.figure(figsize=(10, 8))
for dof in range(n_dof):
    plt.subplot(2, 2, dof + 1)
    for system in range(5):
        plt.plot(np.arange(steps) * dt, q_history[:, system, dof], 
                label=f'System {system}', marker='.')
    plt.title(f'DOF {dof} Trajectory')
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.legend()
plt.tight_layout()
plt.show()
