import jax
import jax.numpy as jnp
from jax import vmap, jit
import numpy as np
import time

print(jax.devices())

# Parameters
n_systems = 1
n_dof = 4
dt = 0.01

# Initial states and parameters as JAX arrays
q = jnp.zeros((n_systems, n_dof))  # Positions
q_dot = jnp.ones((n_systems, n_dof)) * 0.1  # Velocities
m = jnp.linspace(1.0, 2.0, n_systems)  # Varying mass
h = jnp.linspace(0.5, 1.0, n_systems)  # Varying height

# Define dynamics function (replace with your actual equations)
def compute_dynamics(q, q_dot, m, h):
    # Example matrices (simplified)
    M = jnp.stack([jnp.eye(n_dof) * m_i for m_i in m])  # Shape: (n_systems, n_dof, n_dof)
    C = jnp.zeros((n_systems, n_dof, n_dof))  # Zero Coriolis
    G = jnp.stack([jnp.array([0.0, 0.0, 0.0, 9.81 * m_i]) for m_i in m])  # Gravity
    tau = jnp.ones((n_systems, n_dof)) * jnp.array([1.0, 0.0, 0.0, 0.0])  # Torque

    # Compute acceleration
    M_inv = jnp.linalg.inv(M)
    q_ddot = jax.vmap(jnp.dot)(M_inv, (tau - jax.vmap(jnp.dot)(C, q_dot) - G))
    return q_ddot

# Vectorized Euler step
@jit
def step(q, q_dot, m, h):
    q_ddot = compute_dynamics(q, q_dot, m, h)
    q_dot_new = q_dot + q_ddot * dt
    q_new = q + q_dot * dt
    return q_new, q_dot_new

# Main simulation loop
start_time = time.time()
steps = 1000
for t in range(steps):
    q, q_dot = step(q, q_dot, m, h)

# Block until computation is done and print results
q = jax.device_get(q)
print("Final positions (first 5 systems):", q[:5])
print(f"Simulation took {time.time() - start_time:.2f}s")


print(jax.devices())