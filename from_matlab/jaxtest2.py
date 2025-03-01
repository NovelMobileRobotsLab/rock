import jax
import jax.numpy as jnp

# Test basic operation
n_systems = 1000
n_dof = 4
q = jnp.zeros((n_systems, n_dof))
print("Array created:", q.shape)
print("Devices:", jax.devices())