import jax.numpy as jnp
import jax.random as random
from stencilnet.model import init_mlp_params
from stencilnet.kernel import apply_mlp_to_kernels, reshape_kernel_neighbors

KEY = random.PRNGKey(1)

kernel_shape = (5, 5)

n_in = kernel_shape[0] * kernel_shape[1]
operator_params = init_mlp_params(KEY, (n_in, 1))


shape = (64,) * 2
u = jnp.arange(shape[0] * shape[1]).reshape(shape)

print("Compiling reshape_kernel_neighbors...")
ush = reshape_kernel_neighbors(u, kernel_shape)
print("Done.")

print(ush)

print("Compiling reshape_kernel_neighbors...")
uprime = apply_mlp_to_kernels(operator_params, u, kernel_shape)
print("Done.")
