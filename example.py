import jax.numpy as jnp
import jax.random as random
from stencilnet.model import init_mlp_params
from stencilnet.kernel import apply_mlp_to_kernels, reshape_kernel_neighbors

KEY = random.PRNGKey(1)

kernel_shape = (5, 5)

n_in = kernel_shape[0] * kernel_shape[1]
operator_params = init_mlp_params(KEY, (n_in, 1))
print(operator_params)


shape = (64,) * 2
u = jnp.arange(shape[0] * shape[1]).reshape(shape)

ush = reshape_kernel_neighbors(u, (1, 1))

print(ush)

for _ in range(10):
    uprime = apply_mlp_to_kernels(operator_params, u, kernel_shape)
    print("ran once")
print(u)
print(uprime)
