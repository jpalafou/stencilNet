import pytest
import jax.numpy as jnp
import jax.random as random
from typing import Tuple
from stencilnet.model import init_mlp_params
from stencilnet.kernel import apply_mlp_to_kernels

KEY = random.PRNGKey(1)

trivial_params = init_mlp_params(KEY, (1, 1))


@pytest.mark.parametrize("shape", [(5, 5), (7, 9), (1000, 1001)])
def test_trivial_operator(shape: Tuple[int, int]):
    """
    The trivial operator with a 1x1 kernel should return an unmodified array
    """
    u_initial = jnp.arange(shape[0] * shape[1]).reshape(shape)
    u_prime = apply_mlp_to_kernels(trivial_params, u_initial, (1, 1))
    assert jnp.all(u_initial == u_prime)
