import pytest
import jax.numpy as jnp
from stencilnet.finite_volume import generate_rectilinear_mesh, u0, dynamics


@pytest.mark.parametrize("p", [0, 1, 2, 3, 4, 5, 7])
def test_zero_velocity_dynamics(p: int):
    """
    args:
        p   degree of interpolating polynomial
    """
    n = 64
    x, y = generate_rectilinear_mesh((0, 1), (0, 1), n)
    u = u0(x, y, "square")
    udot = dynamics(u=u, v=(0, 0), p=p, h=(1 / n, 1 / n))
    u_next = u + 0.1 * udot
    assert jnp.all(u == u_next)
