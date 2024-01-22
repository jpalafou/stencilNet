import pytest
import jax.numpy as jnp
import matplotlib.pyplot as plt
from stencilnet.finite_volume import dynamics


def u0(x: jnp.ndarray, y: jnp.ndarray, type: str) -> jnp.ndarray:
    if type == "square":
        inside_square = jnp.logical_and(x > 0.25, x < 0.75)
        inside_square = jnp.logical_and(inside_square, y > 0.25)
        inside_square = jnp.logical_and(inside_square, y < 0.75)
        return jnp.where(inside_square, 1.0, 0.0)


@pytest.mark.parametrize("p", [0, 1, 2, 3, 4, 5, 7])
def test_zero_velocity_dynamics(p: int):
    """
    args:
        p   degree of interpolating polynomial
    """
    n = 64
    x_interface = jnp.linspace(0, 1, n + 1)
    x_cell_center = 0.5 * (x_interface[1:] + x_interface[:-1])
    x, y = jnp.meshgrid(x_cell_center, x_cell_center)
    u = u0(x, y, "square")
    udot = dynamics(u=u, v=(0, 0), p=p, h=(1 / n, 1 / n))
    u_next = u + 0.1 * udot
    assert jnp.all(u == u_next)
