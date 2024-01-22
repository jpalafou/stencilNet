from jax import jit
import jax.numpy as jnp
import numpy as onp
from functools import partial
from typing import Tuple
from stencilnet.kernel import reshape_kernel_neighbors, get_inner_shape
from stencilnet.stencils import get_conservative_LCR, get_transverse_integral


@partial(jit, static_argnums=(1, 2, 3, 4))
def apply_bc(
    arr: jnp.ndarray,
    size: int,
    method: str = "periodic",
    expand: bool = False,
    inv: bool = False,
) -> jnp.ndarray:
    """
    args:
        arr         2D array
        size        width of boundary region around inner array
        method      "periodic", others not implemented
        expand      if arr doesn't already have boundaries included in its shape
        inv         return inner array
    returns:
        array with same shape as u and boundary conditions applied
    """
    if isinstance(size, int):
        sizes = (size, size)
    elif isinstance(size, tuple):
        sizes = size
    if inv:
        return arr[sizes[0] : -sizes[0], sizes[1] : -sizes[1]]
    if expand:
        bc_shape = onp.array(arr.shape, dtype=int) + 2 * onp.ones(2, dtype=int) * size
        arr_bc = jnp.zeros(bc_shape) + jnp.nan
        arr_bc = arr_bc.at[sizes[0] : -sizes[0], sizes[1] : -sizes[1]].set(arr)
    else:
        arr_bc = arr
    if method == "periodic":
        cut0, cut1 = sizes
        cut0_x2, cut1_x2 = 2 * sizes[0], 2 * sizes[1]
        arr_bc = arr_bc.at[:cut0, :].set(arr_bc[-cut0_x2:-cut0, :])
        arr_bc = arr_bc.at[-cut0:, :].set(arr_bc[cut0:cut0_x2, :])
        arr_bc = arr_bc.at[:, :cut1].set(arr_bc[:, -cut1_x2:-cut1])
        arr_bc = arr_bc.at[:, -cut1:].set(arr_bc[:, cut1:cut1_x2])
    return arr_bc


@partial(jit, static_argnums=(2,))
def apply_1d_stencil(arr: jnp.ndarray, stencil: jnp.ndarray, axis: int) -> jnp.ndarray:
    """
    args:
        arr         2D array
        stencil     1D array
        axis        axis the 1D stencil is applied along
    returns:
        out         arr convolved with a (n, 1) or (1, n) stencil
                    removes n // 2 cells along axis
    """
    kernel_shape = [1, 1]
    kernel_shape[axis] = len(stencil)
    kernel_shape = tuple(kernel_shape)
    arrsh = reshape_kernel_neighbors(arr, kernel_shape)
    weighted_sums = jnp.sum(arrsh * stencil, axis=1)
    out = weighted_sums.reshape(get_inner_shape(arr.shape, kernel_shape))
    return out


@partial(jit, static_argnums=(1,))
def transverse_interpolation(
    u: jnp.ndarray, p: int
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    args:
        u       2D array of cell volume averages
        p       degree of interpolating polynomial
    returns:
        ((axis0 left mdpnt, axis0 right mdpnt), (axis1 left mdpnt, axis1 right mdpnt))
        2 * ((p + 1) // 2) removed from both axes
    """
    stencils = get_conservative_LCR(p)
    interp_line_axis0 = apply_1d_stencil(u, jnp.array(stencils["center"]), axis=0)  # -
    interp_line_axis1 = apply_1d_stencil(u, jnp.array(stencils["center"]), axis=1)  # |
    interpolated_midpoints = (
        (
            apply_1d_stencil(interp_line_axis1, jnp.array(stencils["left"]), axis=0),
            apply_1d_stencil(interp_line_axis1, jnp.array(stencils["right"]), axis=0),
        ),
        (
            apply_1d_stencil(interp_line_axis0, jnp.array(stencils["left"]), axis=1),
            apply_1d_stencil(interp_line_axis0, jnp.array(stencils["right"]), axis=1),
        ),
    )
    #   _._     ___
    #  / | \   /   \
    #  | | |   .___.
    #  | | |   |   |
    #  \_._/ , \___/
    return interpolated_midpoints


@partial(jit, static_argnums=(1, 2))
def transverse_integral(u_midpoints: jnp.ndarray, p: int, axis: int) -> jnp.ndarray:
    """
    args:
        u_midpoints     2D array, interpolated values at cell interface midpoints
        p               degree of interpolating polynomial
        axis
    returns;
        out             estimate of integral along specified cell interface
                        removes 2 * (p // 2) along axis
    """
    stencil = get_transverse_integral(p)
    out = apply_1d_stencil(u_midpoints, jnp.array(stencil), axis)
    return out


@jit
def riemann_solver(
    value_left_of_interface: jnp.ndarray,
    value_right_of_interface: jnp.ndarray,
    v: jnp.ndarray,
) -> jnp.ndarray:
    """
    args:
        value_left_of_interface
        value_right_of_interface
        v                           velocity at interface
    returns:
        upwinded flux
    """
    return jnp.where(
        v > 0,
        v * value_left_of_interface,
        jnp.where(v < 0, v * value_right_of_interface, 0),
    )


@partial(jit, static_argnums=(2, 3))
def compute_fluxes_from_padded_u(
    u: jnp.ndarray, v: Tuple[jnp.ndarray, jnp.ndarray], p: int, excess: int = 0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    args:
        u       2D array, cell volume averages, includes necessary boundaries
        v       v_x, v_y (2D arrays or floats)
        p       degree of interpolating polynomial
        excess  unecessary array width after riemann problem is solved
                (excess in non-riemann direction, excess in riemann direction)
    returns:
        fluxes_x, fluxes_y
    """
    points = transverse_interpolation(u, p)
    # - 2 * ((p + 1) // 2) from both axes
    excess_slices = slice(excess[0] or None, -excess[0] or None), slice(
        excess[1] or None, -excess[1] or None
    )
    x_interface_flux_points = riemann_solver(
        points[1][1][:, :-1], points[1][0][:, 1:], v[0]
    )
    y_interface_flux_points = riemann_solver(
        points[0][1][:-1, :], points[0][0][1:, :], v[1]
    )
    # (_, - 1), (- 1, _)
    # trim excess
    x_interface_flux_points = x_interface_flux_points[
        excess_slices[0], excess_slices[1]
    ]
    y_interface_flux_points = y_interface_flux_points[
        excess_slices[1], excess_slices[0]
    ]
    fluxes_x = transverse_integral(x_interface_flux_points, p, axis=0)
    fluxes_y = transverse_integral(y_interface_flux_points, p, axis=1)
    # (- 2 * (p // 2), _), (_, - 2 * (p // 2))
    return fluxes_x, fluxes_y


@partial(jit, static_argnums=(2,))
def get_fluxes(
    u: jnp.ndarray,
    v: Tuple[jnp.ndarray, jnp.ndarray],
    p: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    args:
        u       cell volume averages, no boundaries
        v       float or array of advection velocities (v_x, v_y)
        p       polynomial degree of interpolation
    """
    if p == 0:
        needs = 1
        excess = (1, 0)
    elif p == 1:
        needs = 2
        excess = (1, 0)
    elif p == 2:
        needs = 2
        excess = (0, 0)
    elif p == 3:
        needs = 3
        excess = (0, 0)
    elif p == 4:
        needs = 4
        excess = (0, 1)
    elif p == 5:
        needs = 5
        excess = (0, 1)
    elif p == 7:
        needs = 7
        excess = (0, 2)
    u_with_boundaries = apply_bc(
        u,
        size=needs,
        method="periodic",
        expand=True,
    )
    fluxes_x, fluxes_y = compute_fluxes_from_padded_u(
        u=u_with_boundaries, v=v, p=p, excess=excess
    )
    return fluxes_x, fluxes_y


@partial(jit, static_argnums=(2,))
def dynamics(
    u: jnp.ndarray,
    v: Tuple[jnp.ndarray, jnp.ndarray],
    p: int,
    h: [float, float],
) -> jnp.ndarray:
    """
    args:
        u                   cell volume averages, no boundaries, (m, n)
        v                   (vx, vy), array or float (((m, n + 1)), (m + 1, n))
        p                   degree of interpolating polynomial
        h                   cell side lengths (x, y)
        TODO
        blending_func(u)    computes theta for blended high order and low order fluxes
                            theta * high_order_fluxes + (1 - theta) * low_order_fluxes
    """
    fluxes_x, fluxes_y = get_fluxes(u=u, v=v, p=p)
    dudx = (1 / h[0]) * (fluxes_x[:, 1:] - fluxes_x[:, :-1])
    dudy = (1 / h[1]) * (fluxes_y[1:, :] - fluxes_y[:-1, :])
    return -v[0] * dudx + -v[1] * dudy