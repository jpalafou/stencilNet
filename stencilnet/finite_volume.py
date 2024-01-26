from jax import jit
import jax.numpy as jnp
import numpy as onp
from functools import partial
from typing import Tuple
from stencilnet.kernel import reshape_kernel_neighbors, get_inner_shape
from stencilnet.stencils import get_conservative_LCR, get_transverse_integral
import stencilnet.ode as ode


@partial(jit, static_argnums=(0, 1, 2))
def generate_rectilinear_mesh(
    x_lims: Tuple[float, float],
    y_lims: Tuple[float, float],
    n_cells: Tuple[int, int],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    args:
        x_lims      (lower, upper)
        y_lims      (lower, upper)
        n_cells     (n_x, n_y) or n_both
    returns:
        x, y        2D mesh
    """
    if isinstance(n_cells, tuple):
        n_cells_tuple = n_cells
    elif isinstance(n_cells, int):
        n_cells_tuple = (n_cells, n_cells)
    x_interfaces = jnp.linspace(x_lims[0], x_lims[1], n_cells_tuple[0] + 1)
    y_interfaces = jnp.linspace(y_lims[0], y_lims[1], n_cells_tuple[1] + 1)
    x_cell_centers = 0.5 * (x_interfaces[1:] + x_interfaces[:-1])
    y_cell_centers = 0.5 * (y_interfaces[1:] + y_interfaces[:-1])
    x, y = jnp.meshgrid(x_cell_centers, y_cell_centers)
    return x, y


@partial(jit, static_argnums=(2,))
def u0(x: jnp.ndarray, y: jnp.ndarray, type: str) -> jnp.ndarray:
    """
    args:
        x, y    2D mesh
        type    "sinus", "square"
    returns:
        array with same shape as x and y
    """
    if type == "sinus":
        return jnp.sin(2 * jnp.pi * (x + y))
    elif type == "square":
        inside_square = jnp.logical_and(x > 0.25, x < 0.75)
        inside_square = jnp.logical_and(inside_square, y > 0.25)
        inside_square = jnp.logical_and(inside_square, y < 0.75)
        return jnp.where(inside_square, 1.0, 0.0)


@partial(jit, static_argnums=(1, 2, 3, 4))
def apply_bc(
    arr: jnp.ndarray,
    size: Tuple[int, int],
    method: str = "periodic",
    expand: bool = False,
    inv: bool = False,
) -> jnp.ndarray:
    """
    args:
        arr         2D array
        size        width of boundary region (axis 0, axis 1)
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
                    removes 2 * n // 2 cells along axis
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
    lower_upper_midpoints = (
        apply_1d_stencil(interp_line_axis1, jnp.array(stencils["left"]), axis=0),
        apply_1d_stencil(interp_line_axis1, jnp.array(stencils["right"]), axis=0),
    )
    left_right_midpoints = (
        apply_1d_stencil(interp_line_axis0, jnp.array(stencils["left"]), axis=1),
        apply_1d_stencil(interp_line_axis0, jnp.array(stencils["right"]), axis=1),
    )
    interpolated_midpoints = lower_upper_midpoints, left_right_midpoints
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
    lower_upper, left_right = transverse_interpolation(u, p)
    # - 2 * ((p + 1) // 2) from both axes
    excess_slices = slice(excess[0] or None, -excess[0] or None), slice(
        excess[1] or None, -excess[1] or None
    )
    x_interface_flux_points = riemann_solver(
        left_right[1][:, :-1], left_right[0][:, 1:], v[0]
    )
    y_interface_flux_points = riemann_solver(
        lower_upper[1][:-1, :], lower_upper[0][1:, :], v[1]
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


@jit
def compute_theta(u: jnp.ndarray, params) -> jnp.ndarray:
    """
    args:
        u       2D array of cell volume averages
        params
    returns:
        theta_x, theta_y    flux limiting values at interfaces
    """
    theta = (
        jnp.zeros(onp.array(u.shape, dtype=int) + onp.array([2, 2], dtype=int)) + params
    )
    theta_x = jnp.maximum(theta[1:-1, 1:], theta[1:-1, :-1])
    theta_y = jnp.maximum(theta[1:, 1:-1], theta[:-1, 1:-1])
    return theta_x, theta_y


@partial(jit, static_argnums=(2, 4))
def dynamics(
    u: jnp.ndarray,
    v: Tuple[jnp.ndarray, jnp.ndarray],
    p: int,
    h: [float, float],
    theta_limiting_params=None,
) -> jnp.ndarray:
    """
    args:
        u                       cell volume averages, no boundaries
        v                       (vx, vy) defined at their normal cell interfaces
        p                       degree of interpolating polynomial
        h                       cell side lengths (x, y)
        theta_limiting_params   no slope limiting if None
    """
    fluxes_x, fluxes_y = get_fluxes(u=u, v=v, p=p)
    if theta_limiting_params is not None:
        low_order_fluxes_x, low_order_fluxes_y = get_fluxes(u=u, v=v, p=0)
        theta_x, theta_y = compute_theta(u=u, params=theta_limiting_params)
        fluxes_x = (1 - theta_x) * fluxes_x + theta_x * low_order_fluxes_x
        fluxes_y = (1 - theta_y) * fluxes_y + theta_y * low_order_fluxes_y
    dudt = -(1 / h[0]) * (fluxes_x[:, 1:] - fluxes_x[:, :-1]) + -(1 / h[1]) * (
        fluxes_y[1:, :] - fluxes_y[:-1, :]
    )
    return dudt


def get_dt_from_cfl(cfl: float, h: Tuple[float, float], v: Tuple[float, float]):
    """
    args:
        cfl     Courant-Friedrichs-Lewy condition
        h       cell side lengths (x, y)
        v       (vx, vy) 2D array or float
    returns:
        dt      time step size
    """
    dt = cfl / (onp.abs(v[0]) / h[0] + onp.abs(v[1] / h[1]))
    return dt


@partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7))
def advection_solver(
    u_init: jnp.ndarray,
    h: Tuple[float, float],
    v: Tuple[float, float],
    T: float,
    cfl: float,
    p: int,
    forward: str,
    theta_limiting_params=None,
):
    """
    solve the IVP: du/dt = d(v_x * u)/dx + d(v_y * u)/dy
    args:
        step(f, u, dt)  explicit integration method
        u_init          state at time 0
        h               cell side lengths (x, y)
        v               (vx, vy) defined at their normal cell interfaces
        T               solving time
        cfl             maximum allowable Courant-Friedrichs-Lewy condition
        p               degree of interpolating polynomial
        forward         "euler", "ssprk2", "ssprk3", "rk4"
        theta_limiting_params   no slope limiting if None
    returns:
        history of states at each time step (n_steps, u.shape)
    """
    step_options = {
        "euler": ode.euler_step,
        "ssprk2": ode.ssprk2_step,
        "ssprk3": ode.ssprk3_step,
        "rk4": ode.rk4_step,
    }
    # compute number of steps
    dt = get_dt_from_cfl(cfl, h, v)
    step_count = int(onp.ceil(T / dt))
    # integrate
    U = ode.integrator(
        f=lambda u: dynamics(
            u=u, v=v, p=p, h=h, theta_limiting_params=theta_limiting_params
        ),
        step=step_options[forward],
        u_init=u_init,
        T=T,
        step_count=step_count,
    )
    return U
