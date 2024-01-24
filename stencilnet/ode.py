from jax import jit
import jax.lax as lax
import jax.numpy as jnp
from functools import partial


@partial(jit, static_argnums=(0,))
def euler_step(f: callable, u: jnp.ndarray, dt: float) -> jnp.ndarray:
    """
    args:
        f(u)    system dynamics
        u       state at time t
        dt      time step
    returns:
        euler evolution of system at t + dt
    """
    return u + dt * f(u)


@partial(jit, static_argnums=(0,))
def ssprk2_step(f: callable, u: jnp.ndarray, dt: float) -> jnp.ndarray:
    """
    args:
        f(u)    system dynamics
        u       state at time t
        dt      time step
    returns:
        ssprk2 evolution of system at t + dt
    """
    u1 = u
    u2 = u1 + dt * f(u1)
    return (1 / 2) * u1 + (1 / 2) * (u2 + dt * f(u2))


@partial(jit, static_argnums=(0,))
def ssprk3_step(f: callable, u: jnp.ndarray, dt: float) -> jnp.ndarray:
    """
    args:
        f(u)    system dynamics
        u       state at time t
        dt      time step
    returns:
        ssprk3 evolution of system at t + dt
    """
    u1 = u
    u2 = u1 + dt * f(u1)
    u3 = (3 / 4) * u1 + (1 / 4) * (u2 + dt * f(u2))
    return (1 / 3) * u1 + (2 / 3) * (u2 + dt * f(u2))


@partial(jit, static_argnums=(0,))
def rk4_step(f: callable, u: jnp.ndarray, dt: float) -> jnp.ndarray:
    """
    args:
        f(u)    system dynamics
        u       state at time t
        dt      time step
    returns:
        rk4 evolution of system at t + dt
    """
    k1 = f(u)
    k2 = f(u + 0.5 * dt * k1)
    k3 = f(u + 0.5 * dt * k2)
    k4 = f(u + dt * k3)
    dudt = (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return u + dt * dudt


@partial(jit, static_argnums=(0, 1, 3, 4))
def integrator(
    f: callable, step: callable, u_init: jnp.ndarray, n_steps: int, T: float
) -> jnp.ndarray:
    """
    args:
        f(u)            system dynamics
        step(f, u, dt)  explicit integration method
        u_init          state at time 0
        n_steps         number of time steps
        T               solving time
    returns:
        history of states at each time step (n_steps, u.shape)
    """
    t = jnp.linspace(0, T, n_steps)
    U = jnp.zeros((n_steps,) + u_init.shape) + jnp.nan
    U = U.at[0, ...].set(u_init)

    def fori_loop_helper(i, U):
        u0 = U[i - 1]
        dt = t[i] - t[i - 1]
        u = step(f, u0, dt)
        U = U.at[i, ...].set(u)
        return U

    U = lax.fori_loop(1, n_steps, fori_loop_helper, U)
    return U
