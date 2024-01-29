import jax
import numpy as onp
from stencilnet.finite_volume import generate_rectilinear_mesh, u0, advection_solver

# domain
n = 32
x, y = generate_rectilinear_mesh((0, 1), (0, 1), n)
u_init = u0(x, y, type="square")

advection_solver_config = dict(
    u_init=u_init,
    h=(1 / n, 1 / n),
    v=(2, 1),
    T=1,
    cfl=0.8,
    p=3,
    forward="rk4",
    limit_slopes=False,
)

with jax.profiler.trace("profiling/"):
    # execute solver
    U = advection_solver(
        **advection_solver_config,
    )
    U.block_until_ready()

print(f"devices: {jax.devices()}")
print(f"l2 err: {onp.mean(onp.square(U[-1] - U[0])):.5f}")
