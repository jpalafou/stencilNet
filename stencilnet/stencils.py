from jax import jit
from functools import partial
from typing import Dict


@partial(jit, static_argnums=(0,))
def get_conservative_LCR(p: int) -> Dict[str, list]:
    """
    stencil weights for conservative lagrange polynomial interpolation
    args:
        p       degree of interpolating polynomial
    returns:
        {"left": [w_l0, ...], "center": [w_c0, ...], "right": [w_r0, ...]}
    """
    stencils = {}
    if p == 0:
        stencils["left"] = [1]
        stencils["center"] = [1]
        stencils["right"] = [1]
    elif p == 1:
        stencils["left"] = [1 / 4, 1, -1 / 4]
        stencils["center"] = [0, 1, 0]
        stencils["right"] = [-1 / 4, 1, 1 / 4]
    elif p == 2:
        stencils["left"] = [1 / 3, 5 / 6, -1 / 6]
        stencils["center"] = [-1 / 24, 13 / 12, -1 / 24]
        stencils["right"] = [-1 / 6, 5 / 6, 1 / 3]
    elif p == 3:
        stencils["left"] = [-1 / 24, 5 / 12, 5 / 6, -1 / 4, 1 / 24]
        stencils["center"] = [0, -1 / 24, 13 / 12, -1 / 24, 0]
        stencils["right"] = [1 / 24, -1 / 4, 5 / 6, 5 / 12, -1 / 24]
    elif p == 4:
        stencils["left"] = [-1 / 20, 9 / 20, 47 / 60, -13 / 60, 1 / 30]
        stencils["center"] = [3 / 640, -29 / 480, 1067 / 960, -29 / 480, 3 / 640]
        stencils["right"] = [1 / 30, -13 / 60, 47 / 60, 9 / 20, -1 / 20]
    elif p == 5:
        stencils["left"] = [
            1 / 120,
            -1 / 12,
            59 / 120,
            47 / 60,
            -31 / 120,
            1 / 15,
            -1 / 120,
        ]
        stencils["center"] = [0, 3 / 640, -29 / 480, 1067 / 960, -29 / 480, 3 / 640, 0]
        stencils["right"] = [
            -1 / 120,
            1 / 15,
            -31 / 120,
            47 / 60,
            59 / 120,
            -1 / 12,
            1 / 120,
        ]
    elif p == 7:
        stencils["left"] = [
            -1 / 560,
            17 / 840,
            -97 / 840,
            449 / 840,
            319 / 420,
            -223 / 840,
            71 / 840,
            -1 / 56,
            1 / 560,
        ]
        stencils["center"] = [
            0,
            -5 / 7168,
            159 / 17920,
            -7621 / 107520,
            30251 / 26880,
            -7621 / 107520,
            159 / 17920,
            -5 / 7168,
            0,
        ]
        stencils["right"] = [
            1 / 560,
            -1 / 56,
            71 / 840,
            -223 / 840,
            319 / 420,
            449 / 840,
            -97 / 840,
            17 / 840,
            -1 / 560,
        ]
    else:
        raise NotImplementedError(f"{p=}")
    return stencils


@partial(jit, static_argnums=(0,))
def get_transverse_integral(p: int) -> list:
    """
    stencil weights for evaluating integral of cell face from transverse neighbors
    args:
        p       degree of interpolating polynomial
    returns:
        [w0, ...]
    """
    if p == 0 or p == 1:
        stencil = [1]
    elif p == 2 or p == 3:
        stencil = [1 / 24, 11 / 12, 1 / 24]
    elif p == 4 or p == 5:
        stencil = [-17 / 5760, 77 / 1440, 863 / 960, 77 / 1440, -17 / 5760]
    elif p == 6 or p == 7:
        stencil = [
            367 / 967680,
            -281 / 53760,
            6361 / 107520,
            215641 / 241920,
            6361 / 107520,
            -281 / 53760,
            367 / 967680,
        ]
    else:
        raise NotImplementedError(f"{p=}")
    return stencil
