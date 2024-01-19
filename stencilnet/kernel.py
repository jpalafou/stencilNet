from jax import jit
import jax.lax as lax
import jax.numpy as jnp
import numpy as onp
from typing import Tuple
from stencilnet.model import batched_forward, Params_List

JIT = True


def get_inner_shape(
    arr_shape: Tuple[int, int], kernel_shape: Tuple[int, int]
) -> Tuple[int, int]:
    """
    args:
        arr_shape       rows, cols
        kernel_shape    rows, cols
    returns:
        shape of inner array when a kernel is applied to arr
    """
    onp_arr_shape = onp.array(arr_shape, dtype=int)
    onp_kernel_shape = onp.array(kernel_shape, dtype=int)
    one_one = onp.ones(2, dtype=int)
    ni_inner, nj_inner = onp_arr_shape - onp_kernel_shape + one_one
    return ni_inner, nj_inner


jit_dec_third = (lambda f: jit(f, static_argnums=(2))) if JIT else lambda f: f


@jit_dec_third
def update_from_window(
    i: int, j: int, kernel_shape: Tuple[int, int], val: Tuple[jnp.ndarray, jnp.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    args:
        i               row index
        j               column index
        kernel_shape    rows, cols
        val
            arr             2D array to be reshaped into rows of neighbors
            arrsh           array where each row is an inner element and each column is
                            a kernel neighor of that element
    returns:
        arr             unchanged
        arrsh           kernel neighbors at i, j are set to the i + j * ni th row
    """
    arr, arrsh = val
    _, nj = get_inner_shape(arr.shape, kernel_shape)
    window = lax.dynamic_slice(arr, (i, j), kernel_shape)
    arrsh = arrsh.at[i * nj + j, :].set(window.flatten())
    return arr, arrsh


jit_dec_second = (lambda f: jit(f, static_argnums=(1,))) if JIT else lambda f: f


@jit_dec_second
def reshape_kernel_neighbors(
    arr: jnp.ndarray, kernel_shape: Tuple[int, int]
) -> jnp.ndarray:
    """
    args:
        arr             2D array
        kernel_shape    rows, cols
    returns:
        out             array where each row represents an inner element of arr and
                        each column a kernel neighbor
    """
    ni_inner, nj_inner = get_inner_shape(arr.shape, kernel_shape)
    out = jnp.zeros((ni_inner * nj_inner, kernel_shape[0] * kernel_shape[1])) + jnp.nan

    # double for loop
    col_func = lambda j, val: lax.fori_loop(
        0,
        ni_inner,
        lambda i, v: update_from_window(i=i, j=j, kernel_shape=kernel_shape, val=v),
        val,
    )
    # _, out = lax.fori_loop(0, nj_inner, col_func, (arr, out))
    val = arr, out
    for i in range(ni_inner):
        for j in range(nj_inner):
            val = update_from_window(i, j, kernel_shape, val)
    _, out = val
    return out


@jit_dec_third
def apply_mlp_to_kernels(
    params: Params_List, arr: jnp.ndarray, kernel_shape=Tuple[int, int]
) -> jnp.ndarray:
    reshaped_arr = reshape_kernel_neighbors(arr, kernel_shape)
    out = batched_forward(params, reshaped_arr)
    out = out.reshape(get_inner_shape(arr.shape, kernel_shape))
    return out
