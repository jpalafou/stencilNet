from jax import jit
import jax.numpy as jnp
import numpy as onp
from stencilnet.model import Params_List


@jit
def mse(x1: jnp.ndarray, x2: jnp.ndarray) -> float:
    """
    args:
        x1, x2
    returns:
        mean square error of x1 and x2
    """
    return jnp.mean(jnp.square(x1 - x2))


@jit
def symmetry_regularization(params: Params_List) -> float:
    """
    args:
        params      has square layer sizes
    returns:
        MSE of assymetry of layer weights
    """
    loss = 0
    for w, b in params:
        kernel_side_length = int(onp.sqrt(w.shape[1]))
        reshaped_w = w.reshape((w.shape[0], kernel_side_length, kernel_side_length))
        half_idx = kernel_side_length // 2
        loss += mse(reshaped_w[:, :half_idx, :], reshaped_w[:, -half_idx:, :])
        loss += mse(reshaped_w[:, :, :half_idx], reshaped_w[:, :, -half_idx:])
        loss += mse(b[:half_idx], b[-half_idx:])
    return loss


def update_params(params: Params_List, grads: Params_List, lr: float) -> Params_List:
    """
    args:
        params      [(w0, b0), ...]
        grads       same shape as params
        lr          learning rate
    returns:
        updated params
    """
    return [(w - lr * wg, b - lr * bg) for (w, b), (wg, bg) in zip(params, grads)]
