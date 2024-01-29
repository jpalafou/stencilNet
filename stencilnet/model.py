from jax import dtypes, jit, vmap
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as random
from functools import partial
from typing import TypeAlias, List, Tuple

Params_List: TypeAlias = List[Tuple[jnp.ndarray, jnp.ndarray]]


def init_layer(key: dtypes.prng_key, n_in: int, n_out: int) -> tuple:
    """
    args:
        key
        n_in    size of layer inputs
        n_out   size of layer ouputs
    returns:
        weights     initialzed with a uniform value summing to 1
        biases      initialzed with zeros
    """
    w_key, b_key = random.split(key)
    w = jnn.initializers.zeros(w_key, (n_out, n_in)) + 1 / n_in
    b = jnn.initializers.zeros(b_key, (n_out,))
    return w, b


def init_mlp_params(key: dtypes.prng_key, sizes: tuple) -> Params_List:
    """
    args:
        key
        sizes   sequence of layer sizes starting with inputs
    returns:
        [(w0, b0), (w1, b1), ...]
    """
    keys = random.split(key, len(sizes))
    params = [init_layer(k, m, n) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]
    return params


@jit
def forward(params: Params_List, input: jnp.ndarray) -> jnp.ndarray:
    """
    args:
        params
        input   single (unbatched) input
    returns:
        out     model with params evaluated at input
    """
    out = input
    for w, b in params[:-1]:
        out = jnp.dot(w, out) + b
        out = jnn.relu(out)
    w, b = params[-1]
    out = jnp.dot(w, out) + b
    return out


@partial(vmap, in_axes=(None, 0))
def batched_forward(params: Params_List, inputs: jnp.ndarray) -> jnp.ndarray:
    """
    args:
        params
        inputs  batch of inputs along first axis
    returns:
        out     batch of outputs with same first axis size of inputs
    """
    return forward(params, inputs)
