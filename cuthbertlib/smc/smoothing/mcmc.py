import jax
from jax import numpy as jnp
from jax import random

from cuthbertlib.resampling import multinomial
from cuthbertlib.smc.smoothing.tracing import simulate as ancestor_tracing_simulate
from cuthbertlib.types import (
    Array,
    ArrayLike,
    ArrayTree,
    ArrayTreeLike,
    KeyArray,
    LogConditionalDensity,
)


def simulate(
    key: KeyArray,
    x0_all: ArrayTreeLike,
    x1_all: ArrayTreeLike,
    log_weight_x0_all: ArrayLike,
    log_density: LogConditionalDensity,
    x1_ancestor_indices: ArrayLike,
    n_steps: int,
) -> tuple[ArrayTree, Array]:
    """
    An implementation of the IMH algorithm for smoothing in SMC.

    Args:
        key: A JAX random key.
        x0_all: Collection of previous states.
        x1_all: Collection of current states.
        log_weight_x0_all: Collection of log weights of the previous state.
        log_density: The log density function of x1 given x0.
        x1_ancestor_indices: The ancestor indices of x1.
        n_steps: number of MCMC steps

    Returns:
        A collection of x0 and their sampled indices.

    References:
        https://arxiv.org/abs/2207.00976
    """
    key, subkey = random.split(key)
    x0_init, x1_ancestor_indices = ancestor_tracing_simulate(
        subkey, x0_all, x1_all, log_weight_x0_all, log_density, x1_ancestor_indices
    )
    n_samples = x1_ancestor_indices.shape[0]

    keys = random.split(key, (n_steps * 2)).reshape((n_steps, 2))

    def body(carry, keys_t):
        # IMH proposal
        idx, x0_res, idx_log_p = carry
        key_prop, key_acc = keys_t

        prop_idx = multinomial.resampling(key_prop, log_weight_x0_all, n_samples)
        x0_prop = jax.tree.map(lambda z: z[prop_idx], x0_all)
        prop_log_p = jax.vmap(log_density)(x0_prop, x1_all)

        log_alpha = prop_log_p - idx_log_p

        lu = jnp.log(random.uniform(key_acc, (n_samples,)))
        acc = lu < log_alpha

        idx: Array = jnp.where(acc, prop_idx, idx)
        x0_res: ArrayTreeLike = jax.tree.map(lambda z: z[idx], x0_all)
        idx_log_p: Array = jnp.where(acc, prop_log_p, idx_log_p)
        return (idx, x0_res, idx_log_p), None

    x0_init = jax.tree.map(lambda z: z[x1_ancestor_indices], x0_all)
    init_log_p = jax.vmap(log_density)(x1_all, x0_init)
    init = (x1_ancestor_indices, x0_init, init_log_p)
    (out_index, out_samples, _), _ = jax.lax.scan(body, init, keys)
    return out_samples, out_index
