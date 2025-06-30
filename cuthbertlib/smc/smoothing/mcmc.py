from jax import numpy as jnp, random
import jax
from cuthbertlib.types import (
    ArrayLike,
    Array,
    ArrayTreeLike,
    KeyArray,
    LogConditionalDensity,
)
from cuthbertlib.resampling.multinomial import resampling as multinomial


def simulate(
    key: KeyArray,
    x0_all: ArrayTreeLike,
    x1_all: ArrayTreeLike,
    log_weight_x0_all: ArrayLike,
    log_density: LogConditionalDensity,
    x1_ancestors: Array,
    n_steps: int,
    *_args,
    **_kwargs,
) -> tuple[ArrayTreeLike, Array]:
    """
    An implementation of the IMH algorithm for smoothing smoothing in SMC.

    Args:
        key: A JAX random key.
        x0_all: Collection of previous states.
        x1_all: Collection of current states.
        log_weight_x0_all: Collection of log weights of the previous state.
        log_density: The log density of x1 given x0.
        x1_ancestors: The ancestors of x1 in the genealogy tracking
        n_steps: number of MCMC steps

    Returns:
        A collection of x0 and their sampled indices.

    References:
        https://arxiv.org/abs/2207.00976
    """
    init_key, mcmc_key = jax.random.split(key, 2)
    n_samples = x1_ancestors.shape[0]

    keys = random.split(mcmc_key, n_steps)

    def body(carry, key_t):
        # IMH proposal
        idx, x0_res, idx_log_p = carry
        key_prop, key_acc = jax.random.split(key_t, 2)

        prop_idx = multinomial(key_prop, log_weight_x0_all, n_samples)
        x0_prop = x0_all[prop_idx]
        prop_log_p = jax.vmap(log_density)(x0_prop, x1_all)

        log_alpha = prop_log_p - idx_log_p

        lu = jnp.log(jax.random.uniform(key_prop, (n_samples,)))
        acc = lu < log_alpha

        idx = jnp.where(acc, prop_idx, idx)
        x0_res = jax.tree_map(lambda z: z[idx], x0_all)
        idx_log_p = jnp.where(acc, prop_log_p, idx_log_p)
        return (idx, x0_res, idx_log_p), None

    x0_init = jax.tree_map(lambda z: z[x1_ancestors], x0_all)
    init_log_p = jax.vmap(log_density)(x1_all, x0_init)
    init = (x1_ancestors, x0_init, init_log_p)
    (out_index, out_samples, _), _ = jax.lax.scan(body, init, keys)
    return out_samples, out_index
