from functools import partial

from jax import Array, random, numpy as jnp
from jax.typing import ArrayLike
from jax.scipy.special import logsumexp
from jax.lax import cond, select

from cuthbertlib.resampling.protocols import (
    resampling_decorator,
    conditional_resampling_decorator,
)
from cuthbertlib.resampling.utils import inverse_cdf

_DESCRIPTION = """
The Systematic resampling is a variance reduction which places marginally
uniform samples into the [0, 1] interval but only requires one uniform random.
"""


@partial(resampling_decorator, name="Systematic", desc=_DESCRIPTION)
def resampling(key: Array, logits: ArrayLike, n: int) -> Array:
    us = (random.uniform(key, ()) + jnp.arange(n)) / n
    return inverse_cdf(us, logits)


@partial(conditional_resampling_decorator, name="Systematic", desc=_DESCRIPTION)
def conditional_resampling(
    key: Array, logits: ArrayLike, n: int, pivot_in: int, pivot_out: int
) -> Array:
    logits = jnp.asarray(logits)
    # FIXME: no need for normalizing in theory
    N = logits.shape[0]
    logits -= logsumexp(logits)

    # FIXME: this rolling should be done in a single function, but this is killing me.
    arange = jnp.arange(N)
    logits = jnp.roll(logits, -pivot_out)
    arange = jnp.roll(arange, -pivot_out)

    idx = conditional_resampling_0_to_0(key, logits, n)
    idx = arange[idx]
    idx = jnp.roll(idx, pivot_in)
    return idx


def conditional_resampling_0_to_0(
    key: Array,
    logits: ArrayLike,
    n: int,
) -> Array:
    logits = jnp.asarray(logits)

    N = logits.shape[0]
    weights = jnp.exp(logits - logsumexp(logits))
    tmp = n * weights[0]
    tmp_floor = jnp.floor(tmp)

    U, V, W = random.uniform(key, (3,))

    def _otherwise():
        rem = tmp - tmp_floor
        p_cond = rem * (tmp_floor + 1) / tmp
        return select(V < p_cond, rem * U, rem + (1.0 - rem) * U)

    uniform = cond(tmp <= 1, lambda: tmp * U, _otherwise)

    linspace = (jnp.arange(n) + uniform) / n
    idx = inverse_cdf(linspace, logits)

    n_zero = jnp.sum(idx == 0)
    zero_loc = jnp.flatnonzero(idx == 0, size=n, fill_value=-1)
    roll_idx = jnp.floor(n_zero * W).astype(int)

    idx = select(n_zero == 1, idx, jnp.roll(idx, -zero_loc[roll_idx]))
    return jnp.clip(idx, 0, N - 1)
