from functools import partial

from jax import Array, random, numpy as jnp
from jax.typing import ArrayLike
from jax.scipy.special import logsumexp

from resampling.protocols import (
    resampling_decorator,
    conditional_resampling_decorator,
)
from resampling import multinomial

_DESCRIPTION = """
The Killing resampling is a simple resampling mechanism that checks if 
particles should be replaced or not, based on their weights. 
If they should be replaced, they are replaced by another particle using 
multinomial resampling on residual weights. It presents the benefit of not 
"breaking" trajectories as much as multinomial resampling, and therefore is
stable in contexts where the trajectories are important (typically when dealing
with continuous-time models). 

By construction, it requires the same number of sampled indices `n` as the
number of particles `logits.shape[0]`.
"""


@partial(resampling_decorator, name="Killing", desc=_DESCRIPTION)
def resampling(key: Array, logits: ArrayLike, n: int) -> Array:
    logits = jnp.asarray(logits)
    key_1, key_2 = random.split(key)
    N = logits.shape[0]
    if n != N:
        raise AssertionError(
            "The number of sampled indices must be equal to the number of "
            f"particles for `Killing` resampling. Got {n} instead of {N}."
        )

    max_logit = jnp.max(logits)
    log_uniforms = jnp.log(random.uniform(key_1, (N,)))

    survived = log_uniforms <= logits - max_logit
    if_survived = jnp.arange(N)  # If the particle survives, it keeps its index
    otherwise = multinomial.resampling(
        key_2, logits, N
    )  # otherwise, it is replaced by another particle
    idx = jnp.where(survived, if_survived, otherwise)
    return idx


@partial(conditional_resampling_decorator, name="Killing", desc=_DESCRIPTION)
def conditional_resampling(
    key: Array, logits: ArrayLike, n: int, pivot_in: int, pivot_out: int
) -> Array:
    # Unconditional resampling
    key_resample, key_shuffle = random.split(key)
    idx = resampling(key_resample, logits, n)

    # Conditional rolling pivot
    max_logit = jnp.max(logits)

    pivot_logits = _log1mexp(logits - max_logit)
    pivot_logits -= jnp.log(n)
    pivot_logits = pivot_logits.at[pivot_out].set(-jnp.inf)
    pivot_logits_i = _log1mexp(logsumexp(pivot_logits))
    pivot_logits = pivot_logits.at[pivot_out].set(pivot_logits_i)

    pivot_weights = jnp.exp(pivot_logits - logsumexp(pivot_logits))
    pivot = random.choice(key_shuffle, n, p=pivot_weights)
    idx = jnp.roll(idx, pivot_in - pivot)
    idx = idx.at[pivot_in].set(pivot_out)
    return idx


def _log1mexp(x: ArrayLike) -> Array:
    # There is probably a better way to do this
    return jnp.log(1 - jnp.exp(x))
