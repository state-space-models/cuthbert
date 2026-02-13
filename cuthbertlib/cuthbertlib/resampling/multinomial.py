"""Implements multinomial resampling."""

from functools import partial

import jax
from jax import numpy as jnp
from jax import random

from cuthbertlib.resampling.protocols import (
    conditional_resampling_decorator,
    resampling_decorator,
)
from cuthbertlib.resampling.utils import inverse_cdf
from cuthbertlib.types import Array, ArrayLike

_DESCRIPTION = """
This has higher variance than other resampling schemes as it samples from
the ancestors independently. It should only be used for illustration purposes,
or if your algorithm *REALLY REALLY* needs independent samples.
As a rule of thumb, you often don't."""


@partial(resampling_decorator, name="Multinomial", desc=_DESCRIPTION)
def resampling(key: Array, logits: ArrayLike, n: int) -> Array:
    # In practice we don't have to sort the generated uniforms, but searchsorted
    # works faster and is more stable if both inputs are sorted, so we use the
    # _sorted_uniforms from N. Chopin, but still use searchsorted instead of his
    # O(N) loop as our code is meant to work on GPU where searchsorted is
    # O(log(N)) anyway.
    # We then permute the indices to enforce exchangeability.

    key_uniforms, key_shuffle = random.split(key)
    sorted_uniforms = _sorted_uniforms(key_uniforms, n)
    idx = inverse_cdf(sorted_uniforms, logits)
    return random.permutation(key_shuffle, idx)


@partial(conditional_resampling_decorator, name="Multinomial", desc=_DESCRIPTION)
def conditional_resampling(
    key: Array, logits: ArrayLike, n: int, pivot_in: int, pivot_out: int
) -> Array:
    idx = resampling(key, logits, n)
    idx = idx.at[pivot_in].set(pivot_out)
    return idx


@partial(jax.jit, static_argnames=("n",))
def _sorted_uniforms(key: Array, n: int) -> Array:
    # This is a small modification of the code from N. Chopin to output sorted
    # log-uniforms *directly*. N. Chopin's code outputs sorted uniforms.
    us = random.uniform(key, (n + 1,))
    z = jnp.cumsum(-jnp.log(us))
    return z[:-1] / z[-1]
