from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp

from cuthbertlib.kalman.smoothing import associative_params_single
from cuthbertlib.types import Array, ArrayLike


class SamplerScanElement(NamedTuple):
    gain: Array
    sample: Array


def sqrt_associative_params(
    key: ArrayLike,
    ms: Array,
    chol_Ps: Array,
    Fs: Array,
    cs: Array,
    chol_Qs: Array,
    shape: Sequence[int],
) -> SamplerScanElement:
    """Compute the sampler scan elements."""
    shape = tuple(shape)
    eps = jax.random.normal(key, ms.shape[:1] + shape + ms.shape[1:])
    interm_elems = jax.vmap(_sqrt_associative_params_interm)(
        ms[:-1], chol_Ps[:-1], Fs, cs, chol_Qs, eps[:-1]
    )
    last_elem = _sqrt_associative_params_final(ms[-1], chol_Ps[-1], eps[-1])
    return jax.tree.map(
        lambda x, y: jnp.concatenate([x, y[None]]), interm_elems, last_elem
    )


def _sqrt_associative_params_interm(
    m: Array, chol_P: Array, F: Array, c: Array, chol_Q: Array, eps: Array
) -> SamplerScanElement:
    inc_m, gain, L = associative_params_single(m, chol_P, F, c, chol_Q)
    inc = inc_m + eps @ L.T
    return SamplerScanElement(gain, inc)


def _sqrt_associative_params_final(
    m: Array, chol_P: Array, eps: Array
) -> SamplerScanElement:
    gain = jnp.zeros_like(chol_P)
    sample = m + eps @ chol_P.T
    return SamplerScanElement(gain, sample)


def sampling_operator(
    elem_i: SamplerScanElement, elem_j: SamplerScanElement
) -> SamplerScanElement:
    """Binary associative operator for sampling."""
    G_i, e_i = elem_i
    G_j, e_j = elem_j
    G = G_j @ G_i
    e = e_i @ G_j.T + e_j
    return SamplerScanElement(G, e)
