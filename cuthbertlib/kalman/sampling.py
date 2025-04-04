from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from cuthbertlib.kalman.smoothing import _sqrt_associative_params_single
from cuthbertlib.kalman.utils import append_tree


class SamplerScanElement(NamedTuple):
    gain: Array
    sample: Array


def sampler(
    key: ArrayLike,
    ms: Array,
    chol_Ps: Array,
    Fs: Array,
    cs: Array,
    chol_Qs: Array,
    shape: Sequence[int] = (),
    parallel: bool = True,
) -> Array:
    """Sample from the smoothing distribution of a linear-Gaussian state-space model (LGSSM).

    Args:
        key: A PRNG key.
        ms: Filtering means.
        chol_Ps: Generalized Cholesky factors of the filtering covariances.
        Fs: State transition matrices.
        cs: State transition shift vectors.
        chol_Qs: Generalized Cholesky factors of the state transition noise covariances.
        shape: The shape of the samples to draw. This represents the prefix of the
            output shape which will have an additional two axes representing the
            number of time steps and the state dimension.
        parallel: Whether to use temporal parallelization.

    Returns:
        An array of shape `shape + (num_time_steps, x_dim)` containing the samples.
    """

    associative_params = sqrt_associative_params(
        key, ms, chol_Ps, Fs, cs, chol_Qs, shape
    )
    if parallel:
        all_prefix_sums = jax.lax.associative_scan(
            jax.vmap(sampling_operator), associative_params, reverse=True
        )
    else:
        final_element = jax.tree.map(lambda x: x[-1], associative_params)
        inputs = jax.tree.map(lambda x: x[:-1], associative_params)

        def body(carry, inp):
            next_elem = sampling_operator(carry, inp)
            return next_elem, next_elem

        _, all_prefix_sums = jax.lax.scan(body, final_element, inputs, reverse=True)
        all_prefix_sums = append_tree(all_prefix_sums, final_element)

    return jnp.moveaxis(all_prefix_sums.sample, 0, -2)


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
    return append_tree(interm_elems, last_elem)


def _sqrt_associative_params_interm(
    m: Array, chol_P: Array, F: Array, c: Array, chol_Q: Array, eps: Array
) -> SamplerScanElement:
    inc_m, gain, L = _sqrt_associative_params_single(m, chol_P, F, c, chol_Q)
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
