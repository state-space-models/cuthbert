from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from kalman.filter import KalmanState
from kalman.utils import append_tree, tria


class SmootherScanElement(NamedTuple):
    g: Array
    E: Array
    D: Array


def smoother(
    filtered_ms: ArrayLike,
    filtered_chol_Ps: ArrayLike,
    Fs: ArrayLike,
    cs: ArrayLike,
    chol_Qs: ArrayLike,
    parallel: bool = True,
) -> KalmanState:
    """The square root Kalman smoother.

    All ArrayLike inputs must be batched over time along the first axis.

    Args:
        filtered_ms: The means of the filtered states.
        filtered_chol_Ps: The generalized Cholesky factors of the covariances of the filtered states.
        Fs: State transition matrices.
        cs: State transition shift vectors.
        chol_Qs: Generalized Cholesky factors of the state transition noise covariances.
        parallel: Whether to use temporal parallelization.

    Returns:
        The smoothed states at every time step.

    References:
            Paper: Yaghoobi, Corenflos, Hassan and Särkkä (2022) - https://arxiv.org/pdf/2207.00426
            Code: https://github.com/EEA-sensors/sqrt-parallel-smoothers/blob/main/parsmooth/parallel
    """
    ms, Ps = jnp.asarray(filtered_ms), jnp.asarray(filtered_chol_Ps)
    Fs, cs, chol_Qs = jnp.asarray(Fs), jnp.asarray(cs), jnp.asarray(chol_Qs)
    associative_params = sqrt_associative_params(ms, Ps, Fs, cs, chol_Qs)

    if parallel:
        all_prefix_sums = jax.lax.associative_scan(
            jax.vmap(sqrt_smoothing_operator), associative_params, reverse=True
        )
    else:
        init_carry = jax.tree.map(lambda x: x[-1], associative_params)
        inputs = jax.tree.map(lambda x: x[:-1], associative_params)

        def body(carry, inp):
            next_elem = sqrt_smoothing_operator(carry, inp)
            return next_elem, next_elem

        _, all_prefix_sums = jax.lax.scan(body, init_carry, inputs, reverse=True)
        all_prefix_sums = append_tree(all_prefix_sums, init_carry)

    smoothed_means, _, smoothed_chol_covs = all_prefix_sums
    return KalmanState(smoothed_means, smoothed_chol_covs)


def sqrt_associative_params(
    ms: Array, Ps: Array, Fs: Array, cs: Array, chol_Qs: Array
) -> SmootherScanElement:
    """Compute the smoother scan elements for the square root parallel Kalman smoother."""
    scan_elements = jax.vmap(_sqrt_associative_params_single)(
        ms[:-1], Ps[:-1], Fs, cs, chol_Qs
    )
    final_element = SmootherScanElement(ms[-1], jnp.zeros_like(Ps[-1]), Ps[-1])
    return append_tree(scan_elements, final_element)


def _sqrt_associative_params_single(
    m: Array,
    chol_P: Array,
    F: Array,
    c: Array,
    chol_Q: Array,
) -> SmootherScanElement:
    """Compute the smoother scan element for the square root parallel Kalman
    smoother for a single time step."""
    nx = chol_Q.shape[0]

    Phi = jnp.block([[F @ chol_P, chol_Q], [chol_P, jnp.zeros_like(chol_Q)]])
    Tria_Phi = tria(Phi)
    Phi11 = Tria_Phi[:nx, :nx]
    Phi21 = Tria_Phi[nx:, :nx]
    D = Tria_Phi[nx:, nx:]

    E = jax.scipy.linalg.solve_triangular(Phi11.T, Phi21.T).T
    g = m - E @ (F @ m + c)
    return SmootherScanElement(g, E, D)


def sqrt_smoothing_operator(
    elem_i: SmootherScanElement, elem_j: SmootherScanElement
) -> SmootherScanElement:
    """Binary associative operator for the square root Kalman smoother.

    Args:
        elem_i, elem_j: Smoother scan elements.

    Returns:
        SmootherScanElement: The output of the associative operator applied to the input elements.
    """
    g_i, E_i, D_i = elem_i
    g_j, E_j, D_j = elem_j

    g = E_j @ g_i + g_j
    E = E_j @ E_i
    D = tria(jnp.concatenate([E_j @ D_i, D_j], axis=1))

    return SmootherScanElement(g, E, D)
