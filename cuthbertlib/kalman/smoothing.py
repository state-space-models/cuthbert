from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from cuthbertlib.linalg import tria
from cuthbertlib.types import Array, ArrayLike


class SmootherScanElement(NamedTuple):
    g: Array
    E: Array
    D: Array


# Note that `update` is aliased as `kalman.smoother_update` in `kalman.__init__.py`
def update(
    filter_m: ArrayLike,
    filter_chol_P: ArrayLike,
    smoother_m: ArrayLike,
    smoother_chol_P: ArrayLike,
    F: ArrayLike,
    c: ArrayLike,
    chol_Q: ArrayLike,
) -> tuple[tuple[Array, Array], Array]:
    """Single step of the square root Rauch–Tung–Striebel (RTS) smoother.

    Args:
        filter_m: Mean of the filtered state.
        filter_chol_P: Generalized Cholesky factor of the filtering covariance.
        smoother_m: Mean of the smoother state.
        smoother_chol_P: Generalized Cholesky factor of the smoothing covariance.
        F: State transition matrix.
        c: State transition shift vector.
        chol_Q: Generalized Cholesky factor of the state transition noise covariance.

    Returns:
        A tuple `(smooth_state, info)`.
        `smooth_state` contains the smoothed mean and square root covariance.
        `info` contains the smoothing gain matrix.

    References:
        Paper: Park and Kailath (1994) - Square-root RTS smoothing algorithms
        Code: https://github.com/EEA-sensors/sqrt-parallel-smoothers/tree/main/parsmooth/sequential
    """
    filter_m, filter_chol_P = jnp.asarray(filter_m), jnp.asarray(filter_chol_P)
    smoother_m, smoother_chol_P = jnp.asarray(smoother_m), jnp.asarray(smoother_chol_P)
    F, c, chol_Q = jnp.asarray(F), jnp.asarray(c), jnp.asarray(chol_Q)

    nx = F.shape[0]
    Phi = jnp.block([[F @ filter_chol_P, chol_Q], [filter_chol_P, jnp.zeros_like(F)]])
    tria_Phi = tria(Phi)
    Phi11 = tria_Phi[:nx, :nx]
    Phi21 = tria_Phi[nx:, :nx]
    Phi22 = tria_Phi[nx:, nx:]
    gain = solve_triangular(Phi11, Phi21.T, trans=True, lower=True).T

    mean_diff = smoother_m - (c + F @ filter_m)
    mean = filter_m + gain @ mean_diff
    chol = tria(jnp.concatenate([Phi22, gain @ smoother_chol_P], axis=1))
    return (mean, chol), gain


def associative_params_single(
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


def smoothing_operator(
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
