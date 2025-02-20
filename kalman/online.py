import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from jax.typing import ArrayLike

from kalman.filter import KalmanState, KalmanFilterInfo
from kalman.utils import mvn_logpdf, tria


def predict(
    m: ArrayLike,
    chol_P: ArrayLike,
    F: ArrayLike,
    c: ArrayLike,
    chol_Q: ArrayLike,
) -> KalmanState:
    """
    Propagate the mean and square root covariance through linear Gaussian dynamics.

    Args:
        m: Mean of the state.
        chol_P: Square root of the covariance of the state.
        F: Transition matrix.
        c: Transition shift.
        chol_Q: Square root of the transition noise covariance.

    Returns:
        Propagated mean and square root covariance.

    References:
        Paper: G. J. Bierman, Factorization Methods for Discrete Sequential Estimation,
        Code: https://github.com/EEA-sensors/sqrt-parallel-smoothers/tree/main/parsmooth/sequential
    """
    m, chol_P = jnp.asarray(m), jnp.asarray(chol_P)
    F, c, chol_Q = jnp.asarray(F), jnp.asarray(c), jnp.asarray(chol_Q)
    m1 = F @ m + c
    A = jnp.concatenate([F @ chol_P, chol_Q], axis=1)
    chol_P1 = tria(A)
    return KalmanState(mean=m1, chol_cov=chol_P1)


def update(
    m: ArrayLike,
    chol_P: ArrayLike,
    H: ArrayLike,
    d: ArrayLike,
    chol_R: ArrayLike,
    y: ArrayLike,
) -> tuple[KalmanState, KalmanFilterInfo]:
    """
    Update the mean and square root covariance with a linear Gaussian observation.

    Args:
        m: Mean of the state.
        chol_P: Square root of the covariance of the state.
        H: Observation matrix.
        d: Observation shift.
        chol_R: Square root of the observation noise covariance.
        y: Observation.

    Returns:
        Updated mean and square root covariance as well as the log marginal likelihood.

    References:
        Paper: G. J. Bierman, Factorization Methods for Discrete Sequential Estimation,
        Code: https://github.com/EEA-sensors/sqrt-parallel-smoothers/tree/main/parsmooth/sequential
    """
    m, chol_P = jnp.asarray(m), jnp.asarray(chol_P)
    H, d, chol_R = jnp.asarray(H), jnp.asarray(d), jnp.asarray(chol_R)
    y = jnp.asarray(y)

    n_y, n_x = H.shape

    y_hat = H @ m + d
    y_diff = y - y_hat

    M = jnp.block(
        [
            [H @ chol_P, chol_R],
            [chol_P, jnp.zeros((n_x, n_y), dtype=chol_P.dtype)],
        ]
    )
    chol_S = tria(M)
    chol_Py = chol_S[n_y:, n_y:]

    Gmat = chol_S[n_y:, :n_y]
    Imat = chol_S[:n_y, :n_y]

    my = m + Gmat @ solve_triangular(Imat, y_diff, lower=True)
    ell = mvn_logpdf(y_diff, Imat)
    return KalmanState(mean=my, chol_cov=chol_Py), KalmanFilterInfo(log_likelihoods=ell)
