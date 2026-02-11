"""Implements the square root parallel Kalman filter and associative variant."""

from typing import NamedTuple

import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, solve_triangular

from cuthbertlib.linalg import tria
from cuthbertlib.stats import multivariate_normal
from cuthbertlib.stats.multivariate_normal import collect_nans_chol
from cuthbertlib.types import Array, ArrayLike, ScalarArray


class FilterScanElement(NamedTuple):
    """Arrays carried through the Kalman filter scan."""

    A: Array
    b: Array
    U: Array
    eta: Array
    Z: Array
    ell: ScalarArray


def predict(
    m: ArrayLike,
    chol_P: ArrayLike,
    F: ArrayLike,
    c: ArrayLike,
    chol_Q: ArrayLike,
) -> tuple[Array, Array]:
    """Propagate the mean and square root covariance through linear Gaussian dynamics.

    Args:
        m: Mean of the state.
        chol_P: Generalized Cholesky factor of the covariance of the state.
        F: Transition matrix.
        c: Transition shift.
        chol_Q: Generalized Cholesky factor of the transition noise covariance.

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
    return m1, chol_P1


# Note that `update` is aliased as `kalman.filter_update` in `kalman.__init__.py`
def update(
    m: ArrayLike,
    chol_P: ArrayLike,
    H: ArrayLike,
    d: ArrayLike,
    chol_R: ArrayLike,
    y: ArrayLike,
    log_normalizing_constant: ArrayLike = 0.0,
) -> tuple[tuple[Array, Array], Array]:
    """Update the mean and square root covariance with a linear Gaussian observation.

    Args:
        m: Mean of the state.
        chol_P: Generalized Cholesky factor of the covariance of the state.
        H: Observation matrix.
        d: Observation shift.
        chol_R: Generalized Cholesky factor of the observation noise covariance.
        y: Observation.
        log_normalizing_constant: Optional input of log normalizing constant to be added to
            log normalizing constant of the Bayesian update.

    Returns:
        Updated mean and square root covariance as well as the log marginal likelihood.

    References:
        Paper: G. J. Bierman, Factorization Methods for Discrete Sequential Estimation,
        Code: https://github.com/EEA-sensors/sqrt-parallel-smoothers/tree/main/parsmooth/sequential
    """
    # Handle case where there is no observation
    flag = jnp.isnan(y)
    flag, chol_R, H, d, y = collect_nans_chol(flag, chol_R, H, d, y)

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

    ell = multivariate_normal.logpdf(y, y_hat, Imat, nan_support=False)
    return (my, chol_Py), jnp.asarray(ell + log_normalizing_constant)


def associative_params_single(
    F: Array, c: Array, chol_Q: Array, H: Array, d: Array, chol_R: Array, y: Array
) -> FilterScanElement:
    """Single time step for scan element for square root parallel Kalman filter.

    Args:
        F: State transition matrix.
        c: State transition shift vector.
        chol_Q: Generalized Cholesky factor of the state transition noise covariance.
        H: Observation matrix.
        d: Observation shift.
        chol_R: Generalized Cholesky factor of the observation noise covariance.
        y: Observation.

    Returns:
        Prepared scan element for the square root parallel Kalman filter.
    """
    # Handle case where there is no observation
    flag = jnp.isnan(y)
    flag, chol_R, H, d, y = collect_nans_chol(flag, chol_R, H, d, y)

    ny, nx = H.shape

    # joint over the predictive and the observation
    Psi_ = jnp.block([[H @ chol_Q, chol_R], [chol_Q, jnp.zeros((nx, ny))]])

    Tria_Psi_ = tria(Psi_)

    Psi11 = Tria_Psi_[:ny, :ny]
    Psi21 = Tria_Psi_[ny : ny + nx, :ny]
    U = Tria_Psi_[ny : ny + nx, ny:]

    # pre-compute inverse of Psi11: we apply it to matrices and vectors alike.
    Psi11_inv = solve_triangular(Psi11, jnp.eye(ny), lower=True)

    # predictive model given one observation
    K = Psi21 @ Psi11_inv  # local Kalman gain
    HF = H @ F  # temporary variable
    A = F - K @ HF  # corrected transition matrix

    b = c + K @ (y - H @ c - d)  # corrected transition offset

    # information filter
    Z = HF.T @ Psi11_inv.T
    eta = Psi11_inv @ (y - H @ c - d)
    eta = Z @ eta

    if nx > ny:
        Z = jnp.concatenate([Z, jnp.zeros((nx, nx - ny))], axis=1)
    else:
        Z = tria(Z)

    # local log marginal likelihood
    ell = jnp.asarray(
        multivariate_normal.logpdf(y, H @ c + d, Psi11, nan_support=False)
    )

    return FilterScanElement(A, b, U, eta, Z, ell)


def filtering_operator(
    elem_i: FilterScanElement, elem_j: FilterScanElement
) -> FilterScanElement:
    """Binary associative operator for the square root Kalman filter.

    Args:
        elem_i: Filter scan element for the previous time step.
        elem_j: Filter scan element for the current time step.

    Returns:
        FilterScanElement: The output of the associative operator applied to the input elements.
    """
    A1, b1, U1, eta1, Z1, ell1 = elem_i
    A2, b2, U2, eta2, Z2, ell2 = elem_j

    nx = Z2.shape[0]

    Xi = jnp.block([[U1.T @ Z2, jnp.eye(nx)], [Z2, jnp.zeros_like(A1)]])
    tria_xi = tria(Xi)
    Xi11 = tria_xi[:nx, :nx]
    Xi21 = tria_xi[nx : nx + nx, :nx]
    Xi22 = tria_xi[nx : nx + nx, nx:]

    tmp_1 = solve_triangular(Xi11, U1.T, lower=True).T
    D_inv = jnp.eye(nx) - tmp_1 @ Xi21.T
    tmp_2 = D_inv @ (b1 + U1 @ (U1.T @ eta2))

    A = A2 @ D_inv @ A1
    b = A2 @ tmp_2 + b2
    U = tria(jnp.concatenate([A2 @ tmp_1, U2], axis=1))
    eta = A1.T @ (D_inv.T @ (eta2 - Z2 @ (Z2.T @ b1))) + eta1
    Z = tria(jnp.concatenate([A1.T @ Xi22, Z1], axis=1))

    mu = cho_solve((U1, True), b1)
    t1 = b1 @ mu - (eta2 + mu) @ tmp_2
    ell = ell1 + ell2 - 0.5 * t1 + 0.5 * jnp.linalg.slogdet(D_inv)[1]

    return FilterScanElement(A, b, U, eta, Z, ell)
