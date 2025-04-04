from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import cho_solve
from jax.scipy.linalg import solve_triangular
from jax.typing import ArrayLike

from cuthbertlib.kalman.utils import mvn_logpdf, tria


class KalmanState(NamedTuple):
    """Gaussian state with mean and square root of covariance.

    Attributes:
        mean: Mean of the Gaussian state.
        chol_cov: Generalized Cholesky factor of the covariance of the Gaussian state.
    """

    mean: Array
    chol_cov: Array


class KalmanFilterInfo(NamedTuple):
    """Additional output from the Kalman filter.

    Attributes:
        log_likelihoods: Cumulative log marginal likelihoods.
    """

    log_likelihoods: Array


class FilterScanElement(NamedTuple):
    A: Array
    b: Array
    U: Array
    eta: Array
    Z: Array
    ell: Array


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
    return KalmanState(mean=m1, chol_cov=chol_P1)


# Note that `update` is aliased as `kalman.filter_update` in `kalman.__init__.py`
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
        chol_P: Generalized Cholesky factor of the covariance of the state.
        H: Observation matrix.
        d: Observation shift.
        chol_R: Generalized Cholesky factor of the observation noise covariance.
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


def filter(
    m0: ArrayLike,
    chol_P0: ArrayLike,
    F: ArrayLike,
    c: ArrayLike,
    chol_Q: ArrayLike,
    H: ArrayLike,
    d: ArrayLike,
    chol_R: ArrayLike,
    y: ArrayLike,
    parallel: bool = True,
) -> tuple[KalmanState, KalmanFilterInfo]:
    """The square root Kalman filter.

    The square root Kalman filter is more numerically stable than the standard implementation that
    uses full covariance matrices, especially when using single-precision floating point numbers.
    It also ensures that covariance matrices remain positive semi-definite.

    Matrices and vectors that define the transition and observation models for
    every time step, along with the observations, must be batched along the first axis.

    chol_P0, chol_Q and chol_R must be generalized Cholesky factors. A generalized Cholesky factor
    of a positive semi-definite matrix A is a lower triangular matrix L such that L @ L.T = A.

    Args:
        m0: Mean of the initial state.
        chol_P0: Generalized Cholesky factor of the covariance of the initial state.
        F: State transition matrices.
        c: State transition shift vectors.
        chol_Q: Generalized Cholesky factors of the transition noise covariance.
        H: Observation matrices.
        d: Observation shift vectors.
        chol_R: Generalized Cholesky factors of the observation noise covariance.
        y: Observations.
        parallel: Whether to use temporal parallelization.

    Returns:
        A tuple of the filtered states at every time step and the log marginal likelihood.

    References:
        Paper: Yaghoobi, Corenflos, Hassan and Särkkä (2022) - https://arxiv.org/pdf/2207.00426
        Code: https://github.com/EEA-sensors/sqrt-parallel-smoothers/blob/main/parsmooth/parallel
    """
    m0, chol_P0 = jnp.asarray(m0), jnp.asarray(chol_P0)
    F, c, chol_Q = jnp.asarray(F), jnp.asarray(c), jnp.asarray(chol_Q)
    H, d, chol_R, y = (
        jnp.asarray(H),
        jnp.asarray(d),
        jnp.asarray(chol_R),
        jnp.asarray(y),
    )
    associative_params = sqrt_associative_params(
        m0, chol_P0, F, c, chol_Q, H, d, chol_R, y
    )

    if parallel:
        all_prefix_sums = jax.lax.associative_scan(
            jax.vmap(sqrt_filtering_operator), associative_params
        )
    else:
        init_carry = jax.tree.map(lambda x: x[0], associative_params)
        inputs = jax.tree.map(lambda x: x[1:], associative_params)

        def body(carry, inp):
            next_elem = sqrt_filtering_operator(carry, inp)
            return next_elem, next_elem

        _, all_prefix_sums = jax.lax.scan(body, init_carry, inputs)
        all_prefix_sums = jax.tree.map(
            lambda x, y: jnp.concatenate([x[None, ...], y]), init_carry, all_prefix_sums
        )

    _, filt_means, filt_chol_covs, _, _, ells = all_prefix_sums
    filt_means = jnp.vstack([m0[None, ...], filt_means])
    filt_chol_covs = jnp.vstack([chol_P0[None, ...], filt_chol_covs])
    return KalmanState(filt_means, filt_chol_covs), KalmanFilterInfo(-ells)


def sqrt_associative_params(
    m0: Array,
    chol_P0: Array,
    F: Array,
    c: Array,
    chol_Q: Array,
    H: Array,
    d: Array,
    chol_R: Array,
    y: Array,
) -> FilterScanElement:
    """Compute the filter scan elements for the square root parallel Kalman filter."""
    T = y.shape[0]
    ms = jnp.concatenate([m0[None, ...], jnp.zeros_like(m0, shape=(T - 1,) + m0.shape)])
    chol_Ps = jnp.concatenate(
        [chol_P0[None, ...], jnp.zeros_like(chol_P0, shape=(T - 1,) + chol_P0.shape)]
    )

    return jax.vmap(_sqrt_associative_params_single)(
        ms, chol_Ps, F, c, chol_Q, H, d, chol_R, y
    )


def _sqrt_associative_params_single(
    m0: Array,
    chol_P0: Array,
    F: Array,
    c: Array,
    chol_Q: Array,
    H: Array,
    d: Array,
    chol_R: Array,
    y: Array,
) -> FilterScanElement:
    """Compute the filter scan element for the square root parallel Kalman
    filter for a single time step."""

    ny, nx = H.shape

    # one step prediction
    m1 = F @ m0 + c
    N1 = tria(jnp.concatenate([F @ chol_P0, chol_Q], 1))

    # joint over the predictive and the observation
    Psi_ = jnp.block([[H @ N1, chol_R], [N1, jnp.zeros((nx, ny))]])
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
    b = m1 + K @ (y - H @ m1 - d)  # corrected transition offset

    # information filter
    Z = HF.T @ Psi11_inv.T
    eta = Psi11_inv @ (y - H @ c - d)
    eta = Z @ eta

    if nx > ny:
        Z = jnp.concatenate([Z, jnp.zeros((nx, nx - ny))], axis=1)
    else:
        Z = tria(Z)

    # local log marginal likelihood
    residual = y - H @ m1 - d
    ell = -mvn_logpdf(residual, Psi11)

    return FilterScanElement(A, b, U, eta, Z, ell)


def sqrt_filtering_operator(
    elem_i: FilterScanElement, elem_j: FilterScanElement
) -> FilterScanElement:
    """Binary associative operator for the square root Kalman filter.

    Args:
        elem_i, elem_j: Filter scan elements.

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
    ell = ell1 + ell2 + 0.5 * t1 - 0.5 * jnp.linalg.slogdet(D_inv)[1]

    return FilterScanElement(A, b, U, eta, Z, ell)
