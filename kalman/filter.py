from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import solve_triangular
from jax.typing import ArrayLike

from kalman.utils import mvn_logpdf, tria


class KalmanState(NamedTuple):
    """Gaussian state with mean and square root of covariance.

    Attributes:
        mean: Mean of the Gaussian state.
        chol_cov: Generalized Cholesky factor of the covariance of the Gaussian state.
    """

    mean: Array
    chol_cov: Array


class FilterScanElement(NamedTuple):
    A: Array
    b: Array
    U: Array
    eta: Array
    Z: Array
    ell: Array


def offline_filter(
    initial_state: KalmanState,
    F: ArrayLike,
    c: ArrayLike,
    chol_Q: ArrayLike,
    H: ArrayLike,
    d: ArrayLike,
    chol_R: ArrayLike,
    y: ArrayLike,
    parallel: bool = True,
) -> tuple[KalmanState, Array]:
    """The square root Kalman filter.

    Matrices and vectors that define the transition and observation models for
    every time step, along with the observations, must be batched along the first axis.

    Args:
        initial_state: Initial Gaussian state.
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
    F, c, chol_Q = jnp.asarray(F), jnp.asarray(c), jnp.asarray(chol_Q)
    H, d, chol_R, y = (
        jnp.asarray(H),
        jnp.asarray(d),
        jnp.asarray(chol_R),
        jnp.asarray(y),
    )
    associative_params = sqrt_associative_params(
        initial_state, F, c, chol_Q, H, d, chol_R, y
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
    filt_means = jnp.vstack([initial_state.mean[None, ...], filt_means])
    filt_chol_covs = jnp.vstack([initial_state.chol_cov[None, ...], filt_chol_covs])
    return KalmanState(filt_means, filt_chol_covs), ells[-1]


def sqrt_associative_params(
    initial_state: KalmanState,
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
    m0, chol_P0 = initial_state
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
    m1 = F @ m0 + c
    N1 = tria(jnp.concatenate([F @ chol_P0, chol_Q], 1))

    Psi_ = jnp.block(
        [[H @ N1, chol_R], [N1, jnp.zeros((N1.shape[0], chol_R.shape[1]))]]
    )
    Tria_Psi_ = tria(Psi_)

    nx = chol_Q.shape[0]
    ny = chol_R.shape[0]
    Psi11 = Tria_Psi_[:ny, :ny]
    Psi21 = Tria_Psi_[ny : ny + nx, :ny]
    U = Tria_Psi_[ny : ny + nx, ny:]

    K = solve_triangular(Psi11, Psi21.T, trans="T", lower=True).T

    A = F - K @ H @ F
    b = m1 + K @ (y - H @ m1 - d)

    Z = solve_triangular(Psi11, H @ F, lower=True).T
    eta = solve_triangular(Psi11, Z.T, trans="T", lower=True).T @ (y - H @ c - d)

    if nx > ny:
        Z = jnp.concatenate([Z, jnp.zeros((nx, nx - ny))], axis=1)
    else:
        Z = tria(Z)

    residual = y - H @ m1 - d
    ell = mvn_logpdf(residual, Psi11)

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

    tmp = solve_triangular(Xi11, U1.T, lower=True).T
    D_inv = jnp.eye(nx) - tmp @ Xi21.T  # D = I + C1 @ J2
    A = A2 @ D_inv @ A1
    b = A2 @ D_inv @ (b1 + U1 @ U1.T @ eta2) + b2
    U = tria(jnp.concatenate([A2 @ tmp, U2], axis=1))
    eta = A1.T @ D_inv.T @ (eta2 - Z2 @ Z2.T @ b1) + eta1
    Z = tria(jnp.concatenate([A1.T @ Xi22, Z1], axis=1))

    # Marginal likelihood computation.
    # Code from dynamax for reference.
    # mu = jnp.linalg.solve(C1, b1)
    # t1 = (b1 @ mu - (eta2 + mu) @ jnp.linalg.solve(I_C1J2, C1 @ eta2 + b1))
    # logZ = (logZ1 + logZ2 + 0.5 * jnp.linalg.slogdet(I_C1J2)[1] + 0.5 * t1)

    mu = solve_triangular(U1.T, solve_triangular(U1, b1, lower=True))
    t1 = b1 @ mu - (eta2 + mu) @ D_inv @ (b1 + U1 @ U1.T @ eta2)
    ell = ell1 + ell2 + 0.5 * t1 + 0.5 * jnp.linalg.slogdet(D_inv)[1]

    return FilterScanElement(A, b, U, eta, Z, ell)
