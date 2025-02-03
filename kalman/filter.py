from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import solve_triangular
from jax.typing import ArrayLike

from kalman.utils import arraylike_to_array, tria


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


def offline_filter(
    x0: KalmanState,
    F: ArrayLike,
    c: ArrayLike,
    chol_Q: ArrayLike,
    H: ArrayLike,
    d: ArrayLike,
    chol_R: ArrayLike,
    y: ArrayLike,
    parallel: bool = True,
):
    """The square root Kalman filter.

    Matrices and vectors that define the transition and observation models for
    every time step, along with the observations, must be batched along the first axis.

    Args:
        x0: Initial Gaussian state.
        F: State transition matrices.
        c: State transition shift vectors.
        chol_Q: Generalized Cholesky factors of the transition noise covariance.
        H: Observation matrices.
        d: Observation shift vectors.
        chol_R: Generalized Cholesky factors of the observation noise covariance.
        y: Observations.
        parallel: Whether to use temporal parallelization.

    Returns:
        KalmanState: Filtered states at every time step.

    References:
        Paper: Yaghoobi, Corenflos, Hassan and Särkkä (2022) - https://arxiv.org/pdf/2207.00426
        Code: https://github.com/EEA-sensors/sqrt-parallel-smoothers/blob/main/parsmooth/parallel
    """
    associative_params = sqrt_associative_params(x0, F, c, chol_Q, H, d, chol_R, y)

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
            lambda x, y: jnp.vstack([x[None, ...], y]), init_carry, all_prefix_sums
        )

    _, filt_means, filt_chol_covs, _, _ = all_prefix_sums
    filt_means = jnp.vstack([x0.mean[None, ...], filt_means])
    filt_chol_covs = jnp.vstack([x0.chol_cov[None, ...], filt_chol_covs])
    return KalmanState(filt_means, filt_chol_covs)


def sqrt_associative_params(
    x0: KalmanState,
    F: ArrayLike,
    c: ArrayLike,
    chol_Q: ArrayLike,
    H: ArrayLike,
    d: ArrayLike,
    chol_R: ArrayLike,
    y: ArrayLike,
) -> FilterScanElement:
    """Compute the filter scan elements for the square root parallel Kalman filter."""
    if not isinstance(y, ArrayLike):
        raise TypeError("y must be an ArrayLike.")
    y = jnp.asarray(y)

    T = y.shape[0]
    m0, chol_P0 = x0
    ms = jnp.concatenate([m0[None, ...], jnp.zeros_like(m0, shape=(T - 1,) + m0.shape)])
    chol_Ps = jnp.concatenate(
        [chol_P0[None, ...], jnp.zeros_like(chol_P0, shape=(T - 1,) + chol_P0.shape)]
    )

    return jax.vmap(_sqrt_associative_params_single)(
        ms, chol_Ps, F, c, chol_Q, H, d, chol_R, y
    )


def _sqrt_associative_params_single(
    m0: ArrayLike,
    chol_P0: ArrayLike,
    F: ArrayLike,
    c: ArrayLike,
    chol_Q: ArrayLike,
    H: ArrayLike,
    d: ArrayLike,
    chol_R: ArrayLike,
    y: ArrayLike,
) -> FilterScanElement:
    """Compute the filter scan element for the square root parallel Kalman
    filter for a single time step."""
    m0, chol_P0, F, c, chol_Q, H, d, chol_R, y = arraylike_to_array(
        "_sqrt_associative_params_single", m0, chol_P0, F, c, chol_Q, H, d, chol_R, y
    )
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

    K = solve_triangular(Psi11, Psi21.T, trans=True, lower=True).T

    A = F - K @ H @ F
    b = m1 + K @ (y - H @ m1 - d)

    Z = solve_triangular(Psi11, H @ F, lower=True).T
    eta = solve_triangular(Psi11, Z.T, trans=True, lower=True).T @ (y - H @ c - d)

    if nx > ny:
        Z = jnp.concatenate([Z, jnp.zeros((nx, nx - ny))], axis=1)
    else:
        Z = tria(Z)

    # TODO: Figure out how to compute the log marginal likelihood. Like
    # https://github.com/probml/dynamax/blob/51b7dc5440ff25df731958a12ffbba75a1380001/dynamax/linear_gaussian_ssm/parallel_inference.py#L248
    # but for the square root version.

    return FilterScanElement(A, b, U, eta, Z)


def sqrt_filtering_operator(
    elem_i: FilterScanElement, elem_j: FilterScanElement
) -> FilterScanElement:
    """Binary associative operator for the square root Kalman filter.

    Args:
        elem_i, elem_j: Filter scan elements.

    Returns:
        FilterScanElement: The output of the associative operator applied to the input elements.
    """
    A1, b1, U1, eta1, Z1 = elem_i
    A2, b2, U2, eta2, Z2 = elem_j

    nx = Z2.shape[0]

    Xi = jnp.block([[U1.T @ Z2, jnp.eye(nx)], [Z2, jnp.zeros_like(A1)]])
    tria_xi = tria(Xi)
    Xi11 = tria_xi[:nx, :nx]
    Xi21 = tria_xi[nx : nx + nx, :nx]
    Xi22 = tria_xi[nx : nx + nx, nx:]

    A = A2 @ A1 - solve_triangular(Xi11, U1.T @ A2.T, lower=True).T @ Xi21.T @ A1
    b = (
        A2
        @ (jnp.eye(nx) - solve_triangular(Xi11, U1.T, lower=True).T @ Xi21.T)
        @ (b1 + U1 @ U1.T @ eta2)
        + b2
    )
    U = tria(
        jnp.concatenate([solve_triangular(Xi11, U1.T @ A2.T, lower=True).T, U2], axis=1)
    )
    eta = (
        A1.T
        @ (
            jnp.eye(nx)
            - solve_triangular(Xi11, Xi21.T, lower=True, trans=True).T @ U1.T
        )
        @ (eta2 - Z2 @ Z2.T @ b1)
        + eta1
    )
    Z = tria(jnp.concatenate([A1.T @ Xi22, Z1], axis=1))

    return FilterScanElement(A, b, U, eta, Z)
