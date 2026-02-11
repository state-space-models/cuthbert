"""Implements linearization of conditional log densities."""

from typing import overload

import jax.numpy as jnp
from jax import grad, hessian, jacobian

from cuthbertlib.linalg import chol_cov_with_nans_to_cov, symmetric_inv_sqrt
from cuthbertlib.types import (
    Array,
    ArrayLike,
    ArrayTree,
    LogConditionalDensity,
    LogConditionalDensityAux,
)


@overload
def linearize_log_density(
    log_density: LogConditionalDensity,
    x: ArrayLike,
    y: ArrayLike,
    has_aux: bool = False,
    rtol: float | None = None,
    ignore_nan_dims: bool = False,
) -> tuple[Array, Array, Array]: ...
@overload
def linearize_log_density(
    log_density: LogConditionalDensityAux,
    x: ArrayLike,
    y: ArrayLike,
    has_aux: bool = True,
    rtol: float | None = None,
    ignore_nan_dims: bool = False,
) -> tuple[Array, Array, Array, ArrayTree]: ...


def linearize_log_density(
    log_density: LogConditionalDensity | LogConditionalDensityAux,
    x: ArrayLike,
    y: ArrayLike,
    has_aux: bool = False,
    rtol: float | None = None,
    ignore_nan_dims: bool = False,
) -> tuple[Array, Array, Array] | tuple[Array, Array, Array, ArrayTree]:
    r"""Linearizes a conditional log density around given points.

    The linearization is exact in the case of a linear-Gaussian `log_density`, i.e., it returns
    $(H, d, L)$ if `log_density` is of the form

    $$
    \log p(y \mid x) = -\frac{1}{2}(y - H x - d)^\top (LL^\top)^{-1} (y - H x - d) + \textrm{const}.
    $$

    The Cholesky factor of the covariance is calculated using the negative Hessian
    of `log_density` with respect to `y` as the precision matrix.
    `symmetric_inv_sqrt` is used to calculate the inverse square root by
    ignoring any singular values that are sufficiently close to zero
    (this is a projection in the case the Hessian is not positive definite).

    Alternatively, the Cholesky factor can be provided directly
    in `linearize_log_density_given_chol_cov`.

    Args:
        log_density: A conditional log density of y given x. Returns a scalar.
        x: The input points.
        y: The output points.
        has_aux: Whether `log_density` returns an auxiliary value.
        rtol: The relative tolerance for the singular values of the precision matrix
            when passed to `symmetric_inv_sqrt`.
            Cutoff for small singular values; singular values smaller than
            `rtol * largest_singular_value` are treated as zero.
            The default is determined based on the floating point precision of the dtype.
            See https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.pinv.html.
        ignore_nan_dims: Whether to treat dimensions with NaN on the diagonal of the
            precision matrix as missing and ignore all rows and columns associated with
            them.

    Returns:
        Linearized matrix, shift, and Cholesky factor of the covariance matrix.
            The auxiliary value is also returned if `has_aux` is `True`.
    """
    prec_and_maybe_aux = hessian(log_density, 1, has_aux=has_aux)(x, y)
    prec = -prec_and_maybe_aux[0] if has_aux else -prec_and_maybe_aux
    if ignore_nan_dims:
        prec_diag = jnp.diag(prec)
        nan_mask = jnp.isnan(y) | jnp.isnan(prec_diag)
        prec = prec.at[jnp.diag_indices_from(prec)].set(
            jnp.where(nan_mask, jnp.nan, prec_diag)
        )

    chol_cov = symmetric_inv_sqrt(prec, rtol=rtol, ignore_nan_dims=ignore_nan_dims)
    mat, shift, *extra = linearize_log_density_given_chol_cov(
        log_density, x, y, chol_cov, has_aux=has_aux, ignore_nan_dims=ignore_nan_dims
    )
    return mat, shift, chol_cov, *extra


@overload
def linearize_log_density_given_chol_cov(
    log_density: LogConditionalDensity,
    x: ArrayLike,
    y: ArrayLike,
    chol_cov: ArrayLike,
    has_aux: bool = False,
    ignore_nan_dims: bool = False,
) -> tuple[Array, Array]: ...
@overload
def linearize_log_density_given_chol_cov(
    log_density: LogConditionalDensityAux,
    x: ArrayLike,
    y: ArrayLike,
    chol_cov: ArrayLike,
    has_aux: bool = True,
    ignore_nan_dims: bool = False,
) -> tuple[Array, Array, ArrayTree]: ...


def linearize_log_density_given_chol_cov(
    log_density: LogConditionalDensity | LogConditionalDensityAux,
    x: ArrayLike,
    y: ArrayLike,
    chol_cov: ArrayLike,
    has_aux: bool = False,
    ignore_nan_dims: bool = False,
) -> tuple[Array, Array] | tuple[Array, Array, ArrayTree]:
    r"""Linearizes a conditional log density around given points.

    The linearization is exact in the case of a linear-Gaussian `log_density`, i.e., it returns
    $(H, d)$ if `log_density` is of the form

    $$
    \log p(y \mid x) = -\frac{1}{2}(y - H x - d)^\top (LL^\top)^{-1} (y - H x - d) + \textrm{const},
    $$

    where $L$ is the argument `chol_cov`.

    Args:
        log_density: A conditional log density of y given x. Returns a scalar.
        x: The input points.
        y: The output points.
        chol_cov: The Cholesky factor of the covariance matrix of the Gaussian.
        has_aux: Whether `log_density` returns an auxiliary value.
        ignore_nan_dims: Whether to ignore dimensions with NaN on the diagonal of the
            precision matrix or in y.

    Returns:
        Linearized matrix and shift. The auxiliary value is also returned if `has_aux` is `True`.
    """
    chol_cov = jnp.asarray(chol_cov)

    cov = (
        chol_cov_with_nans_to_cov(chol_cov)
        if ignore_nan_dims
        else chol_cov @ chol_cov.T
    )

    if has_aux:

        def grad_log_density_wrapper_aux(x, y):
            g, aux = grad(log_density, 1, has_aux=True)(x, y)
            return g, (g, aux)

        jac, (g, *extra) = jacobian(grad_log_density_wrapper_aux, 0, has_aux=True)(x, y)
    else:

        def grad_log_density_wrapper(x, y):
            g = grad(log_density, 1)(x, y)
            return g, (g,)

        jac, (g, *extra) = jacobian(grad_log_density_wrapper, 0, has_aux=True)(x, y)

    mat = cov @ jac
    shift = y - mat @ x + cov @ g
    return mat, shift, *extra
