from typing import overload

import jax.numpy as jnp
from jax import grad, hessian, jacobian

from cuthbertlib.linearize.utils import symmetric_inv_sqrt
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
    rtol: float | None = None,
    has_aux: bool = False,
) -> tuple[Array, Array, Array]: ...
@overload
def linearize_log_density(
    log_density: LogConditionalDensityAux,
    x: ArrayLike,
    y: ArrayLike,
    rtol: float | None = None,
    has_aux: bool = True,
) -> tuple[Array, Array, Array, ArrayTree]: ...


def linearize_log_density(
    log_density: LogConditionalDensity | LogConditionalDensityAux,
    x: ArrayLike,
    y: ArrayLike,
    rtol: float | None = None,
    has_aux: bool = False,
) -> tuple[Array, Array, Array] | tuple[Array, Array, Array, ArrayTree]:
    """Linearize a conditional log density around given points.

    Is exact in the case of a linear Gaussian log_density that is returns
    :math:`H, d, L` in the case that the log_density is of the form

    .. math::
        \\log p(y \\mid x) = -\\frac{1}{2}(y - H x - d)^T (LL^T)^{-1} (y - H x - d) + const

    The cholesky factor of the covariance is calculated using the negative hessian
    of the log_density with respect to y as the precision matrix.
    `symmetric_inv_sqrt` is used to calculate the inverse square root by
    ignoring any singular values that are sufficiently close to zero
    (this is an projection in the case the hessian is not positive definite).

    Alternatively, the cholesky factor can be provided directly
    in `linearize_log_density_given_chol_cov`.

    Args:
        log_density: A conditional log density of y given x. Returns a scalar.
        x: The input points.
        y: The output points.
        rtol: The relative tolerance for the singular values of the precision matrix.
            Passed to `linearize.utils.inv_sqrt` with default calculated based on
            singular values of the precision matrix.
        has_aux: Whether the log_density function returns an auxiliary value.

    Returns:
        Linearized matrix, shift, and cholesky factor of the covariance matrix.
            As well as the auxiliary value if `has_aux` is True.
    """
    prec_and_maybe_aux = hessian(log_density, 1, has_aux=has_aux)(x, y)
    prec = -prec_and_maybe_aux[0] if has_aux else -prec_and_maybe_aux
    chol_cov = symmetric_inv_sqrt(prec, rtol=rtol)
    mat, shift, *extra = linearize_log_density_given_chol_cov(
        log_density, x, y, chol_cov, has_aux=has_aux
    )
    return mat, shift, chol_cov, *extra


@overload
def linearize_log_density_given_chol_cov(
    log_density: LogConditionalDensity,
    x: ArrayLike,
    y: ArrayLike,
    chol_cov: ArrayLike,
    has_aux: bool = False,
) -> tuple[Array, Array]: ...
@overload
def linearize_log_density_given_chol_cov(
    log_density: LogConditionalDensityAux,
    x: ArrayLike,
    y: ArrayLike,
    chol_cov: ArrayLike,
    has_aux: bool = True,
) -> tuple[Array, Array, ArrayTree]: ...


def linearize_log_density_given_chol_cov(
    log_density: LogConditionalDensity | LogConditionalDensityAux,
    x: ArrayLike,
    y: ArrayLike,
    chol_cov: ArrayLike,
    has_aux: bool = False,
) -> tuple[Array, Array] | tuple[Array, Array, ArrayTree]:
    """Linearize a conditional log density around given points.

    Is exact in the case of a linear Gaussian log_density that is returns
    :math:`H, d, L` in the case that the log_density is of the form

    .. math::
        \\log p(y \\mid x) = -\\frac{1}{2}(y - H x - d)^T (LL^T)^{-1} (y - H x - d) + const

    Args:
        log_density: A conditional log density of y given x. Returns a scalar.
        x: The input points.
        y: The output points.
        chol_cov: The cholesky factor of the covariance matrix of the Gaussian.

    Returns:
        Linearized matrix, shift, and cholesky factor of the covariance matrix.
            As well as the auxiliary value if `has_aux` is True.
    """
    chol_cov = jnp.asarray(chol_cov)
    cov = chol_cov @ chol_cov.T

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
