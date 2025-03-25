from typing import Callable
from jax.typing import ArrayLike
from jax import Array, hessian, jacobian, grad
import jax.numpy as jnp
from linearize.utils import symmetric_inv_sqrt


def linearize_density(
    log_density: Callable[[ArrayLike, ArrayLike], Array],
    x: ArrayLike,
    y: ArrayLike,
    rtol: float | None = None,
) -> tuple[Array, Array, Array]:
    """Linearize a conditional log density around given points.

    Is exact in the case of a linear Gaussian log_density that is returns
    :math:`H, d, L` in the case that the log_density is of the form

    .. math::
        \\log p(y \\mid x) = -\\frac{1}{2}(y - H x - d)^T (LL^T)^{-1} (y - H x - d) + const

    Args:
        log_density: A conditional log density of y given x. Returns a scalar.
        x: The input points.
        y: The output points.
        rtol: The relative tolerance for the singular values of the precision matrix.
            Passed to `linearize.utils.inv_sqrt` with default calculated based on
            singular values of the precision matrix.

    Returns:
        Linearized matrix, shift, and cholesky factor of the covariance matrix.
    """
    prec = -hessian(log_density, 1)(x, y)
    chol_cov = symmetric_inv_sqrt(prec, rtol=rtol)
    mat, shift = linearize_density_given_chol_cov(log_density, x, y, chol_cov)
    return mat, shift, chol_cov


def linearize_density_given_chol_cov(
    log_density: Callable[[ArrayLike, ArrayLike], Array],
    x: ArrayLike,
    y: ArrayLike,
    chol_cov: ArrayLike,
) -> tuple[Array, Array]:
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
            If not provided, it will use symmetric_inv_sqrt(-hessian(log_density)).

    Returns:
        Linearized matrix, shift, and cholesky factor of the covariance matrix.
    """
    chol_cov = jnp.asarray(chol_cov)

    cov = chol_cov @ chol_cov.T
    jac = jacobian(grad(log_density, 1), 0)(x, y)
    mat = cov @ jac
    g = grad(log_density, 1)(x, y)
    shift = y - mat @ x + cov @ g
    return mat, shift
