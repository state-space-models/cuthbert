from typing import Callable
from jax.typing import ArrayLike
from jax import Array, numpy as jnp, hessian, jacobian, grad


def linearize(
    log_density: Callable[[ArrayLike, ArrayLike], Array],
    x: ArrayLike,
    y: ArrayLike,
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

    Returns:
        Linearized matrix, shift, and cholesky factor of the covariance matrix.
    """

    mat, shift, cov = _linearize_full_cov(log_density, x, y)
    chol_cov = jnp.linalg.cholesky(cov)
    return mat, shift, chol_cov


def _linearize_full_cov(
    log_density: Callable[[ArrayLike, ArrayLike], Array],
    x: ArrayLike,
    y: ArrayLike,
) -> tuple[Array, Array, Array]:
    prec = -hessian(log_density, 1)(x, y)
    cov = jnp.linalg.inv(prec)
    jac = jacobian(grad(log_density, 1), 0)(x, y)
    mat = cov @ jac
    g = grad(log_density, 1)(x, y)
    dynamics_shift = y - mat @ x + cov @ g
    return mat, dynamics_shift, cov
