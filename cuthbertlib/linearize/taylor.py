from typing import Callable
import jax
from jax import Array
from jax.typing import ArrayLike
from cuthbertlib.linearize.utils import symmetric_inv_sqrt


def linearize_taylor(
    log_potential: Callable[[ArrayLike], Array],
    x: ArrayLike,
) -> tuple[Array, Array]:
    """Linearize a log potential function around a given point using Taylor expansion.

    Unlike the other linearisation methods, this applies to a potential function
    with no required notion of observation $y$ or conditional dependence.

    Instead we have the linearisation

    log G(x) = -0.5 (x - m)^T (L L^T)^{-1} (x - m)

    Args:
        log_likelihood: A callable that returns a non-negative scalar. Does not need
            to be a normalized probability density in its input.
        x: The point to linearize around.

    Returns:
        Linearized mean and cholesky factor of the covariance matrix.
    """
    g = jax.grad(log_potential)(x)
    prec = -jax.hessian(log_potential)(x)
    L = symmetric_inv_sqrt(prec)
    m = x + L @ L.T @ g
    return m, L
