from typing import Callable
import jax
from jax import Array, numpy as jnp
from jax.typing import ArrayLike


def linearize_moments(
    mean_function: Callable[[ArrayLike], Array],
    chol_cov_function: Callable[[ArrayLike], Array],
    x: ArrayLike,
) -> tuple[Array, Array, Array]:
    """Linearize conditional mean and cholesky factor of the covariance matrix
    functions into a linear Gaussian form.

    Takes functions mean_function(x) and chol_cov_function(x) that return the
    conditional mean and cholesky factor of the covariance matrix of the distribution
    p(y | x) for a given input x.

    Returns (H, d, L) defining a linear Gaussian approximation to the conditional
    distribution p(y | x) ≈ N(y | H x + d, L L^T).

    Args:
        mean_function: A callable that returns the conditional mean of the distribution
            for a given input.
        chol_cov_function: A callable that returns the cholesky factor of the covariance
            matrix of the distribution for a given input.
        x: Point to linearize around.

    Returns:
        Linearized matrix, shift, and cholesky factor of the covariance matrix.

    References:
        Code: https://github.com/EEA-sensors/sqrt-parallel-smoothers/blob/main/parsmooth/linearization/_extended.py
    """
    F = jax.jacfwd(mean_function, 0)(x)
    b = mean_function(x) - F @ x
    Chol = chol_cov_function(x)
    return F, b, Chol


### I think this can be removed, as it's just a special case of linearize_moments
### with mean_function(x) = f(x) + m_q and chol_cov_function = lambda x: chol_q
def linearize_callable(
    f: Callable[[ArrayLike], Array], x: ArrayLike, m_q: ArrayLike, chol_q: ArrayLike
) -> tuple[Array, Array, Array]:
    """Linearize a non-linear conditional Gaussian into a linear Gaussian form.

    Takes a non-linear function f(x) and a conditional mean m_q and cholesky factor
    of the covariance matrix chol_q of the distribution
    p(y | x) = N(y | f(x) + m_q, chol_q @ chol_q.T)
    as well as linearization point x.

    Returns (H, d, L) defining a linear Gaussian approximation to the conditional
    distribution p(y | x) ≈ N(y | H x + d, L L^T).

    Args:
        f: A non-linear function of x.
        x: The point to linearize around.
        m_q: The conditional mean of the additive noise distribution.
        chol_q: The cholesky factor of the covariance matrix of the additive noise
            distribution.

    Returns:
        Linearized matrix, shift, and cholesky factor of the covariance matrix.
    """
    chol_q = jnp.asarray(chol_q)
    res = f(x)
    F_x = jax.jacfwd(f, 0)(x)
    return F_x, res - F_x @ x + m_q, chol_q
