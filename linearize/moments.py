from typing import Callable
import jax
from jax import Array
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
