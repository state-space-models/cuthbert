from typing import Callable
import jax
from cuthbertlib.types import Array
from jax.typing import ArrayLike


def linearize_moments(
    mean_and_chol_cov_function: Callable[[ArrayLike], tuple[Array, Array]],
    x: ArrayLike,
) -> tuple[Array, Array, Array]:
    """Linearize conditional mean and cholesky factor of the covariance matrix
    functions into a linear Gaussian form.

    Takes a function mean_and_chol_cov_function(x) that returns the
    conditional mean and cholesky factor of the covariance matrix of the distribution
    p(y | x) for a given input x.

    Returns (H, d, L) defining a linear Gaussian approximation to the conditional
    distribution p(y | x) ≈ N(y | H x + d, L L^T).

    Args:
        mean_and_chol_cov_function: A callable that returns the conditional mean and
            cholesky factor of the covariance matrix of the distribution for a given
            input.
        x: Point to linearize around.

    Returns:
        Linearized matrix, shift, and cholesky factor of the covariance matrix.

    References:
        Code: https://github.com/EEA-sensors/sqrt-parallel-smoothers/blob/main/parsmooth/linearization/_extended.py
    """

    def mean_and_chol_cov_function_wrapper(
        x: ArrayLike,
    ) -> tuple[Array, tuple[Array, Array]]:
        mean, chol_cov = mean_and_chol_cov_function(x)
        return mean, (mean, chol_cov)

    F, (m, chol_cov) = jax.jacfwd(mean_and_chol_cov_function_wrapper, has_aux=True)(x)
    b = m - F @ x
    return F, b, chol_cov
