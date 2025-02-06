import jax
import jax.numpy as jnp
from jax import Array


def tria(A: Array) -> Array:
    """A triangularization operator using QR decomposition.

    Args:
        A: The matrix to triangularize.

    Returns:
        A lower triangular matrix R such that R @ R.T = A @ A.T.
    """
    _, R = jax.scipy.linalg.qr(A.T, mode="economic")
    return R.T


def mvn_logpdf(x: Array, chol_cov: Array) -> Array:
    """Log pdf of a zero-mean multivariate normal.

    Args:
        x: The point at which to evaluate the log pdf.
        chol_cov: The generalized Cholesky factor of the covariance matrix.

    Returns:
        The log pdf of the multivariate normal at x.
    """
    dim = chol_cov.shape[0]
    y = jax.scipy.linalg.solve_triangular(chol_cov, x, lower=True)
    normalizing_constant = (
        jnp.sum(jnp.log(jnp.abs(jnp.diag(chol_cov)))) + dim * jnp.log(2 * jnp.pi) / 2.0
    )
    norm_y = jnp.sum(y**2, -1)
    return -0.5 * norm_y - normalizing_constant
