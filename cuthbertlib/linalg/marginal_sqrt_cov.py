"""Extract marginal square root covariance from a joint square root covariance."""

from functools import partial

from jax import jit, vmap
from jax import numpy as jnp
from jax.lax import dynamic_slice

from cuthbertlib.linalg.tria import tria
from cuthbertlib.types import Array, ArrayLike


@partial(jit, static_argnums=(2,))
def marginal_sqrt_cov(chol_cov: ArrayLike, start: int | Array, size: int) -> Array:
    """Extracts square root submatrix from a joint square root matrix.

    Specifically, returns B such that
    B @ B.T = (chol_cov @ chol_cov.T)[start:start+size, start:start+size]

    Args:
        chol_cov: Generalized Cholesky factor of the covariance matrix.
        start: Start index of the submatrix (int or 0-d array for use under vmap).
        size: Number of rows/columns of the marginal block. Must be a Python int
            so that the function can be JIT-compiled.

    Returns:
        Lower triangular square root matrix of the marginal covariance matrix.
    """
    chol_cov = jnp.asarray(chol_cov)
    slice_sizes = (size, chol_cov.shape[1])
    chol_cov_select_rows = dynamic_slice(chol_cov, (start, 0), slice_sizes)
    return tria(chol_cov_select_rows)


@partial(jit, static_argnums=(1,))
def block_marginal_sqrt_cov(chol_cov: ArrayLike, subdim: int) -> Array:
    """Extracts all square root submatrices of specified size from joint square root matrix.

    Args:
        chol_cov: Generalized Cholesky factor of the covariance matrix.
        subdim: Size of the square root submatrices to extract.
            Must be a divisor of the number of rows in chol_cov.

    Returns:
        Array of shape (chol_cov.shape[0] // subdim, subdim, subdim)
        containing the square root submatrices.
    """
    chol_cov = jnp.asarray(chol_cov)
    n_blocks = chol_cov.shape[0] // subdim
    return vmap(lambda i: marginal_sqrt_cov(chol_cov, i * subdim, subdim))(
        jnp.arange(n_blocks)
    )
