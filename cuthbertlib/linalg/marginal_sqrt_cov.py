"""Extract marginal square root covariance(s) from a joint square root covariance."""

from jax import numpy as jnp
from jax import vmap
from jax.lax import dynamic_slice

from cuthbertlib.linalg.tria import tria
from cuthbertlib.types import Array, ArrayLike


def marginal_sqrt_cov(chol_cov: ArrayLike, start: int, size: int) -> Array:
    """Extracts square root submatrix from a joint square root matrix.

    Specifically, returns B such that
    B @ B.T = (chol_cov @ chol_cov.T)[start:start+size, start:start+size]

    Args:
        chol_cov: Generalized Cholesky factor of the covariance matrix.
        start: Start index of the submatrix.
        size: Number of contiguous rows/columns of the marginal block.

    Returns:
        Lower triangular square root matrix of the marginal covariance matrix.
    """
    chol_cov = jnp.asarray(chol_cov)

    assert chol_cov.ndim == 2, "chol_cov must be a 2D array"
    assert chol_cov.shape[0] == chol_cov.shape[1], "chol_cov must be square"
    # assert start >= 0 and start + size <= chol_cov.shape[0], (
    #     "start and start + size must be within the bounds of chol_cov"
    # ) We don't assert based on start since start doesn't need to be a static argument
    assert size > 0, "size must be positive"

    slice_sizes = (size, chol_cov.shape[1])
    chol_cov_select_rows = dynamic_slice(chol_cov, (start, 0), slice_sizes)
    return tria(chol_cov_select_rows)


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

    assert chol_cov.ndim == 2, "chol_cov must be a 2D array"
    assert chol_cov.shape[0] == chol_cov.shape[1], "chol_cov must be square"
    assert subdim > 0 and chol_cov.shape[0] % subdim == 0, (
        "subdim must be a positive divisor of the number of rows in chol_cov"
    )

    n_blocks = chol_cov.shape[0] // subdim
    return vmap(lambda i: marginal_sqrt_cov(chol_cov, i * subdim, subdim))(
        jnp.arange(n_blocks)
    )
