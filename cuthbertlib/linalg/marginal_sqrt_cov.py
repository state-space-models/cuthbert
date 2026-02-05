"""Extract marginal square root covariance from a joint square root covariance."""

from typing import Sequence

from jax import numpy as jnp

from cuthbertlib.linalg.tria import tria
from cuthbertlib.types import Array, ArrayLike


def marginal_sqrt_cov(chol_cov: ArrayLike, start: int, end: int) -> Array:
    """Extracts square root submatrix from a joint square root matrix.

    Specifically, returns B such that
    B @ B.T = (chol_cov @ chol_cov.T)[start:end, start:end]

    Args:
        chol_cov: Generalized Cholesky factor of the covariance matrix.
        start: Start index of the submatrix.
        end: End index of the submatrix.

    Returns:
        Lower triangular square root matrix of the marginal covariance matrix.
    """
    chol_cov = jnp.asarray(chol_cov)
    chol_cov_select_rows = chol_cov[start:end, :]
    return tria(chol_cov_select_rows)
