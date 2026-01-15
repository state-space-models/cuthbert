"""Implements triangularization operator a matrix via QR decomposition."""

import jax

from cuthbertlib.types import Array


def tria(A: Array) -> Array:
    r"""A triangularization operator using QR decomposition.

    Args:
        A: The matrix to triangularize.

    Returns:
        A lower triangular matrix $R$ such that $R R^\top = A A^\top$.

    Reference:
        [Arasaratnam and Haykin (2008)](https://ieeexplore.ieee.org/document/4524036): Square-Root Quadrature Kalman Filtering
    """
    _, R = jax.scipy.linalg.qr(A.T, mode="economic")
    return R.T
