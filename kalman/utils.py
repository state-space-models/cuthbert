import jax
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
