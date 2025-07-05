import jax.numpy as jnp

from cuthbertlib.linalg import tria
from cuthbertlib.types import Array, ArrayLike


def symmetric_inv_sqrt(A: ArrayLike, rtol: float | ArrayLike | None = None) -> Array:
    """Compute the inverse square root of a symmetric matrix.

    I.e. a lower triangular matrix L such that L @ L.T = A^{-1} (for positive definite
    A), note that this is not unique and will generally not match the Cholesky factor
    of A^{-1}.

    For singular matrices, small singular values will be cut off reminiscent of
    the Moore-Penrose pseudoinverse - https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.pinv.html.

    In the case of singular or indefinite A, the output will be an approximation
    and L @ L.T = A^{-1} will not hold in general.

    Args:
        A: A symmetric matrix.
        rtol: The relative tolerance for the singular values.
            Cutoff for small singular values; singular values smaller than
            `rtol * largest_singular_value` are treated as zero.
            See https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.pinv.html.

    Returns:
        A lower triangular matrix L such that L @ L.T = A^{-1}.
    """
    arr = jnp.asarray(A)

    # From https://github.com/jax-ml/jax/blob/75d8702023fca6fe4a223bf1e08545c1c80581c0/jax/_src/numpy/linalg.py#L972
    if rtol is None:
        max_rows_cols = max(arr.shape[-2:])
        rtol = jnp.asarray(10.0 * max_rows_cols * jnp.finfo(arr.dtype).eps)
    u, s, _ = jnp.linalg.svd(A, full_matrices=False, hermitian=True)
    cutoff = rtol * s[0]
    s = jnp.where(s > cutoff, s, jnp.inf).astype(u.dtype)
    B = u * (1.0 / jnp.sqrt(s))  # Square root but not lower triangular
    L = tria(B)  # Make lower triangular
    return L
