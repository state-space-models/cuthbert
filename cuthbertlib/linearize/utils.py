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

    Dimensions with NaN or 0 on the diagonal are treated as "missing" and the inverse
    square root is computed only on the valid dimensions, with the result padded with
    NaNs or 0s in the appropriate places.

    Args:
        A: A symmetric matrix.
        rtol: The relative tolerance for the singular values.
            Cutoff for small singular values; singular values smaller than
            `rtol * largest_singular_value` are treated as zero.
            See https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.pinv.html.

    Returns:
        A lower triangular matrix L such that L @ L.T = A^{-1} (for valid dimensions).
    """
    arr = jnp.asarray(A)

    # Check for NaNs or zeros on the diagonal
    diag_vals = jnp.diag(arr)
    nan_mask = jnp.isnan(diag_vals) | (diag_vals == 0.0)

    # Sort to group valid dimensions first (needed for SVD to work correctly)
    argsort = jnp.argsort(nan_mask, stable=True)
    arr_sorted = arr[argsort[:, None], argsort]
    nan_mask_sorted = nan_mask[argsort]

    # Zero out invalid dimensions before computation
    valid_mask_2d = (~nan_mask_sorted[:, None]) & (~nan_mask_sorted[None, :])
    arr_sorted = jnp.where(valid_mask_2d, arr_sorted, 0.0)

    # Compute inverse square root on sorted, masked matrix
    L_sorted = _symmetric_inv_sqrt(arr_sorted, rtol)

    # Post-process: zero out invalid rows/cols, set NaN on invalid diagonal
    L_sorted = jnp.where(valid_mask_2d, L_sorted, 0.0)
    diag_L = jnp.where(nan_mask_sorted, jnp.nan, jnp.diag(L_sorted))
    L_sorted = L_sorted.at[jnp.diag_indices_from(L_sorted)].set(diag_L)

    # Un-sort to restore original order
    inv_argsort = jnp.argsort(argsort)
    L = L_sorted[inv_argsort[:, None], inv_argsort]

    return L


def _symmetric_inv_sqrt(A: ArrayLike, rtol: float | ArrayLike | None = None) -> Array:
    """Implementation of symmetric inverse square root without NaN handling."""
    arr = jnp.asarray(A)

    # From https://github.com/jax-ml/jax/blob/75d8702023fca6fe4a223bf1e08545c1c80581c0/jax/_src/numpy/linalg.py#L972
    if rtol is None:
        max_rows_cols = max(arr.shape[-2:])
        rtol = jnp.asarray(10.0 * max_rows_cols * jnp.finfo(arr.dtype).eps)
    u, s, _ = jnp.linalg.svd(A, full_matrices=False, hermitian=True)
    cutoff = rtol * s[0]
    # Use 0 for invalid singular values to avoid inf/NaN propagation in tria
    inv_sqrt_s = jnp.where(s > cutoff, 1.0 / jnp.sqrt(s), 0.0).astype(u.dtype)
    B = u * inv_sqrt_s  # Square root but not lower triangular
    L = tria(B)  # Make lower triangular
    return L
