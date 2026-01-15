"""Implements inverse square root of a symmetric matrix."""

import jax.numpy as jnp

from cuthbertlib.linalg.tria import tria
from cuthbertlib.types import Array, ArrayLike


def symmetric_inv_sqrt(
    A: ArrayLike,
    rtol: float | ArrayLike | None = None,
    ignore_nan_dims: bool = False,
) -> Array:
    r"""Computes the inverse square root of a symmetric matrix.

    I.e., a lower triangular matrix $L$ such that $L L^{\top} = A^{-1}$ (for positive definite
    $A$). Note that this is not unique and will generally not match the Cholesky factor
    of $A^{-1}$.

    For singular matrices, small singular values will be cut off reminiscent of
    the Moore-Penrose pseudoinverse - https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.pinv.html.

    In the case of singular or indefinite $A$, the output will be an approximation
    and $L L^{\top} = A^{-1}$ will not hold in general.

    Args:
        A: A symmetric matrix.
        rtol: The relative tolerance for the singular values.
            Cutoff for small singular values; singular values smaller than
            `rtol * largest_singular_value` are treated as zero.
            See https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.pinv.html.
        ignore_nan_dims: Whether to treat dimensions with NaN on the diagonal as missing
            and ignore all rows and columns associated with them (with result in those
            dimensions being NaN on the diagonal and zero off-diagonal).

    Returns:
        A lower triangular matrix $L$ such that $L L^{\top} = A^{-1}$ (for valid dimensions).
    """
    arr = jnp.asarray(A)

    # Check for NaNs on the diagonal (missing dimensions)
    diag_vals = jnp.diag(arr)
    nan_diag_mask = jnp.isnan(diag_vals) * ignore_nan_dims

    # Check for dimensions whose row and column are all 0
    zero_mask = jnp.all(arr == 0.0, axis=0) & jnp.all(arr == 0.0, axis=1)

    nan_mask = nan_diag_mask | zero_mask

    # Sort to group valid dimensions first (needed for SVD to work correctly)
    argsort = jnp.argsort(nan_mask, stable=True)
    arr_sorted = arr[argsort[:, None], argsort]
    nan_mask_sorted = nan_mask[argsort]

    # Zero out invalid dimensions before computation
    invalid_mask_2d = ((nan_mask_sorted[:, None]) | (nan_mask_sorted[None, :])) & (
        ignore_nan_dims
    )
    arr_sorted = jnp.where(invalid_mask_2d, 0.0, arr_sorted)

    # Compute inverse square root on sorted, masked matrix
    L_sorted = _symmetric_inv_sqrt(arr_sorted, rtol)

    # Post-process: zero out invalid rows/cols, set NaN on invalid diagonal
    L_sorted = jnp.where(invalid_mask_2d, 0.0, L_sorted)
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
    u, s, _ = jnp.linalg.svd(arr, full_matrices=False, hermitian=True)
    cutoff = rtol * s[0]
    # Use 0 for invalid singular values to avoid inf/NaN propagation in tria
    valid_mask = s > cutoff
    inv_sqrt_s = jnp.where(valid_mask, 1.0 / jnp.sqrt(s), 0.0).astype(u.dtype)
    B = u * inv_sqrt_s  # Square root but not lower triangular
    L = tria(B)  # Make lower triangular
    # Mark dimensions with all 0 rows and columns as NaN
    zero_dims_mask = jnp.all(L == 0.0, axis=0) & jnp.all(L == 0.0, axis=1)
    L = jnp.where(zero_dims_mask[:, None] | zero_dims_mask[None, :], jnp.nan, L)
    return L


def chol_cov_with_nans_to_cov(chol_cov: ArrayLike) -> Array:
    """Converts a Cholesky factor to a covariance matrix.

    NaNs on the diagonal specify dimensions to be ignored.

    Args:
        chol_cov: A Cholesky factor of a covariance matrix with NaNs on the diagonal
            specifying dimensions to be ignored.

    Returns:
        A covariance matrix equivalent to chol_cov @ chol_cov.T in dimensions where
            the Cholesky factor is valid and for invalid dimensions (ones with NaN on the
            diagonal in chol_cov) with NaN on the diagonal and zero off-diagonal.
    """
    chol_cov = jnp.asarray(chol_cov)

    nan_mask = jnp.isnan(jnp.diag(chol_cov))

    # Set all rows and columns with invalid diagonal to zero
    chol_cov = jnp.where(nan_mask[:, None] | nan_mask[None, :], 0, chol_cov)

    # Calculate the covariance matrix
    cov = chol_cov @ chol_cov.T

    # Set the diagonal to NaN
    cov = cov.at[jnp.diag_indices_from(cov)].set(
        jnp.where(nan_mask, jnp.nan, jnp.diag(cov))
    )

    return cov
