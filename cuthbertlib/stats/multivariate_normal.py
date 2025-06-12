from typing import Any
import numpy as np
from jax import lax, tree
from jax import numpy as jnp
from jax._src.numpy.util import promote_dtypes_inexact

from cuthbertlib.types import Array, ArrayLike
from cuthbertlib.linalg import tria


def collect_nans_chol(flag: ArrayLike, chol: ArrayLike, *rest: Any) -> Any:
    """
    Converts a generalized Cholesky factor of a covariance matrix with NaNs
    into an ordered generalized Cholesky factor with NaNs rows and columns
    moved to the end with diagonal elements set to 1.

    Also reorders the rest of the arguments in the same way along the first axis.

    Example behavior:
    ```
    flag = jnp.array([False, True, False, True])
    new_flag, new_chol, new_mean = collect_nans_chol(flag, chol, mean)
    ```

    Args:
        flag: Array, boolean array indicating which entries are NaN
            True for NaN entries, False for valid
        chol: Array, Cholesky factor of the covariance matrix
        rest: Any, rest of the arguments to be reordered in the same way
            along the first axis

    Returns:
        flag, chol and rest reordered so that valid entries are first and NaNs are last.
            Diagonal elements of chol are set to 1 for the NaN coordinates to avoid
            division by zero
    """
    flag = jnp.asarray(flag)

    # group the NaN entries together
    argsort = jnp.argsort(flag, stable=True)
    chol = jnp.where(flag[:, None], 0.0, chol)
    chol = chol[argsort]
    flag = flag[argsort]

    # compute the tria of the covariance matrix with NaNs set to 0
    chol = tria(chol)

    # set the diagonal of chol_cov to 1 where nans were present to avoid division by zero
    diag_chol = jnp.diag(chol)
    diag_chol = jnp.where(flag, 1.0, diag_chol)
    diag_indices = jnp.diag_indices_from(chol)
    chol = chol.at[diag_indices].set(diag_chol)

    return flag, chol, *tree.map(lambda x: x[argsort], rest)


def logpdf(
    x: ArrayLike, mean: ArrayLike, chol_cov: ArrayLike, nan_support: bool = True
) -> ArrayLike:
    """Multivariate normal log probability distribution function
    with (generalized) Cholesky factor of covariance input.

    Modified version of `jax.scipy.stats.multivariate_normal.logpdf` which takes
    full covariance matrix as input.

    Args:
      x: arraylike, value at which to evaluate the PDF
      mean: arraylike, centroid of distribution
      chol_cov: arraylike,
        generalized Cholesky factor of the covariance matrix of distribution
      nan_support: bool, if True, ignores NaNs in x by projecting the distribution onto
        the lower-dimensional subspace spanned by the non-NaN entries of x
        Note that `nan_support=True` uses tria (QR decomposition) and therefore
        increases the internal complexity of the function from O(n^2) to O(n^3).

    Returns:
      array of logpdf values.
    """
    x, mean, chol_cov = promote_dtypes_inexact(x, mean, chol_cov)

    if not mean.shape:
        # nan-support is not relevant for scalar case
        return -1 / 2 * jnp.square(x - mean) / chol_cov**2 - 1 / 2 * (
            jnp.log(2 * np.pi) + 2 * jnp.log(chol_cov)
        )
    else:
        n = mean.shape[-1]
        if nan_support:
            flag = jnp.isnan(x)
            # count nans in x
            n_nan = jnp.sum(flag, axis=-1)

            # set nans to the mean
            x = jnp.where(flag, mean, x)
        else:
            n_nan = 0
            flag = jnp.zeros(x.shape, dtype=bool)

        if not np.shape(chol_cov):
            y = x - mean
            return -1 / 2 * jnp.einsum("i,i->", y, y) / chol_cov**2 - (
                n - n_nan
            ) / 2 * (jnp.log(2 * np.pi) + 2 * jnp.log(chol_cov))
        elif chol_cov.ndim == 1:
            if chol_cov.shape[0] != n:
                raise ValueError("multivariate_normal.logpdf got incompatible shapes")
            if nan_support:
                # set nans in chol_cov to 1
                chol_cov = jnp.where(flag, 1.0, chol_cov)
            y = (x - mean) / chol_cov
            return (
                -1 / 2 * jnp.einsum("i,i->", y, y)
                - (n - n_nan) / 2 * jnp.log(2 * np.pi)
                - jnp.log(chol_cov).sum()
            )

        else:
            if chol_cov.shape != (n, n):
                raise ValueError("multivariate_normal.logpdf got incompatible shapes")
            if nan_support:
                # This is a tria based implementation of marginal covariance
                # The idea is to group the NaN entries together and then for the tria operation of [observed, rest] obtain the marginal covariance.

                # group the NaN entries together
                flag, chol_cov, mean, x = collect_nans_chol(flag, chol_cov, mean, x)
                x = jnp.asarray(x)
                mean = jnp.asarray(mean)

            y = lax.linalg.triangular_solve(
                chol_cov, x - mean, lower=True, transpose_a=True
            )
            return (
                -1 / 2 * jnp.einsum("i,i->", y, y)
                - (n - n_nan) / 2 * jnp.log(2 * np.pi)
                - jnp.log(jnp.abs(jnp.diag(chol_cov))).sum(-1)
            )


def pdf(
    x: ArrayLike, mean: ArrayLike, chol_cov: ArrayLike, nan_support: bool = True
) -> Array:
    """Multivariate normal probability distribution function
    with (generalized) Cholesky factor of covariance input.

    Modified version of `jax.scipy.stats.multivariate_normal.pdf` which takes
    full covariance matrix as input.

    Args:
      x: arraylike, value at which to evaluate the PDF
      mean: arraylike, centroid of distribution
      chol_cov: arraylike,
        generalized Cholesky factor of the covariance matrix of distribution
    nan_support: bool, if True, ignores NaNs in x by projecting the distribution onto
        the lower-dimensional subspace spanned by the non-NaN entries of x
        Note that `nan_support=True` uses tria (QR decomposition) and therefore
        increases the internal complexity of the function from O(n^2) to O(n^3).

    Returns:
      array of pdf values.
    """
    return lax.exp(logpdf(x, mean, chol_cov, nan_support))
