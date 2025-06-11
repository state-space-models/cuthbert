import numpy as np
from jax import lax
from jax import numpy as jnp
from jax._src.numpy.util import promote_dtypes_inexact

from cuthbertlib.types import Array, ArrayLike
from cuthbertlib.linalg import tria


def _nans_chol_cov(flag: Array, chol_cov: Array) -> tuple[Array, Array, Array]:
    """
    Converts a generalized Cholesky factor of a covariance matrix with NaNs
    into an ordered generalized Cholesky factor with NaNs rows and columns
    moved to the end with diagonal elements set to 1.

    Also returns the reordered flag and the reordering vector.

    Args:
        flag: Array, boolean array indicating which entries are NaN
        chol_cov: Array, Cholesky factor of the covariance matrix

    Returns:
        Cholesky factor of the covariance matrix with NaNs set to 0,
        and the indices of the NaNs
    """

    # group the NaN entries together
    argsort = jnp.argsort(flag, stable=True)
    chol_cov = jnp.where(flag[:, None], 0.0, chol_cov)
    chol_cov = chol_cov[argsort]
    flag = flag[argsort]

    # compute the tria of the covariance matrix with NaNs set to 0
    chol_cov = tria(chol_cov)

    # set the diagonal of chol_cov to 1 where nans were present to avoid division by zero
    diag_chol_cov = jnp.diag(chol_cov)
    diag_chol_cov = jnp.where(flag, 1.0, diag_chol_cov)
    diag_indices = jnp.diag_indices_from(chol_cov)
    chol_cov = chol_cov.at[diag_indices].set(diag_chol_cov)
    return chol_cov, flag, argsort


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

                chol_cov, flag, argsort = _nans_chol_cov(flag, chol_cov)
                mean = mean[argsort]
                x = x[argsort]

                # # group the NaN entries together
                # argsort = jnp.argsort(flag, stable=True)
                # chol_cov = jnp.where(flag[:, None], 0.0, chol_cov)
                # chol_cov, x, mean = chol_cov[argsort], x[argsort], mean[argsort]
                # flag = flag[argsort]

                # # compute the tria of the covariance matrix with NaNs set to 0
                # chol_cov = tria(chol_cov)

                # # set the diagonal of chol_cov to 1 where nans were present to avoid division by zero
                # diag_chol_cov = jnp.diag(chol_cov)
                # diag_chol_cov = jnp.where(flag, 1.0, diag_chol_cov)
                # diag_indices = jnp.diag_indices_from(chol_cov)
                # chol_cov = chol_cov.at[diag_indices].set(diag_chol_cov)

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
