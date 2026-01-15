"""Multivariate normal distribution functions with chol_cov input."""

from functools import partial

import numpy as np
from jax import lax
from jax import numpy as jnp
from jax._src.numpy.util import promote_dtypes_inexact
from jax._src.typing import Array, ArrayLike

from cuthbertlib.linalg import collect_nans_chol


def logpdf(
    x: ArrayLike, mean: ArrayLike, chol_cov: ArrayLike, nan_support: bool = True
) -> Array:
    """Multivariate normal log probability distribution function with chol_cov input.

    Here `chol_cov` is the (generalized) Cholesky factor of the covariance matrix.
    Modified version of `jax.scipy.stats.multivariate_normal.logpdf` which takes
    the full covariance matrix as input.

    Args:
        x: Value at which to evaluate the PDF.
        mean: Mean of the distribution.
        chol_cov: Generalized Cholesky factor of the covariance matrix of the distribution.
        nan_support: If `True`, ignores NaNs in `x` by projecting the distribution onto the
            lower-dimensional subspace spanned by the non-NaN entries of `x`. Note that
            `nan_support=True` uses the [tria][cuthbertlib.linalg.tria] operation (QR
            decomposition), and therefore increases the internal complexity of the function
            from $O(n^2)$ to $O(n^3)$, where $n$ is the dimension of `x`.

    Returns:
        Array of logpdf values.
    """
    x, mean, chol_cov = promote_dtypes_inexact(x, mean, chol_cov)

    # If nan_support is True, we need to collect the NaNs at the top of the covariance matrix
    # this uses a QR decomposition so is more expensive
    if nan_support:
        flag = jnp.isnan(x)
        flag, chol_cov, x, mean = collect_nans_chol(flag, chol_cov, x, mean)
        mean = jnp.asarray(mean)
        x = jnp.asarray(x)
        chol_cov = jnp.asarray(chol_cov)

    if not mean.shape and not np.shape(x):
        # Both mean and x are scalars
        return -1 / 2 * jnp.square(x - mean) / chol_cov**2 - 1 / 2 * (
            jnp.log(2 * np.pi) + 2 * jnp.log(chol_cov)
        )
    else:
        n = mean.shape[-1] if mean.shape else x.shape[-1]
        if not np.shape(chol_cov):
            y = x - mean
            return -1 / 2 * jnp.einsum("...i,...i->...", y, y) / chol_cov**2 - n / 2 * (
                jnp.log(2 * np.pi) + 2 * jnp.log(chol_cov)
            )
        elif chol_cov.ndim == 1:
            y = (x - mean) / chol_cov
            return (
                -1 / 2 * jnp.einsum("...i,...i->...", y, y)
                - n / 2 * jnp.log(2 * np.pi)
                - jnp.log(jnp.abs(chol_cov)).sum(-1)
            )
        else:
            if chol_cov.ndim < 2 or chol_cov.shape[-2:] != (n, n):
                raise ValueError("multivariate_normal.logpdf got incompatible shapes")
            y = jnp.vectorize(
                partial(lax.linalg.triangular_solve, lower=True, transpose_a=True),
                signature="(n,n),(n)->(n)",
            )(chol_cov, x - mean)
            return (
                -1 / 2 * jnp.einsum("...i,...i->...", y, y)
                - n / 2 * jnp.log(2 * np.pi)
                - jnp.log(jnp.abs(chol_cov.diagonal(axis1=-1, axis2=-2))).sum(-1)
            )


def pdf(
    x: ArrayLike, mean: ArrayLike, chol_cov: ArrayLike, nan_support: bool = True
) -> Array:
    """Multivariate normal probability distribution function with chol_cov input.

    Here `chol_cov` is the (generalized) Cholesky factor of the covariance matrix.
    Modified version of `jax.scipy.stats.multivariate_normal.pdf` which takes
    the full covariance matrix as input.

    Args:
        x: Value at which to evaluate the PDF.
        mean: Mean of the distribution.
        chol_cov: Generalized Cholesky factor of the covariance matrix of the distribution.
        nan_support: If `True`, ignores NaNs in `x` by projecting the distribution onto the
            lower-dimensional subspace spanned by the non-NaN entries of `x`. Note that
            `nan_support=True` uses the [tria][cuthbertlib.linalg.tria] operation (QR
            decomposition), and therefore increases the internal complexity of the function
            from $O(n^2)$ to $O(n^3)$, where $n$ is the dimension of `x`.

    Returns:
        Array of pdf values.
    """
    return lax.exp(logpdf(x, mean, chol_cov, nan_support))
