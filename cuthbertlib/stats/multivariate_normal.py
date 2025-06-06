from functools import partial

import numpy as np

from jax import lax
from jax import numpy as jnp
from jax._src.numpy.util import promote_dtypes_inexact
from jax._src.typing import Array, ArrayLike


def logpdf(x: ArrayLike, mean: ArrayLike, chol_cov: ArrayLike) -> ArrayLike:
    """Multivariate normal log probability distribution function
    with (generalized) Cholesky factor of covariance input.

    Modified version of `jax.scipy.stats.multivariate_normal.logpdf` which takes
    full covariance matrix as input.

    Args:
      x: arraylike, value at which to evaluate the PDF
      mean: arraylike, centroid of distribution
      cov: arraylike, covariance matrix of distribution

    Returns:
      array of logpdf values.
    """
    x, mean, chol_cov = promote_dtypes_inexact(x, mean, chol_cov)
    if not mean.shape:
        return -1 / 2 * jnp.square(x - mean) / chol_cov**2 - 1 / 2 * (
            jnp.log(2 * np.pi) + 2 * jnp.log(chol_cov)
        )
    else:
        n = mean.shape[-1]
        if not np.shape(chol_cov):
            y = x - mean
            return -1 / 2 * jnp.einsum("...i,...i->...", y, y) / chol_cov**2 - n / 2 * (
                jnp.log(2 * np.pi) + 2 * jnp.log(chol_cov)
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


def pdf(x: ArrayLike, mean: ArrayLike, chol_cov: ArrayLike) -> Array:
    """Multivariate normal probability distribution function
    with (generalized) Cholesky factor of covariance input.

    Modified version of `jax.scipy.stats.multivariate_normal.pdf` which takes
    full covariance matrix as input.

    Args:
      x: arraylike, value at which to evaluate the PDF
      mean: arraylike, centroid of distribution
      cov: arraylike, covariance matrix of distribution

    Returns:
      array of pdf values.
    """
    return lax.exp(logpdf(x, mean, chol_cov))
