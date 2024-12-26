from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from quadrature.common import Quadrature, SigmaPoints

__all__ = ["weights", "CubatureQuadrature"]


class CubatureQuadrature(NamedTuple):
    wm: ArrayLike
    wc: ArrayLike
    xi: ArrayLike

    def get_sigma_points(self, m: ArrayLike, chol: ArrayLike) -> SigmaPoints:
        return get_sigma_points(m, chol, self.xi, self.wm, self.wc)


def get_sigma_points(
    m: ArrayLike, chol: ArrayLike, xi: ArrayLike, wm: ArrayLike, wc: ArrayLike
) -> SigmaPoints:
    # TODO: Add docstring here
    m = jnp.asarray(m)
    chol = jnp.asarray(chol)
    xi = jnp.asarray(xi)
    wm = jnp.asarray(wm)
    wc = jnp.asarray(wc)
    sigma_points = m[None, :] + jnp.dot(chol, xi.T).T

    return SigmaPoints(sigma_points, wm, wc)


def weights(n_dim: int) -> Quadrature:
    """
    Computes the weights associated with the spherical cubature method.
    The number of sigma-points is 2 * n_dim.

    Args:
        n_dim: Dimensionality of the problem.

    Returns:
        The quadrature object with the weights and sigma-points.

    References:
        Simo Särkkä, Lennard Svensson. *Bayesian Filtering and Smoothing.*
            In: Cambridge University Press 2023.
    """
    wm = np.ones(shape=(2 * n_dim,)) / (2 * n_dim)
    wc = wm
    xi = np.concatenate([np.eye(n_dim), -np.eye(n_dim)], axis=0) * np.sqrt(n_dim)

    return CubatureQuadrature(wm=wm, wc=wc, xi=xi)
