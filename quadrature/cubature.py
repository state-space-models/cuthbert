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

    def get_sigma_points(self, m, chol) -> SigmaPoints:
        return get_sigma_points(m, chol, self.xi, self.wm, self.wc)


def get_sigma_points(m, chol, xi, wm, wc):
    sigma_points = m[None, :] + jnp.dot(chol, xi.T).T

    return SigmaPoints(sigma_points, wm, wc)


def weights(n_dim: int) -> Quadrature:
    """Computes the weights associated with the spherical cubature method.
    The number of sigma-points is 2 * n_dim

    Parameters
    ----------
    n_dim: int
        Dimensionality of the problem

    Returns
    -------
    Quadrature
        The quadrature object with the weights and sigma-points
    """
    wm = np.ones(shape=(2 * n_dim,)) / (2 * n_dim)
    wc = wm
    xi = np.concatenate([np.eye(n_dim), -np.eye(n_dim)], axis=0) * np.sqrt(n_dim)

    return CubatureQuadrature(wm=wm, wc=wc, xi=xi)
