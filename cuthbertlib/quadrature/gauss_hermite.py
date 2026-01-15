"""Implements Gauss-Hermite quadrature."""

from itertools import product
from typing import NamedTuple

import numpy as np
from jax.typing import ArrayLike
from numpy.polynomial.hermite_e import hermegauss

from cuthbertlib.quadrature import cubature
from cuthbertlib.quadrature.common import Quadrature, SigmaPoints

__all__ = ["weights", "GaussHermiteQuadrature"]


class GaussHermiteQuadrature(NamedTuple):
    """Gauss-Hermite quadrature.

    Attributes:
        wm: The mean weights.
        wc: The covariance weights.
        xi: The sigma points.
    """

    wm: ArrayLike
    wc: ArrayLike
    xi: ArrayLike

    def get_sigma_points(self, m, chol) -> SigmaPoints:
        """Get the sigma points.

        Args:
            m: The mean.
            chol: The Cholesky factor of the covariance.

        Returns:
            SigmaPoints: The sigma points.
        """
        return cubature.get_sigma_points(m, chol, self.xi, self.wm, self.wc)


def weights(n_dim: int, order: int = 3) -> Quadrature:
    """Computes the weights associated with the Gauss-Hermite quadrature method.

    The Hermite polynomial is in the probabilist's version.

    Args:
        n_dim: Dimensionality of the problem.
        order: The order of Hermite polynomial. Defaults to 3.

    Returns:
        The quadrature object with the weights and sigma-points.

    References:
        Simo Särkkä. *Bayesian Filtering and Smoothing.*
            In: Cambridge University Press 2013.
    """
    x, w = hermegauss(order)
    xn = np.array(list(product(*(x,) * n_dim)))
    wn = np.prod(np.array(list(product(*(w,) * n_dim))), 1)
    wn /= np.sqrt(2 * np.pi) ** n_dim
    return GaussHermiteQuadrature(wm=wn, wc=wn, xi=xn)
