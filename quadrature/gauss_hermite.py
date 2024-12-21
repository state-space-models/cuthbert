from typing import NamedTuple
from itertools import product

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from jax.typing import ArrayLike

import quadrature.cubature as cubature
from quadrature.common import Quadrature, SigmaPoints

__all__ = ["weights", "GaussHermiteQuadrature"]


class GaussHermiteQuadrature(NamedTuple):
    wm: ArrayLike
    wc: ArrayLike
    xi: ArrayLike

    def get_sigma_points(self, m, chol) -> SigmaPoints:
        return cubature.get_sigma_points(m, chol, self.xi, self.wm, self.wc)


def weights(n_dim: int, order: int = 3) -> Quadrature:
    """Computes the weights associated with the Gauss--Hermite quadrature method.
    The Hermite polynomial is in the physician version
    Parameters
    ----------
    n_dim: int
        Dimensionality of the problem
    order: int, optional, default is 3
        The order of Hermite polynomial
    Returns
    -------
    Quadrature
        The quadrature object with the weights and sigma-points

    References
    ----------
    .. [1] Simo Särkkä.
       *Bayesian Filtering and Smoothing.*
       In: Cambridge University Press 2013.
    """
    n = n_dim
    p = order

    x, w = hermegauss(p)
    xn = np.array(list(product(*(x,) * n)))
    wn = np.prod(np.array(list(product(*(w,) * n))), 1)
    wn /= np.sqrt(2 * np.pi) ** n
    return GaussHermiteQuadrature(wm=wn, wc=wn, xi=xn)
