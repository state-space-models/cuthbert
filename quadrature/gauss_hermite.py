import math
from typing import List, NamedTuple
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


def weights_np(n_dim: int, order: int = 3) -> Quadrature:
    n = n_dim
    p = order

    x, w = hermegauss(p)
    xn = np.array(list(product(*(x,) * n)))
    wn = np.prod(np.array(list(product(*(w,) * n))), 1)
    wn /= np.sqrt(2 * np.pi) ** n
    return GaussHermiteQuadrature(wm=wn, wc=wn, xi=xn)


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

    hermite_coeff = _hermite_coeff(p)
    hermite_roots = np.flip(np.roots(hermite_coeff[-1]))

    table = np.zeros(shape=(n, p**n))

    w_1d = np.zeros(shape=(p,))
    for i in range(p):
        w_1d[i] = (
            2 ** (p - 1)
            * math.factorial(p)
            * np.sqrt(np.pi)
            / (p**2 * (np.polyval(hermite_coeff[p - 1], hermite_roots[i])) ** 2)
        )

    # Get roll table
    for i in range(n):
        base = np.ones(shape=(1, p ** (n - i - 1)))
        for j in range(1, p):
            base = np.concatenate(
                [base, (j + 1) * np.ones(shape=(1, p ** (n - i - 1)))], axis=1
            )
        table[n - i - 1, :] = np.tile(base, (1, int(p**i)))

    table = table.astype("int64") - 1

    s = 1 / (np.sqrt(np.pi) ** n)

    wm = s * np.prod(w_1d[table], axis=0)
    xi = np.sqrt(2) * hermite_roots[table]

    return GaussHermiteQuadrature(wm=wm, wc=wm, xi=xi.T)


def _hermite_coeff(order: int) -> List:
    """Give the 0 to p-th order physician Hermite polynomial coefficients, where p is the
    order from the argument. The returned coefficients is ordered from highest to lowest.
    Also note that this implementation is different from the np.hermite method.
    Parameters
    ----------
    order:  int
        The order of Hermite polynomial
    Returns
    -------
    H: List
        The 0 to p-th order Hermite polynomial coefficients in a list.
    """
    H0 = np.array([1])
    H1 = np.array([2, 0])

    H = [H0, H1]

    for i in range(2, order + 1):
        H.append(
            2 * np.append(H[i - 1], 0)
            - 2 * (i - 1) * np.pad(H[i - 2], (2, 0), "constant", constant_values=0)
        )

    return H
