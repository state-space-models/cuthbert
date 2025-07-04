from typing import NamedTuple

import jax.numpy as jnp

from cuthbertlib.quadrature.common import SigmaPoints
from cuthbertlib.types import Array

__all__ = ["weights", "UnscentedQuadrature"]


class UnscentedQuadrature(NamedTuple):
    wm: Array
    wc: Array
    lamda: float

    def get_sigma_points(self, m, chol) -> SigmaPoints:
        n_dim = m.shape[0]
        scaled_chol = jnp.sqrt(n_dim + self.lamda) * chol

        zeros = jnp.zeros((1, n_dim))
        sigma_points = m[None, :] + jnp.concatenate(
            [zeros, scaled_chol.T, -scaled_chol.T], axis=0
        )
        return SigmaPoints(sigma_points, self.wm, self.wc)


def weights(
    n_dim: int, alpha: float = 0.5, beta: float = 2.0, kappa: float | None = None
) -> UnscentedQuadrature:
    """
    Computes the weights associated with the unscented cubature method.
    The number of sigma-points is 2 * n_dim.
    This method is also known as the Unscented Transform, and generalizes the
    `cubature.py` weights: the cubature method is a special case of the unscented
    for the parameters alpha=1.0, beta=0.0, kappa=0.0.

    Args:
        n_dim: Dimension of the space.
        alpha: Parameter of the unscented transform, default is 0.5.
        beta: Parameter of the unscented transform, default is 2.0.
        kappa: Parameter of the unscented transform, default is 3 + n_dim.

    Returns:
        UnscentedQuadrature: The quadrature object with the weights and sigma-points.

    References:
        - https://groups.seas.harvard.edu/courses/cs281/papers/unscented.pdf
    """
    if kappa is None:
        kappa = 3.0 + n_dim

    lamda = alpha**2 * (n_dim + kappa) - n_dim
    wm = jnp.full(2 * n_dim + 1, 1 / (2 * (n_dim + lamda)))

    wm = wm.at[0].set(lamda / (n_dim + lamda))
    wc = wm.at[0].set(lamda / (n_dim + lamda) + (1 - alpha**2 + beta))
    return UnscentedQuadrature(wm=wm, wc=wc, lamda=lamda)
