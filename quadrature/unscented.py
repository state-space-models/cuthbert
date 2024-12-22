from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from quadrature.common import SigmaPoints

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


# TODO: More descriptive docstring for alpha, beta, kappa.
# TODO: Defaults alpha=0.5, beta = 2.0 ?
def weights(
    n_dim: int, alpha: float, beta: float, kappa: float | None = None
) -> UnscentedQuadrature:
    """
    Computes the weights associated with the spherical cubature method.
    The number of sigma-points is 2 * n_dim.

    Args:
        n_dim: Dimension of the space.
        alpha: Parameter of the unscented transform.
        beta: Parameter of the unscented transform.
        kappa: Parameter of the unscented transform.
            Default is 3 + n_dim.

    Returns:
        UnscentedQuadrature: The quadrature object with the weights and sigma-points.
    """
    if kappa is None:
        kappa = 3.0 + n_dim

    lamda = alpha**2 * (n_dim + kappa) - n_dim
    wm = jnp.full(2 * n_dim + 1, 1 / (2 * (n_dim + lamda)))

    wm = wm.at[0].set(lamda / (n_dim + lamda))
    wc = wm.at[0].set(lamda / (n_dim + lamda) + (1 - alpha**2 + beta))
    return UnscentedQuadrature(wm=wm, wc=wc, lamda=lamda)
