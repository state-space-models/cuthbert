from typing import Optional, NamedTuple

import jax.numpy as jnp
from jax.typing import ArrayLike

from quadrature.common import SigmaPoints

__all__ = ["weights", "UncentedQuadrature"]


class UncentedQuadrature(NamedTuple):
    wm: ArrayLike
    wc: ArrayLike
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
    n_dim: int, alpha: float, beta: float, kappa: Optional[float] = None
) -> UncentedQuadrature:
    """Computes the weights associated with the spherical cubature method.
    The number of sigma-points is 2 * n_dim

    Parameters
    ----------
    n_dim: int
        Dimension of the space
    alpha, beta, kappa: float, optional
        Parameters of the unscented transform. Default is `alpha=0.5`, `beta=2.` and `kappa=3-n`

    Returns
    -------
    UncentedQuadrature
        The quadrature object with the weights and sigma-points
    """
    if kappa is None:
        kappa = 3.0 + n_dim

    lamda = alpha**2 * (n_dim + kappa) - n_dim
    wm = jnp.full(2 * n_dim + 1, 1 / (2 * (n_dim + lamda)))

    wm = wm.at[0].set(lamda / (n_dim + lamda))
    wc = wm.at[0].set(lamda / (n_dim + lamda) + (1 - alpha**2 + beta))
    return UncentedQuadrature(wm=wm, wc=wc, lamda=lamda)
