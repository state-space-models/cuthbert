from typing import NamedTuple, Protocol, runtime_checkable

import jax.numpy as jnp
from jax.typing import ArrayLike

from quadrature.utils import tria


class SigmaPoints(NamedTuple):
    points: ArrayLike
    wm: ArrayLike
    wc: ArrayLike

    @property
    def mean(self):
        return jnp.dot(self.wm, self.points)

    def covariance(self, other=None):
        """
        Computes the covariance between the sigma points and the other sigma points
        Cov[self, other].

        Parameters
        ----------
        other: SigmaPoints, optional
            The other sigma points

        Returns
        -------

        """
        mean = self.mean
        if other is None:
            return _cov(self.wc, self.points, mean, self.points, mean)

        other_mean = other.mean
        return _cov(self.wc, self.points, mean, other.points, other_mean)

    @property
    def sqrt(self):
        """
        Computes the square root of the covariance matrix of the sigma points

        Returns
        -------
        ArrayLike
            The square root of the covariance matrix
        """
        sqrt = jnp.sqrt(self.wc[:, None]) * (self.points - self.mean[None, :])
        sqrt = tria(sqrt.T)
        return sqrt


@runtime_checkable
class Quadrature(Protocol):
    def get_sigma_points(self, m, chol) -> SigmaPoints: ...


def _cov(wc, x_pts, x_mean, y_points, y_mean):
    one = (x_pts - x_mean[None, :]).T * wc[None, :]
    two = y_points - y_mean[None, :]
    return jnp.dot(one, two)
