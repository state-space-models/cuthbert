from typing import NamedTuple, Protocol, Self, runtime_checkable

import jax.numpy as jnp

from cuthbertlib.linalg import tria
from cuthbertlib.types import Array, ArrayLike

__all__ = ["SigmaPoints", "Quadrature"]


class SigmaPoints(NamedTuple):
    """
    Represents integration (quadrature) sigma points as a collection of points in the space, weights corresponding to
    mean and covariance calculations.

    Attributes:
        points: The sigma points.
        wm: The mean weights.
        wc: The covariance weights.

    Methods:
        mean: Computes the mean of the sigma points.
        covariance: Computes the covariance between the sigma points and the other sigma points (or itself).
        sqrt: Computes a square root of the covariance matrix of the sigma points.

    References:
        Simo Särkkä, Lennard Svensson. *Bayesian Filtering and Smoothing.*
            In: Cambridge University Press 2023.
    """

    points: Array
    wm: Array
    wc: Array

    @property
    def mean(self) -> Array:
        """
        Computes the mean of the sigma points.

        Returns:
            The mean of the sigma points.
        """
        return jnp.dot(self.wm, self.points)

    # Should this be property too?
    def covariance(self, other: Self | None = None) -> Array:
        """
        Computes the covariance between the sigma points and the other sigma points
        Cov[self, other].

        Args:
            other: The optional other sigma points.

        Returns:
            The covariance matrix.
        """
        mean = self.mean
        if other is None:
            return _cov(self.wc, self.points, mean, self.points, mean)

        other_mean = other.mean
        return _cov(self.wc, self.points, mean, other.points, other_mean)

    @property
    def sqrt(self) -> Array:
        """
        Computes the square root of the covariance matrix of the sigma points.

        Returns:
            The square root of the covariance matrix.
        """
        sqrt = jnp.sqrt(self.wc[:, None]) * (self.points - self.mean[None, :])
        sqrt = tria(sqrt.T)
        return sqrt


@runtime_checkable
class Quadrature(Protocol):
    def get_sigma_points(self, m: ArrayLike, chol: ArrayLike) -> SigmaPoints: ...


def _cov(
    wc: Array,
    x_pts: Array,
    x_mean: Array,
    y_points: Array,
    y_mean: Array,
) -> Array:
    one = (x_pts - x_mean[None, :]).T * wc[None, :]
    two = y_points - y_mean[None, :]
    return jnp.dot(one, two)
