"""Provides shared types for Gaussian representations in state-space models."""

from typing import NamedTuple, Protocol

from cuthbertlib.kalman import filtering
from cuthbertlib.types import Array, ArrayTree, ArrayTreeLike


### Kalman types
class GetInitParams(Protocol):
    """Protocol for defining the initial distribution of a linear Gaussian SSM."""

    def __call__(self, model_inputs: ArrayTreeLike) -> tuple[Array, Array]:
        """Get initial parameters (m0, chol_P0) from model inputs."""
        ...


class GetDynamicsParams(Protocol):
    """Protocol for defining the dynamics model of a linear Gaussian SSM."""

    def __call__(self, model_inputs: ArrayTreeLike) -> tuple[Array, Array, Array]:
        """Get dynamics parameters (F, c, chol_Q) from model inputs."""
        ...


class GetObservationParams(Protocol):
    """Protocol for defining the observation model of a linear Gaussian SSM."""

    def __call__(
        self, model_inputs: ArrayTreeLike
    ) -> tuple[Array, Array, Array, Array]:
        """Get observation parameters (H, d, chol_R, y) from model inputs."""
        ...


### Shared state type for linearized Kalman filters
class LinearizedKalmanFilterState(NamedTuple):
    """Linearized Kalman filter state."""

    elem: filtering.FilterScanElement
    model_inputs: ArrayTree
    mean_prev: Array

    @property
    def mean(self) -> Array:
        """Filtering mean."""
        return self.elem.b

    @property
    def chol_cov(self) -> Array:
        """Filtering generalised Cholesky covariance."""
        return self.elem.U

    @property
    def log_normalizing_constant(self) -> Array:
        """Log normalizing constant (cumulative)."""
        return self.elem.ell
