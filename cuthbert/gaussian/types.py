from typing import NamedTuple, Protocol

from cuthbert.utils import dummy_tree_like
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
    elem: filtering.FilterScanElement
    model_inputs: ArrayTree
    mean_prev: Array

    @property
    def mean(self) -> Array:
        return self.elem.b

    @property
    def chol_cov(self) -> Array:
        return self.elem.U

    @property
    def log_likelihood(self) -> Array:
        return self.elem.ell


def linearized_kalman_filter_state_dummy_elem(
    mean: Array,
    chol_cov: Array,
    log_likelihood: Array,
    model_inputs: ArrayTree,
    mean_prev: Array,
) -> LinearizedKalmanFilterState:
    """Create a LinearizedKalmanFilterState with a dummy element
    I.e. when associated scan is not used.

    Args:
        mean: Mean of the state.
        chol_cov: Cholesky covariance of the state.
        log_likelihood: Log likelihood of the state.
        model_inputs: Model inputs.
        mean_prev: Mean of the previous state.

    Returns:
        LinearizedKalmanFilterState with a dummy elem attribute.
    """
    return LinearizedKalmanFilterState(
        elem=filtering.FilterScanElement(
            A=dummy_tree_like(chol_cov),
            b=mean,
            U=chol_cov,
            eta=dummy_tree_like(mean),
            Z=dummy_tree_like(chol_cov),
            ell=log_likelihood,
        ),
        model_inputs=model_inputs,
        mean_prev=mean_prev,
    )
