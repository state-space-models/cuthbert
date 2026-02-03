from jax import numpy as jnp, Array

from cuthbert.gaussian.kalman import KalmanFilterState
from cuthbert.gaussian.types import LinearizedKalmanFilterState


def extract_and_join(
    factorial_slice: list[int], state: KalmanFilterState | LinearizedKalmanFilterState
) -> KalmanFilterState | LinearizedKalmanFilterState:
    """Extract factors from a Gaussian factorial state and combine into a joint local state."""
    ...


def extract_and_join_arr(factorial_inds: Array, mean: Array) -> Array:
    """Extract factors from a Gaussian factorial state and combine into a joint local state."""
    return mean[factorial_inds].reshape(-1)


def extract_and_join_chol_cov(factorial_inds: Array, chol_cov: Array) -> Array:
    """Extract factors from a Gaussian factorial state and combine into a joint local state."""
    ...
