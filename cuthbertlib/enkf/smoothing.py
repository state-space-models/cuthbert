"""Implements the Ensemble Rauch-Tung-Striebel (EnRTS) smoother update.

Cf. [Raanes (2016)](https://doi.org/10.1002/qj.2728).
"""

import jax.numpy as jnp

from cuthbertlib.types import Array


def update(
    filtered_ensemble: Array,
    predicted_ensemble: Array,
    next_smoothed_ensemble: Array,
) -> tuple[Array, Array]:
    """Applies one EnRTS smoother update step.

    Args:
        filtered_ensemble: Filtered ensemble at time t, shape (N, x_dim).
        predicted_ensemble: Paired forecast ensemble at time t + 1,
            shape (N, next_x_dim).
        next_smoothed_ensemble: Smoothed ensemble at time t + 1,
            shape (N, next_x_dim).

    Returns:
        Tuple of the smoothed ensemble at time t and the EnRTS gain.
    """
    filtered_dev = filtered_ensemble - jnp.mean(filtered_ensemble, axis=0)
    predicted_dev = predicted_ensemble - jnp.mean(predicted_ensemble, axis=0)

    gain = filtered_dev.T @ jnp.linalg.pinv(predicted_dev.T)

    smoothed_ensemble = (
        filtered_ensemble + (next_smoothed_ensemble - predicted_ensemble) @ gain.T
    )
    return smoothed_ensemble, gain
