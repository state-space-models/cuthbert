"""Implements the high-level Ensemble Rauch-Tung-Striebel smoother (EnRTS)."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import tree

from cuthbert.ensemble_kalman.ensemble_kalman_filter import EnKFState
from cuthbert.inference import Smoother
from cuthbert.utils import dummy_tree_like
from cuthbertlib import ensemble_kalman as enkf_lib
from cuthbertlib.linalg import tria
from cuthbertlib.types import Array, ArrayTree, ArrayTreeLike, KeyArray


class EnRTSState(NamedTuple):
    """Ensemble Rauch-Tung-Striebel smoother state."""

    ensemble: Array
    predicted_ensemble: Array
    model_inputs: ArrayTree

    @property
    def n_particles(self) -> int:
        """Number of particles."""
        return self.ensemble.shape[-2]

    @property
    def mean(self) -> Array:
        """Ensemble mean."""
        return jnp.mean(self.ensemble, axis=-2)

    @property
    def chol_cov(self) -> Array:
        """Generalised Cholesky factor of the ensemble sample covariance."""
        mean = self.mean
        dev = self.ensemble - mean[..., None, :]
        n_minus_1 = jnp.asarray(self.n_particles - 1, dtype=dev.dtype)
        scaled_dev_t = jnp.swapaxes(dev, -1, -2) / jnp.sqrt(n_minus_1)

        if scaled_dev_t.ndim == 2:
            return tria(scaled_dev_t)

        return jax.lax.map(tria, scaled_dev_t)


def build_smoother() -> Smoother:
    """Build an Ensemble Rauch-Tung-Striebel smoother object.

    Filtered states must come from an EnKF built with
    ``store_predicted_ensemble=True``.

    Returns:
        Smoother object for the EnRTS.
    """
    return Smoother(
        convert_filter_to_smoother_state=convert_filter_to_smoother_state,
        smoother_prepare=smoother_prepare,
        smoother_combine=smoother_combine,
        associative=False,
    )


def smoother_prepare(
    filter_state: EnKFState,
    model_inputs: ArrayTreeLike,
    key: KeyArray | None = None,
) -> EnRTSState:
    """Prepare a state for an EnRTS step.

    Args:
        filter_state: EnKF state at time t.
        model_inputs: Model inputs for the transition from t to t + 1.
        key: JAX random key; unused.

    Returns:
        Prepared EnRTS state.

    Raises:
        ValueError: If the EnKF did not store predicted ensembles.
    """
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)
    predicted_ensemble = filter_state.predicted_ensemble
    if predicted_ensemble is None:
        raise ValueError(
            "EnRTS requires an EnKF built with store_predicted_ensemble=True."
        )
    return EnRTSState(
        ensemble=filter_state.ensemble,
        predicted_ensemble=predicted_ensemble,
        model_inputs=model_inputs,
    )


def smoother_combine(
    state_1: EnRTSState,
    state_2: EnRTSState,
) -> EnRTSState:
    """Combine a prepared state with the next EnRTS state.

    Args:
        state_1: Prepared state at time t.
        state_2: Smoothed state at time t + 1.

    Returns:
        Smoothed state at time t.
    """
    ensemble, _ = enkf_lib.smoother_update(
        state_1.ensemble,
        state_2.predicted_ensemble,
        state_2.ensemble,
    )
    return EnRTSState(
        ensemble=ensemble,
        predicted_ensemble=state_1.predicted_ensemble,
        model_inputs=state_1.model_inputs,
    )


def convert_filter_to_smoother_state(
    filter_state: EnKFState,
    model_inputs: ArrayTreeLike | None = None,
    key: KeyArray | None = None,
) -> EnRTSState:
    """Convert the final EnKF state to an EnRTS state.

    Requires `filter_state` to contain predicted ensembles (via `store_predicted_ensemble=True` in the filter).

    Args:
        filter_state: Final EnKF state.
        model_inputs: Model inputs used to define the output tree structure.
        key: JAX random key - not used.

    Returns:
        Final EnRTS state with dummy model inputs.

    Raises:
        ValueError: If the EnKF did not store predicted ensembles.
    """
    if model_inputs is None:
        model_inputs = filter_state.model_inputs

    predicted_ensemble = filter_state.predicted_ensemble
    if predicted_ensemble is None:
        raise ValueError(
            "EnRTS requires an EnKF built with store_predicted_ensemble=True."
        )

    return EnRTSState(
        ensemble=filter_state.ensemble,
        predicted_ensemble=predicted_ensemble,
        model_inputs=dummy_tree_like(model_inputs),
    )
