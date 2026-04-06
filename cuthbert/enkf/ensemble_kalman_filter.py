"""Implements the high-level Ensemble Kalman Filter (EnKF).

See Algorithm 10.2, [Sanz-Alonso et al., Inverse Problems and Data Assimilation](https://arxiv.org/abs/1810.06191).
Based in part on the [CD-Dynamax implementation](https://github.com/hd-UQ/cd_dynamax/blob/public/cd_dynamax/src/continuous_discrete_nonlinear_gaussian_ssm/inference_enkf.py).
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import random, tree

from cuthbert.enkf.types import (
    DynamicsFn,
    GetEnKFDynamicsParams,
    GetEnKFInitParams,
    GetEnKFObservationParams,
    ObservationFn,
)
from cuthbert.inference import Filter
from cuthbert.utils import dummy_tree_like
from cuthbertlib import enkf as enkf_lib
from cuthbertlib.types import Array, ArrayTree, ArrayTreeLike, KeyArray, ScalarArray


class EnKFState(NamedTuple):
    """Ensemble Kalman filter state."""

    key: KeyArray
    ensemble: Array
    model_inputs: ArrayTree
    log_normalizing_constant: ScalarArray

    @property
    def n_particles(self) -> int:
        """Number of particles."""
        return self.ensemble.shape[-2]

    @property
    def mean(self) -> Array:
        """Ensemble mean."""
        return jnp.mean(self.ensemble, axis=-2)

    @property
    def cov(self) -> Array:
        """Ensemble sample covariance."""
        mean = self.mean
        # Handle both single state (..., N, x_dim) and batched (T, N, x_dim)
        dev = self.ensemble - mean[..., None, :]
        return jnp.einsum("...ni,...nj->...ij", dev, dev) / (self.n_particles - 1)


def build_filter(
    get_init_params: GetEnKFInitParams,
    dynamics_fn: DynamicsFn,
    get_dynamics_params: GetEnKFDynamicsParams,
    observation_fn: ObservationFn,
    get_observation_params: GetEnKFObservationParams,
    n_particles: int,
    inflation: float = 0.0,
    perturbed_obs: bool = True,
) -> Filter:
    """Builds an Ensemble Kalman Filter object.

    Args:
        get_init_params: Function to get (m0, chol_P0) from model inputs.
        dynamics_fn: Dynamics function mapping (state, model_inputs) -> state.
        get_dynamics_params: Function to get chol_Q from model inputs.
        observation_fn: Observation function mapping (state, model_inputs) -> obs.
        get_observation_params: Function to get (chol_R, y) from model inputs.
        n_particles: Number of particles.
        inflation: Multiplicative inflation factor for ensemble deviations.
        perturbed_obs: If True, use perturbed observations (stochastic EnKF).

    Returns:
        Filter object for the EnKF.

    Raises:
        ValueError: If ``n_particles`` is less than 2.
    """
    if n_particles < 2:
        raise ValueError("n_particles must be at least 2 for EnKF.")

    return Filter(
        init_prepare=partial(
            init_prepare,
            get_init_params=get_init_params,
            n_particles=n_particles,
        ),
        filter_prepare=partial(
            filter_prepare,
            get_init_params=get_init_params,
            n_particles=n_particles,
        ),
        filter_combine=partial(
            filter_combine,
            dynamics_fn=dynamics_fn,
            get_dynamics_params=get_dynamics_params,
            observation_fn=observation_fn,
            get_observation_params=get_observation_params,
            inflation=inflation,
            perturbed_obs=perturbed_obs,
        ),
        associative=False,
    )


def init_prepare(
    model_inputs: ArrayTreeLike,
    get_init_params: GetEnKFInitParams,
    n_particles: int,
    key: KeyArray | None = None,
) -> EnKFState:
    """Prepare the initial state for the EnKF.

    Args:
        model_inputs: Model inputs.
        get_init_params: Function to get (m0, chol_P0) from model inputs.
        n_particles: Number of particles.
        key: JAX random key.

    Returns:
        Initial EnKF state.

    Raises:
        ValueError: If key is None.
    """
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)
    if key is None:
        raise ValueError("A JAX PRNG key must be provided.")

    m0, chol_P0 = get_init_params(model_inputs)
    x_dim = m0.shape[0]

    # Sample ensemble from N(m0, P0)
    keys = random.split(key, n_particles)
    ensemble = jax.vmap(lambda k: m0 + chol_P0 @ random.normal(k, (x_dim,)))(keys)

    return EnKFState(
        key=key,
        ensemble=ensemble,
        model_inputs=model_inputs,
        log_normalizing_constant=jnp.array(0.0),
    )


def filter_prepare(
    model_inputs: ArrayTreeLike,
    get_init_params: GetEnKFInitParams,
    n_particles: int,
    key: KeyArray | None = None,
) -> EnKFState:
    """Prepare a state for an EnKF step.

    Args:
        model_inputs: Model inputs.
        get_init_params: Function to get (m0, chol_P0) from model inputs
            (used to infer state shape).
        n_particles: Number of particles.
        key: JAX random key.

    Returns:
        Prepared EnKF state with dummy ensemble.

    Raises:
        ValueError: If key is None.
    """
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)
    if key is None:
        raise ValueError("A JAX PRNG key must be provided.")

    # Infer state shape from get_init_params
    dummy_m0, _ = jax.eval_shape(get_init_params, model_inputs)
    x_dim = dummy_m0.shape[0]
    ensemble = jnp.empty((n_particles, x_dim))
    ensemble = dummy_tree_like(ensemble)

    return EnKFState(
        key=key,
        ensemble=ensemble,
        model_inputs=model_inputs,
        log_normalizing_constant=jnp.array(0.0),
    )


def filter_combine(
    state_1: EnKFState,
    state_2: EnKFState,
    dynamics_fn: DynamicsFn,
    get_dynamics_params: GetEnKFDynamicsParams,
    observation_fn: ObservationFn,
    get_observation_params: GetEnKFObservationParams,
    inflation: float = 0.0,
    perturbed_obs: bool = True,
) -> EnKFState:
    """Combine previous EnKF state with prepared state for current step.

    Implements the EnKF predict + update cycle.

    Args:
        state_1: EnKF state from the previous time step.
        state_2: EnKF state prepared for the current step.
        dynamics_fn: Dynamics function mapping (state, model_inputs) -> state.
        get_dynamics_params: Function to get chol_Q from model inputs.
        observation_fn: Observation function mapping (state, model_inputs) -> obs.
        get_observation_params: Function to get (chol_R, y) from model inputs.
        inflation: Multiplicative inflation factor.
        perturbed_obs: If True, use perturbed observations.

    Returns:
        Updated EnKF state.
    """
    key_pred, key_update, key_next = random.split(state_1.key, 3)

    # Predict
    chol_Q = get_dynamics_params(state_2.model_inputs)
    predicted = enkf_lib.predict(
        key_pred,
        state_1.ensemble,
        dynamics_fn,
        chol_Q,
        state_2.model_inputs,
        inflation,
    )

    # Update
    chol_R, y = get_observation_params(state_2.model_inputs)
    updated, ll = enkf_lib.update(
        key_update,
        predicted,
        observation_fn,
        chol_R,
        y,
        state_2.model_inputs,
        perturbed_obs,
    )

    return EnKFState(
        key=key_next,
        ensemble=updated,
        model_inputs=state_2.model_inputs,
        log_normalizing_constant=state_1.log_normalizing_constant + ll,
    )
