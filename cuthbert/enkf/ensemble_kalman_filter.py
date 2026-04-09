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
    GetEnKFDynamics,
    GetEnKFObservations,
    InitSample,
)
from cuthbert.inference import Filter
from cuthbert.utils import dummy_tree_like
from cuthbertlib import enkf as enkf_lib
from cuthbertlib.linalg import tria
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
    def chol_cov(self) -> Array:
        """Generalised Cholesky factor of the ensemble sample covariance."""
        mean = self.mean
        # Handle both single state (..., N, x_dim) and batched (T, N, x_dim)
        dev = self.ensemble - mean[..., None, :]
        n_minus_1 = jnp.asarray(self.n_particles - 1, dtype=dev.dtype)
        scaled_dev_t = jnp.swapaxes(dev, -1, -2) / jnp.sqrt(n_minus_1)

        if scaled_dev_t.ndim == 2:
            return tria(scaled_dev_t)

        return jax.lax.map(tria, scaled_dev_t)


def build_filter(
    init_sample: InitSample,
    get_dynamics: GetEnKFDynamics,
    get_observations: GetEnKFObservations,
    n_particles: int,
    inflation: float = 0.0,
    perturbed_obs: bool = True,
) -> Filter:
    """Builds an Ensemble Kalman Filter object.

    Args:
        init_sample: Function to sample from the initial distribution.
        get_dynamics: Function to get dynamics function and chol_Q from model inputs.
        get_observations: Function to get observation function, chol_R, and y from model inputs.
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
            init_sample=init_sample,
            n_particles=n_particles,
        ),
        filter_prepare=partial(
            filter_prepare,
            get_dynamics=get_dynamics,
            n_particles=n_particles,
        ),
        filter_combine=partial(
            filter_combine,
            get_dynamics=get_dynamics,
            get_observations=get_observations,
            inflation=inflation,
            perturbed_obs=perturbed_obs,
        ),
        associative=False,
    )


def init_prepare(
    model_inputs: ArrayTreeLike,
    init_sample: InitSample,
    n_particles: int,
    key: KeyArray | None = None,
) -> EnKFState:
    """Prepare the initial state for the EnKF.

    Args:
        model_inputs: Model inputs.
        init_sample: Function to sample from the initial distribution.
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

    # Sample ensemble from initial distribution
    keys = random.split(key, n_particles)
    ensemble = jax.vmap(init_sample)(keys)

    return EnKFState(
        key=key,
        ensemble=ensemble,
        model_inputs=model_inputs,
        log_normalizing_constant=jnp.array(0.0),
    )


def filter_prepare(
    model_inputs: ArrayTreeLike,
    get_dynamics: GetEnKFDynamics,
    n_particles: int,
    key: KeyArray | None = None,
) -> EnKFState:
    """Prepare a state for an EnKF step.

    Args:
        model_inputs: Model inputs.
        get_dynamics: Function to get dynamics function and chol_Q from model inputs
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

    # Infer state shape from get_dynamics
    # We need to index into get_dynamics so that we don't eval_shape over dynamics_fn
    dummy_chol_Q = jax.eval_shape(lambda mi: get_dynamics(mi)[1], model_inputs)
    x_dim = dummy_chol_Q.shape[0]
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
    get_dynamics: GetEnKFDynamics,
    get_observations: GetEnKFObservations,
    inflation: float = 0.0,
    perturbed_obs: bool = True,
) -> EnKFState:
    """Combine previous EnKF state with prepared state for current step.

    Implements the EnKF predict + update cycle.

    Args:
        state_1: EnKF state from the previous time step.
        state_2: EnKF state prepared for the current step.
        get_dynamics: Function to get dynamics function and chol_Q from model inputs.
        get_observations: Function to get observation function, chol_R, and y from model inputs.
        inflation: Multiplicative inflation factor.
        perturbed_obs: If True, use perturbed observations.

    Returns:
        Updated EnKF state.
    """
    key_pred, key_update, key_next = random.split(state_1.key, 3)

    # Predict
    dynamics_fn, chol_Q = get_dynamics(state_2.model_inputs)
    predicted = enkf_lib.predict(
        key_pred,
        state_1.ensemble,
        dynamics_fn,
        chol_Q,
        inflation,
    )

    # Update
    observation_fn, chol_R, y = get_observations(state_2.model_inputs)
    updated, ll = enkf_lib.update(
        key_update,
        predicted,
        observation_fn,
        chol_R,
        y,
        perturbed_obs,
    )

    return EnKFState(
        key=key_next,
        ensemble=updated,
        model_inputs=state_2.model_inputs,
        log_normalizing_constant=state_1.log_normalizing_constant + ll,
    )
