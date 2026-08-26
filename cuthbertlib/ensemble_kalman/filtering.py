"""Implements the Ensemble Kalman Filter (EnKF) predict and update steps.

See Algorithm 10.2, [Sanz-Alonso et al., Inverse Problems and Data Assimilation](https://arxiv.org/abs/1810.06191).
Based in part on the [CD-Dynamax implementation](https://github.com/hd-UQ/cd_dynamax/blob/public/cd_dynamax/src/continuous_discrete_nonlinear_gaussian_ssm/inference_enkf.py).
"""

from typing import Callable

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.linalg import cho_solve

from cuthbertlib.linalg import collect_nans_chol, tria
from cuthbertlib.stats import multivariate_normal
from cuthbertlib.types import Array, KeyArray, ScalarArray

ObservationFn = Callable[[Array], Array]
DynamicsFn = Callable[[Array, KeyArray], Array]
CrossCovarianceModifier = Callable[[Array], Array]
ConstructCholInnovationCovariance = Callable[[Array, Array], Array]


def no_covariance_modifier(covariance: Array) -> Array:
    """Return an empirical covariance unchanged.

    The identity covariance modifier, used as the default when no modification
    (e.g. localization) is requested.

    Args:
        covariance: Empirical covariance matrix.

    Returns:
        The covariance matrix, unchanged.
    """
    return covariance


def predict(
    key: KeyArray,
    ensemble: Array,
    dynamics_fn: DynamicsFn,
    inflation: float = 0.0,
) -> Array:
    """Propagate ensemble members through an arbitrary simulator p(x_{t+1} | x_t).

    Args:
        key: JAX PRNG key.
        ensemble: Ensemble of state vectors, shape (N, x_dim).
        dynamics_fn: Dynamics function mapping (state, key) -> state.
        inflation: Multiplicative inflation factor applied to ensemble deviations.

    Returns:
        Predicted ensemble, shape (N, x_dim).
    """
    N, x_dim = ensemble.shape

    # Propagate each member through the dynamics
    keys = random.split(key, N)
    propagated = jax.vmap(dynamics_fn, (0, 0))(ensemble, keys)

    # Apply multiplicative inflation
    mean = jnp.mean(propagated, axis=0)
    propagated = mean + (1 + inflation) * (propagated - mean)

    return propagated


def update(
    key: KeyArray,
    predicted_ensemble: Array,
    observation_fn: ObservationFn,
    chol_R: Array,
    y: Array,
    perturbed_obs: bool = True,
    cross_covariance_modifier: CrossCovarianceModifier = no_covariance_modifier,
    construct_chol_innovation_covariance: ConstructCholInnovationCovariance
    | None = None,
) -> tuple[Array, ScalarArray]:
    """Update ensemble members with an observation using the EnKF update.

    NaNs in ``y`` are treated as missing dimensions and are excluded from the
    update. When ``y`` is entirely NaN, the update is a no-op: the predicted
    ensemble is returned unchanged with zero log-likelihood contribution.

    Args:
        key: JAX PRNG key.
        predicted_ensemble: Predicted ensemble, shape (N, x_dim).
        observation_fn: Observation function mapping state -> obs.
        chol_R: Cholesky factor of the observation noise covariance, shape (y_dim, y_dim).
        y: Observation vector, shape (y_dim,). NaNs indicate missing dimensions.
        perturbed_obs: If True, use perturbed observations (stochastic EnKF).
            If False, use deterministic update.
        cross_covariance_modifier: Function that modifies the empirical
            state-observation cross-covariance, shape (x_dim, y_dim), and returns
            an array with the same shape. Defaults to the identity.
        construct_chol_innovation_covariance: Optional function that
            receives normalized observation deviations with shape (y_dim, N) and
            ``chol_R`` with shape (y_dim, y_dim). It must return a generalized
            Cholesky factor of the complete innovation covariance with shape
            (y_dim, y_dim). The deviations have already been divided by
            ``sqrt(N - 1)``. Both inputs use the original observation order.
            ``None`` uses the standard, unlocalized square-root construction.

    Returns:
        Tuple of (updated_ensemble, log_likelihood).
    """
    N, x_dim = predicted_ensemble.shape

    # Map ensemble to observation space
    y_pred = jax.vmap(observation_fn, (0,))(predicted_ensemble)
    x_mean = jnp.mean(predicted_ensemble, axis=0)
    x_dev = predicted_ensemble - x_mean

    missing = jnp.isnan(y)

    # Modify or construct covariances before reordering due to NaNs.
    argsort = jnp.argsort(missing, stable=True)
    original_y_dev = y_pred - jnp.mean(y_pred, axis=0)
    normalized_original_y_dev = original_y_dev.T / jnp.sqrt(N - 1)

    C_xy = x_dev.T @ original_y_dev / (N - 1)
    C_xy = cross_covariance_modifier(C_xy)

    if construct_chol_innovation_covariance is not None:
        original_chol_S = construct_chol_innovation_covariance(
            normalized_original_y_dev, chol_R
        )

    # Handle partially-missing observations by reordering and zeroing missing dims.
    # Use y_pred.T because y_pred is (N, y_dim) and we want to reorder along axis 0.
    flag, chol_R, y, y_pred = collect_nans_chol(missing, chol_R, y, y_pred.T)
    y_pred = y_pred.T
    y_dim = y.shape[0]

    y_mean = jnp.mean(y_pred, axis=0)
    y_dev = y_pred - y_mean
    C_xy = C_xy[:, argsort]
    C_xy = jnp.where(flag[None, :], 0.0, C_xy)

    if construct_chol_innovation_covariance is None:
        chol_S = tria(jnp.concatenate([y_dev.T / jnp.sqrt(N - 1), chol_R], axis=1))
    else:
        # The constructor sees the original indexing. Only collect and refactor its
        # result when dimensions are missing; otherwise preserve its returned factor.
        chol_S = jax.lax.cond(
            jnp.any(missing),
            lambda chol: collect_nans_chol(missing, chol)[1],
            lambda chol: chol,
            original_chol_S,
        )

    # Kalman gain: K = C_xy @ S^{-1} = C_xy @ cho_solve(chol_S, I)
    K = cho_solve((chol_S, True), C_xy.T).T

    # Innovation per member
    if perturbed_obs:
        y_n = y[None, :] + (chol_R @ random.normal(key, (y_dim, N))).T
    else:
        y_n = jnp.broadcast_to(y[None, :], (N, y_dim))

    # Update ensemble
    updated = predicted_ensemble + (y_n - y_pred) @ K.T

    # Log-likelihood
    ll = multivariate_normal.logpdf(y, y_mean, chol_S, nan_support=False)

    return updated, jnp.asarray(ll)
