# type: ignore pyright not used for examples
"""Example using EM to find the MLE of neuroscience experiment from
14.6 in Chopin and Papaspiliopoulos, code at https://github.com/nchopin/particles/blob/master/book/mle/mle_neuro.py

The considered model and data are from Temereanca et al (2008):

    X_0 ~ N(0, sigma^2)
    X_t = rho X_{t-1} + sigma U_t,     U_t ~ N(0, 1)
    Y_t ~ Bin(50, logit_inv(X_t))

    where logit_inv(x) = 1/(1 + exp(-x))

The parameter is theta = (rho, sigma^2), with 0 <= rho <= 1, and sigma^2 >= 0.


MLE is (rho, sigma2) = (0.9981, 0.1089)
"""

from typing import NamedTuple

import jax.numpy as jnp
import pandas as pd
from jax import Array, tree, vmap
from jax.scipy.stats import binom

from cuthbert import filter, smoother

# from jax.scipy.minimize import minimize
from cuthbert.gaussian import moments
from cuthbert.gaussian.kalman import KalmanSmootherState
from cuthbertlib.quadrature.gauss_hermite import weights
from cuthbertlib.stats import multivariate_normal

# Load data from csv hosted on particles GitHub repository
csv_url = "https://raw.githubusercontent.com/nchopin/particles/refs/heads/master/particles/datasets/thaldata.csv"
data = pd.read_csv(csv_url, header=None).to_numpy()[0]
data = jnp.array(data)
data = jnp.concatenate([jnp.array([jnp.nan]), data])


class Params(NamedTuple):
    rho: Array  ### might want to unconstrain rho here using logit transform
    sigma: Array


# Utilities for constraining and unconstraining parameters
class UnconstrainedParams(NamedTuple):
    logit_rho: Array
    log_sigma: Array


def logit(p: Array) -> Array:
    # Converts [0, 1] to [-inf, inf]
    return jnp.log(p / (1 - p))


def logit_inv(x: Array) -> Array:
    # Converts [-inf, inf] to [0, 1]
    return 1 / (1 + jnp.exp(-x))


def constrain_params(params: UnconstrainedParams) -> Params:
    return Params(
        rho=logit_inv(params.logit_rho),
        sigma=jnp.exp(params.log_sigma),
    )


def unconstrain_params(params: Params) -> UnconstrainedParams:
    return UnconstrainedParams(
        logit_rho=logit(params.rho),
        log_sigma=jnp.log(params.sigma),
    )


# Build model objects - this is where the model definition is encapsulated
def model_factory(params: Params):
    def get_init_params(model_inputs: int) -> tuple[Array, Array]:
        return jnp.array([0.0]), jnp.array([params.sigma])

    def get_dynamics_moments(state, model_inputs: int):
        def dynamics_mean_and_chol_cov_func(x):
            return params.rho * x, params.sigma

        return dynamics_mean_and_chol_cov_func, state.mean

    def get_observation_moments(state, model_inputs: int):
        def observation_mean_and_chol_cov_func(x):
            # Binomial parameters
            n = 50
            p = logit_inv(x)
            mean = n * p
            var = n * p * (1 - p)
            sd = jnp.sqrt(var)
            return mean.reshape(
                1,
            ), sd.reshape(1, 1)

        return (
            observation_mean_and_chol_cov_func,
            state.mean,
            data[model_inputs],
        )

    filter_obj = moments.build_filter(
        get_init_params,
        get_dynamics_moments,
        get_observation_moments,
        associative=False,
    )
    smoother_obj = moments.build_smoother(get_dynamics_moments)

    return filter_obj, smoother_obj


# Define model inputs
T = len(data)
model_inputs = jnp.arange(T + 1)


# Initialize parameters
rho_init = 0.1
sig_init = 0.5**0.5
params = Params(rho=jnp.array([rho_init]), sigma=jnp.array([sig_init]))


# Define loss function for M-step
# Use Gauss-Hermite quadrature to approximate the expectation
gauss_hermite_order = 10


def loss_fn(
    unconstrained_params: UnconstrainedParams,
    ys: Array,
    smooth_dist: KalmanSmootherState,
):
    params = constrain_params(unconstrained_params)

    # TODO: Can used vectorized call to multivariate_normal to avoid calling
    # quadrature.get_sigma_points so often inside loss_dynamics and loss_observation
    # I.e. use univariate normal logpdf or vmap multivariate_normal.logpdf rather than
    # vmapping loss_dynamics and loss_observation

    def loss_initial(m, chol_cov):
        # E_{p(x_0 | m, chol_cov)} [log N(x_0 | 0, params.sigma^2)]
        quadrature = weights(1, order=gauss_hermite_order)
        sigma_points = quadrature.get_sigma_points(m, chol_cov)
        # points.shape=wm.shape=wc.shape=(gauss_hermite_order, 1)
        return jnp.dot(
            sigma_points.wm,
            multivariate_normal.logpdf(sigma_points.points, 0.0, params.sigma),
        )

    def loss_dynamics(m_joint, chol_cov_joint):
        # E_{p(x_{t-1}, x_t | m_joint, chol_cov_joint)} [log N(x_t | rho * x_{t-1}, params.sigma^2)]
        quadrature = weights(2, order=gauss_hermite_order)
        sigma_points = quadrature.get_sigma_points(m_joint, chol_cov_joint)
        # points.shape=wm.shape=wc.shape=(gauss_hermite_order, 2)
        # TODO: check this
        return jnp.dot(
            sigma_points.wm,
            multivariate_normal.logpdf(
                sigma_points.points[:, 0],
                sigma_points.points[:, 1] * params.rho,
                params.sigma,
            ),
        )

    def loss_observation(m, chol_cov, y):
        # E_{x_t | m, chol_cov)} [log Bin(y_t | 50, logit_inv(x_t))]
        quadrature = weights(1, order=gauss_hermite_order)
        sigma_points = quadrature.get_sigma_points(m, chol_cov)
        return jnp.dot(
            sigma_points.wm,
            binom.logpmf(y, 50, logit_inv(sigma_points.points)),
        )

    joint_means = jnp.stack([smooth_dist.mean[:-1], smooth_dist.mean[1:]], axis=1)

    def construct_joint_chol_cov(chol_cov_t_plus_1, gain_t, chol_omega_t):
        # From https://github.com/state-space-models/cuthbert/discussions/18
        # TODO: document this or even modify the kalman code to have this more easily accessible
        return jnp.block(
            [
                [chol_cov_t_plus_1, jnp.zeros_like(chol_cov_t_plus_1)],
                [gain_t @ chol_cov_t_plus_1, chol_omega_t],
            ]
        )

    joint_chol_covs = vmap(construct_joint_chol_cov)(
        smooth_dist.chol_cov[1:], smooth_dist.gain[:-1], smooth_dist.chol_omega[:-1]
    )

    total_loss = (
        loss_initial(smooth_dist.mean[0], smooth_dist.chol_cov[0])
        + vmap(loss_dynamics, joint_means, joint_chol_covs)
        + vmap(loss_observation, smooth_dist.mean[1:], smooth_dist.chol_cov[1:], ys[1:])
    )
    return -total_loss


# Run EM!
params_track = tree.map(lambda x: x[None], params)

for epoch in range(10):
    filter_obj, smoother_obj = model_factory(params)
    filtered_states = filter(filter_obj, model_inputs)
    smoother_states = smoother(smoother_obj, filtered_states)
