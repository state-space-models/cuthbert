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
from jax import Array

from cuthbert.gaussian import moments

# Load data from csv hosted on particles GitHub repository
csv_url = "https://raw.githubusercontent.com/nchopin/particles/refs/heads/master/particles/datasets/thaldata.csv"
data = pd.read_csv(csv_url, header=None).to_numpy()[0]
data = jnp.array(data)


class Params(NamedTuple):
    rho: Array  ### might want to unconstrain rho here using logit transform
    sigma: Array


def logit_inv(x: Array) -> Array:
    return 1 / (1 + jnp.exp(-x))


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
            return mean, sd

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


T = len(data)
model_inputs = jnp.arange(T + 1)


# Initialize parameters
rho_init = 0.1
sig_init = 0.5**0.5
params = Params(rho=jnp.array([rho_init]), sigma=jnp.array([sig_init]))
