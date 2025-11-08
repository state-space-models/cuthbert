from typing import NamedTuple

import jax.numpy as jnp
from jax import random

from cuthbertlib.types import Array, ArrayTree


class ParamStruct(NamedTuple):
    a: Array
    b: Array


##### Hyperparameters #####
X_DIM = 6
Y_DIM = 78
SIGMA = 1.0
DELTA = 0.1
T = 25


def get_dynamics_params():
    A = jnp.kron(jnp.array([[1, DELTA], [0, 1]]), jnp.eye(3))
    _chol_Q = SIGMA * jnp.linalg.cholesky(
        jnp.array([[DELTA**3 / 3, DELTA**2 / 2], [DELTA**2 / 2, DELTA]])
    )
    chol_Q = jnp.kron(_chol_Q, jnp.eye(3))
    return A, chol_Q


def get_observation_rate(params: ParamStruct, state: Array):
    rate = params.a + params.b @ state
    return rate


def model_factory(params: ParamStruct, ys: Array):
    def get_init_params(model_inputs: int) -> tuple[Array, Array]:
        return jnp.zeros(X_DIM), jnp.ones((X_DIM, X_DIM)) * 1e-8

    def get_dynamics_moments(state: ArrayTree, model_inputs: int):
        def dynamics_mean_and_chol_cov_func(x):
            A, chol_Q = get_dynamics_params()
            return A @ x, jnp.zeros(X_DIM), chol_Q

        return dynamics_mean_and_chol_cov_func, state

    def get_observation_moments(state: ArrayTree, model_inputs: int):
        def observation_mean_and_chol_cov_func(x):
            rate = get_observation_rate(params, x)
            return rate, jnp.diag(jnp.sqrt(rate))

        return (observation_mean_and_chol_cov_func, state, ys[model_inputs])

    return get_init_params, get_dynamics_moments, get_observation_moments


def sim_data(num_time_steps: int):
    state = jnp.zeros(X_DIM)
    ys = []
    A, chol_Q = get_dynamics_params()

    key = random.key(0)

    # Sample the true parameters
    key, a_key, b_key = random.split(key, 3)
    true_a = 2.5 + random.normal(a_key, (Y_DIM,))
    # bs are uniformly distributed on the 6-dimensional sphere
    true_b = random.normal(b_key, (Y_DIM, X_DIM))
    b_norms = jnp.linalg.norm(true_b, axis=1, keepdims=True)
    true_b /= b_norms + 1e-12
    true_params = ParamStruct(true_a, true_b)

    for _ in range(num_time_steps):
        key, dyn_key, obs_key = random.split(key, 3)
        state = A @ state + chol_Q @ random.normal(dyn_key, (X_DIM,))

        rate = get_observation_rate(true_params, state)
        y = random.poisson(obs_key, rate)
        ys.append(y)

    ys = jnp.array(ys)
    # No observation at time 0
    ys = jnp.concatenate([jnp.full((1, Y_DIM), jnp.nan), ys], axis=0)

    return ys, true_params


out = sim_data(T)
