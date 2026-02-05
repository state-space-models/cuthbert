import itertools

import jax
import jax.numpy as jnp
import pytest
from jax import Array, vmap

from cuthbert import factorial
from cuthbert.gaussian import kalman
from cuthbert.inference import Filter, Smoother
from cuthbertlib.kalman.generate import generate_lgssm
from tests.cuthbertlib.kalman.test_filtering import std_predict, std_update
from tests.cuthbertlib.kalman.test_smoothing import std_kalman_smoother
from tests.cuthbert.factorial.gaussian_utils import generate_factorial_kalman_model


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def load_kalman_pairwise_factorial_inference(
    m0: Array,  # (F, d)
    chol_P0: Array,  # (F, d, d)
    Fs: Array,  # (T, 2 * d, 2 * d)
    cs: Array,  # (T, 2 * d)
    chol_Qs: Array,  # (T, 2 * d, 2 * d)
    Hs: Array,  # (T+1, d_y, 2 * d) with nans for initial time step
    ds: Array,  # (T+1, d_y) with nans for initial time step
    chol_Rs: Array,  # (T+1, d_y, d_y) with nans for initial time step
    ys: Array,  # (T+1, d_y) with nans for initial time step
    factorial_indices: Array,  # (T, 2)
) -> tuple[Filter, Smoother, factorial.Factorializer, Array]:
    """Builds Kalman filter and smoother objects and model_inputs for a linear-Gaussian SSM."""

    def get_init_params(model_inputs: int) -> tuple[Array, Array]:
        return m0, chol_P0

    def get_dynamics_params(model_inputs: int) -> tuple[Array, Array, Array]:
        return Fs[model_inputs - 1], cs[model_inputs - 1], chol_Qs[model_inputs - 1]

    def get_observation_params(model_inputs: int) -> tuple[Array, Array, Array, Array]:
        return (
            Hs[model_inputs],
            ds[model_inputs],
            chol_Rs[model_inputs],
            ys[model_inputs],
        )

    filter = kalman.build_filter(
        get_init_params, get_dynamics_params, get_observation_params
    )
    smoother = kalman.build_smoother(
        get_dynamics_params, store_gain=True, store_chol_cov_given_next=True
    )

    factorializer = factorial.gaussian.build_factorializer(
        get_factorial_indices=lambda model_inputs: factorial_indices[model_inputs - 1]
    )
    model_inputs = jnp.arange(len(ys))
    return filter, smoother, factorializer, model_inputs


seeds = [1, 43]
x_dims = [1, 3]
y_dims = [1, 2]
num_factors = [10, 20]
num_factors_local = [2]  # number of factors to interact at each time step
num_time_steps = [1, 25]

common_params = list(
    itertools.product(seeds, x_dims, y_dims, num_factors, num_time_steps)
)


def test_filter(seed, x_dim, y_dim, num_factors, num_factors_local, num_time_steps):
    model_params = generate_factorial_kalman_model(
        seed, x_dim, y_dim, num_factors, num_factors_local, num_time_steps
    )
    filter_obj, smoother_obj, factorializer, model_inputs = (
        load_kalman_pairwise_factorial_inference(*model_params)
    )

    init_state, local_filter_states = factorial.filter(
        filter_obj, factorializer, model_inputs, output_factorial=False
    )
