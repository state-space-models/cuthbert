import itertools
from typing import cast

import chex
import jax
import jax.numpy as jnp
import pytest
from jax import Array, vmap
from jax.scipy.linalg import block_diag

from cuthbert import factorial
from cuthbert.gaussian import kalman
from cuthbert.inference import Filter, Smoother
from cuthbertlib.kalman.generate import generate_lgssm
from cuthbertlib.types import ArrayTree
from tests.cuthbert.factorial.gaussian_utils import generate_factorial_kalman_model
from tests.cuthbertlib.kalman.test_filtering import std_predict, std_update
from tests.cuthbertlib.kalman.test_smoothing import std_kalman_smoother


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
    itertools.product(
        seeds, x_dims, y_dims, num_factors, num_factors_local, num_time_steps
    )
)


@pytest.mark.parametrize(
    "seed,x_dim,y_dim,num_factors,num_factors_local,num_time_steps", common_params
)
def test_filter(seed, x_dim, y_dim, num_factors, num_factors_local, num_time_steps):
    model_params = generate_factorial_kalman_model(
        seed, x_dim, y_dim, num_factors, num_factors_local, num_time_steps
    )
    filter_obj, smoother_obj, factorializer, model_inputs = (
        load_kalman_pairwise_factorial_inference(*model_params)
    )

    # True means, covs and log norm constants
    fac_means = model_params[0]
    fac_chol_covs = model_params[1]
    fac_covs = fac_chol_covs @ fac_chol_covs.transpose(0, 2, 1)
    ell = jnp.array(0.0)

    local_means = []
    local_covs = []
    ells = []
    fac_means_t_all = [fac_means]
    fac_covs_t_all = [fac_covs]
    for i in model_inputs[1:]:
        F, c, chol_Q = (
            model_params[2][i - 1],
            model_params[3][i - 1],
            model_params[4][i - 1],
        )
        H, d, chol_R, y = (
            model_params[5][i],
            model_params[6][i],
            model_params[7][i],
            model_params[8][i],
        )
        fac_inds = model_params[9][i - 1]

        joint_mean = fac_means[fac_inds].reshape(-1)
        joint_cov = block_diag(*fac_covs[fac_inds])
        Q = chol_Q @ chol_Q.T
        R = chol_R @ chol_R.T
        pred_mean, pred_cov = std_predict(joint_mean, joint_cov, F, c, Q)
        upd_mean, upd_cov, upd_ell = std_update(pred_mean, pred_cov, H, d, R, y)
        marginal_means = upd_mean.reshape(len(fac_inds), -1)
        marginal_covs = jnp.array(
            [
                upd_cov[i * x_dim : (i + 1) * x_dim, i * x_dim : (i + 1) * x_dim]
                for i in range(len(fac_inds))
            ]
        )
        ell += upd_ell
        local_means.append(marginal_means)
        local_covs.append(marginal_covs)
        ells.append(ell)
        fac_means = fac_means.at[fac_inds].set(marginal_means)
        fac_covs = fac_covs.at[fac_inds].set(marginal_covs)
        fac_means_t_all.append(fac_means)
        fac_covs_t_all.append(fac_covs)

    local_means = jnp.stack(local_means)
    local_covs = jnp.stack(local_covs)
    ells = jnp.stack(ells)
    fac_means_t_all = jnp.stack(fac_means_t_all)
    fac_covs_t_all = jnp.stack(fac_covs_t_all)

    # Check output_factorial = False
    init_state, local_filter_states = factorial.filter(
        filter_obj, factorializer, model_inputs, output_factorial=False
    )
    local_filter_covs = (
        local_filter_states.chol_cov
        @ local_filter_states.chol_cov.transpose(0, 1, 3, 2)
    )
    chex.assert_trees_all_close(
        (init_state.mean, init_state.chol_cov), (model_params[0], model_params[1])
    )
    chex.assert_trees_all_close(
        (local_means, local_covs, ells),
        (
            local_filter_states.mean,
            local_filter_covs,
            local_filter_states.log_normalizing_constant,
        ),
    )

    # Check output_factorial = False
    factorial_filtering_states = factorial.filter(
        filter_obj, factorializer, model_inputs, output_factorial=True
    )

    factorial_filtering_states = cast(ArrayTree, factorial_filtering_states)
    factorial_filtering_covs = (
        factorial_filtering_states.chol_cov
        @ factorial_filtering_states.chol_cov.transpose(0, 1, 3, 2)
    )
    chex.assert_trees_all_close(
        (fac_means_t_all, fac_covs_t_all),
        (factorial_filtering_states.mean, factorial_filtering_covs),
    )
    chex.assert_trees_all_close(
        ells, factorial_filtering_states.log_normalizing_constant[1:]
    )
