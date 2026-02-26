import itertools
from typing import cast, Callable

import chex
import jax
import jax.numpy as jnp
import pytest
from jax import Array
from jax.scipy.linalg import block_diag

from cuthbert import factorial
from cuthbert.gaussian import kalman
from cuthbert.inference import Filter, Smoother
from cuthbertlib.types import ArrayTree
from cuthbertlib.linalg import block_marginal_sqrt_cov
from tests.cuthbert.factorial.gaussian_utils import generate_factorial_kalman_model
from tests.cuthbertlib.kalman.test_filtering import std_predict, std_update


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
    Hs: Array,  # (T, d_y, 2 * d)
    ds: Array,  # (T, d_y)
    chol_Rs: Array,  # (T, d_y, d_y)
    ys: Array,  # (T, d_y)
    factorial_indices: Array,  # (T, 2)
    smoother_factorial_index: int,
) -> tuple[Filter, factorial.Factorializer, Array, Smoother, Array]:
    """Builds factorial Kalman filter and smoother objects and model_inputs for a linear-Gaussian SSM."""

    def get_init_params(model_inputs: int) -> tuple[Array, Array]:
        return m0, chol_P0

    def get_dynamics_params(model_inputs: int) -> tuple[Array, Array, Array]:
        return Fs[model_inputs - 1], cs[model_inputs - 1], chol_Qs[model_inputs - 1]

    def get_observation_params(model_inputs: int) -> tuple[Array, Array, Array, Array]:
        return (
            Hs[model_inputs - 1],
            ds[model_inputs - 1],
            chol_Rs[model_inputs - 1],
            ys[model_inputs - 1],
        )

    filter = kalman.build_filter(
        get_init_params, get_dynamics_params, get_observation_params
    )

    factorializer = factorial.gaussian.build_factorializer(
        get_factorial_indices=lambda model_inputs: factorial_indices[model_inputs - 1]
    )
    filter_model_inputs = jnp.arange(len(ys) + 1)

    # Some processing to get smoothing for a single factor
    num_factors = len(m0)
    d_x = m0.shape[1]
    Fs_per_factor = [[] for _ in range(num_factors)]
    cs_per_factor = [[] for _ in range(num_factors)]
    chol_Qs_per_factor = [[] for _ in range(num_factors)]

    for i in range(1, len(ys) + 1):
        h, a = factorial_indices[i - 1]

        F_h = Fs[i - 1][:d_x, :d_x]
        F_a = Fs[i - 1][-d_x:, -d_x:]
        c_h = cs[i - 1][:d_x]
        c_a = cs[i - 1][-d_x:]
        chol_Q_h, chol_Q_a = block_marginal_sqrt_cov(chol_Qs[i - 1], d_x)
        Fs_per_factor[h].append(F_h)
        cs_per_factor[h].append(c_h)
        chol_Qs_per_factor[h].append(chol_Q_h)

        Fs_per_factor[a].append(F_a)
        cs_per_factor[a].append(c_a)
        chol_Qs_per_factor[a].append(chol_Q_a)

    def get_dynamics_params_single_factor(
        model_inputs: int,
    ) -> tuple[Array, Array, Array]:
        return (
            Fs_per_factor[smoother_factorial_index][model_inputs - 1],
            cs_per_factor[smoother_factorial_index][model_inputs - 1],
            chol_Qs_per_factor[smoother_factorial_index][model_inputs - 1],
        )

    smoother = kalman.build_smoother(
        get_dynamics_params_single_factor,
        store_gain=True,
        store_chol_cov_given_next=True,
    )
    smoother_model_inputs = jnp.arange(len(Fs_per_factor[smoother_factorial_index]) + 1)

    return filter, factorializer, filter_model_inputs, smoother, smoother_model_inputs


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
    filter_obj, factorializer, model_inputs, _, _ = (
        load_kalman_pairwise_factorial_inference(
            *model_params, smoother_factorial_index=0
        )
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
            model_params[5][i - 1],
            model_params[6][i - 1],
            model_params[7][i - 1],
            model_params[8][i - 1],
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

    # Check output_factorial = True
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


smoother_indices = [0, 1, 5]

common_smoother_params = list(itertools.product(common_params, smoother_indices))


@pytest.mark.parametrize(
    "seed,x_dim,y_dim,num_factors,num_factors_local,num_time_steps,smoother_factorial_index",
    common_smoother_params,
)
def test_smoother(
    seed,
    x_dim,
    y_dim,
    num_factors,
    num_factors_local,
    num_time_steps,
    smoother_factorial_index,
):
    model_params = generate_factorial_kalman_model(
        seed, x_dim, y_dim, num_factors, num_factors_local, num_time_steps
    )
    filter_obj, factorializer, filter_model_inputs, smoother, smoother_model_inputs = (
        load_kalman_pairwise_factorial_inference(
            *model_params, smoother_factorial_index=smoother_factorial_index
        )
    )

    # Check output_factorial = False
    init_state, local_filter_states = factorial.filter(
        filter_obj, factorializer, filter_model_inputs, output_factorial=False
    )

    # Convert to local smoother states
