import itertools
from typing import cast

import chex
import jax
import jax.numpy as jnp
import pytest
from jax import Array, tree
from jax.scipy.linalg import block_diag

import cuthbert
from cuthbert import factorial
from cuthbert.factorial.utils import serial_to_single_factor
from cuthbert.gaussian import kalman
from cuthbert.inference import Filter, Smoother
from cuthbertlib.linalg import block_marginal_sqrt_cov
from cuthbertlib.types import ArrayTree
from tests.cuthbert.factorial.gaussian_utils import generate_factorial_kalman_model
from tests.cuthbertlib.kalman.test_filtering import std_predict, std_update
from tests.cuthbertlib.kalman.test_smoothing import std_kalman_smoother


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def build_pairwise_factorial_filter(
    model_params: tuple[Array, ...],
) -> tuple[Filter, factorial.Factorializer, Array]:
    """Builds factorial Kalman filter objects and model inputs for a linear-Gaussian SSM.

    model_params is a tuple of:
        m0: Array,  # (F, d).
        chol_P0: Array,  # (F, d, d).
        Fs: Array,  # (T, 2 * d, 2 * d).
        cs: Array,  # (T, 2 * d).
        chol_Qs: Array,  # (T, 2 * d, 2 * d).
        Hs: Array,  # (T, d_y, 2 * d).
        ds: Array,  # (T, d_y).
        chol_Rs: Array,  # (T, d_y, d_y).
        ys: Array,  # (T, d_y).
        factorial_indices: Array,  # (T, 2).
    """

    m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys, factorial_indices = model_params

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
    model_inputs = jnp.arange(len(ys) + 1)

    return filter, factorializer, model_inputs


def project_single_factor_smoothing_problem(
    model_params: tuple[Array, ...],
    smoother_factorial_index: int,
) -> tuple[Smoother, Array, Array, Array, Array]:
    """Projects pairwise dynamics onto a single factor smoothing problem."""

    m0, _, Fs, cs, chol_Qs, _, _, _, ys, factorial_indices = model_params

    d_x = m0.shape[1]
    projected_Fs = []
    projected_cs = []
    projected_chol_Qs = []

    for t, local_factorial_indices in enumerate(factorial_indices):
        local_chol_Qs = block_marginal_sqrt_cov(chol_Qs[t], d_x)
        for local_idx, factorial_index in enumerate(local_factorial_indices):
            if int(factorial_index) != smoother_factorial_index:
                continue

            state_slice = slice(local_idx * d_x, (local_idx + 1) * d_x)
            projected_Fs.append(Fs[t][state_slice, state_slice])
            projected_cs.append(cs[t][state_slice])
            projected_chol_Qs.append(local_chol_Qs[local_idx])

    if projected_Fs:
        projected_Fs = jnp.stack(projected_Fs)
        projected_cs = jnp.stack(projected_cs)
        projected_chol_Qs = jnp.stack(projected_chol_Qs)
    else:
        projected_Fs = jnp.zeros((0, d_x, d_x), dtype=Fs.dtype)
        projected_cs = jnp.zeros((0, d_x), dtype=cs.dtype)
        projected_chol_Qs = jnp.zeros((0, d_x, d_x), dtype=chol_Qs.dtype)

    def get_dynamics_params_single_factor(
        model_inputs: int,
    ) -> tuple[Array, Array, Array]:
        return (
            projected_Fs[model_inputs - 1],
            projected_cs[model_inputs - 1],
            projected_chol_Qs[model_inputs - 1],
        )

    smoother = kalman.build_smoother(
        get_dynamics_params_single_factor,
        store_gain=True,
        store_chol_cov_given_next=True,
    )
    smoother_model_inputs = jnp.arange(len(projected_Fs) + 1)

    return (
        smoother,
        smoother_model_inputs,
        projected_Fs,
        projected_cs,
        projected_chol_Qs,
    )


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
    filter_obj, factorializer, model_inputs = build_pairwise_factorial_filter(
        model_params
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

common_smoother_params = [
    (*params, smoother_idx)
    for params in common_params
    for smoother_idx in smoother_indices
]


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
    filter_obj, factorializer, filter_model_inputs = build_pairwise_factorial_filter(
        model_params
    )
    (
        smoother,
        smoother_model_inputs,
        projected_Fs,
        projected_cs,
        projected_chol_Qs,
    ) = project_single_factor_smoothing_problem(
        model_params,
        smoother_factorial_index=smoother_factorial_index,
    )

    init_state, local_filter_states = factorial.filter(
        filter_obj, factorializer, filter_model_inputs, output_factorial=False
    )

    factorial_inds = model_params[-1]
    local_filter_states_single_factor = serial_to_single_factor(
        factorializer.extract,
        local_filter_states,
        factorial_inds,
        smoother_factorial_index,
        init_factorial_tree=init_state,
    )

    if len(projected_Fs) == 0:
        expected_initial_state = tree.map(
            lambda x: x[None],
            factorializer.extract(init_state, smoother_factorial_index),
        )
        chex.assert_trees_all_close(
            local_filter_states_single_factor, expected_initial_state
        )
        return

    smoother_states = cuthbert.smoother(
        smoother, local_filter_states_single_factor, smoother_model_inputs
    )

    filter_covs = (
        local_filter_states_single_factor.chol_cov
        @ local_filter_states_single_factor.chol_cov.transpose(0, 2, 1)
    )
    projected_Qs = projected_chol_Qs @ projected_chol_Qs.transpose(0, 2, 1)
    (des_means, des_covs), des_cross_covs = std_kalman_smoother(
        local_filter_states_single_factor.mean,
        filter_covs,
        projected_Fs,
        projected_cs,
        projected_Qs,
    )

    smoother_covs = smoother_states.chol_cov @ smoother_states.chol_cov.transpose(
        0, 2, 1
    )
    smoother_cross_covs = smoother_states.gain[:-1] @ smoother_covs[1:]
    chex.assert_trees_all_close(
        (smoother_states.mean, smoother_covs, smoother_cross_covs),
        (des_means, des_covs, des_cross_covs),
        rtol=1e-10,
    )

    def construct_joint_cov(cov_t_plus_1, cov_t, cross_cov_t):
        return jnp.block(
            [
                [cov_t_plus_1, cross_cov_t.T],
                [cross_cov_t, cov_t],
            ]
        )

    des_joint_covs = jax.vmap(construct_joint_cov)(
        des_covs[1:],
        des_covs[:-1],
        des_cross_covs,
    )

    def construct_joint_chol_cov(chol_cov_t_plus_1, gain_t, chol_cov_given_next_t):
        return jnp.block(
            [
                [chol_cov_t_plus_1, jnp.zeros_like(chol_cov_t_plus_1)],
                [gain_t @ chol_cov_t_plus_1, chol_cov_given_next_t],
            ]
        )

    smoother_joint_chol_covs = jax.vmap(construct_joint_chol_cov)(
        smoother_states.chol_cov[1:],
        smoother_states.gain[:-1],
        smoother_states.chol_cov_given_next[:-1],
    )
    smoother_joint_covs = smoother_joint_chol_covs @ smoother_joint_chol_covs.transpose(
        0, 2, 1
    )
    chex.assert_trees_all_close(des_joint_covs, smoother_joint_covs, rtol=1e-10)
