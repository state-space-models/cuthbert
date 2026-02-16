import itertools

import chex
import jax
import jax.numpy as jnp
import pytest
from jax import Array

from cuthbert import filter, smoother
from cuthbert.gaussian import taylor
from cuthbert.gaussian.taylor.types import (
    GetDynamicsLogDensity,
    GetInitLogDensity,
    LogConditionalDensity,
    LogDensity,
    LogPotential,
)
from cuthbert.gaussian.types import (
    LinearizedKalmanFilterState,
)
from cuthbert.inference import Filter, Smoother
from cuthbertlib.kalman.generate import generate_lgssm
from cuthbertlib.stats import multivariate_normal
from tests.cuthbert.gaussian.test_kalman import std_kalman_filter
from tests.cuthbertlib.kalman.test_smoothing import std_kalman_smoother


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def _load_taylor_init_and_dynamics(
    m0: Array,
    chol_P0: Array,
    Fs: Array,
    cs: Array,
    chol_Qs: Array,
) -> tuple[GetInitLogDensity, GetDynamicsLogDensity]:
    """Builds linearized log density Kalman filter and smoother objects and model_inputs
    for a linear-Gaussian SSM.
    """

    def get_init_log_density(model_inputs: int) -> tuple[LogDensity, Array]:
        def init_log_density(x):
            return multivariate_normal.logpdf(x, m0, chol_P0)

        return init_log_density, jnp.zeros_like(m0)

    def get_dynamics_log_density(
        state: LinearizedKalmanFilterState, model_inputs: int
    ) -> tuple[LogConditionalDensity, Array, Array]:
        def dynamics_log_density(x_prev, x):
            return multivariate_normal.logpdf(
                x,
                Fs[model_inputs - 1] @ x_prev + cs[model_inputs - 1],
                chol_Qs[model_inputs - 1],
            )

        return (
            dynamics_log_density,
            jnp.zeros_like(m0),
            jnp.zeros_like(m0),
        )

    return get_init_log_density, get_dynamics_log_density


def load_taylor_inference(
    m0: Array,
    chol_P0: Array,
    Fs: Array,
    cs: Array,
    chol_Qs: Array,
    Hs: Array,
    ds: Array,
    chol_Rs: Array,
    ys: Array,
    associative_filter: bool = False,
    ignore_nan_dims: bool = False,
) -> tuple[Filter, Smoother, Array]:
    """Builds linearized log density Kalman filter and smoother objects and model_inputs
    for a linear-Gaussian SSM.
    """
    get_init_log_density, get_dynamics_log_density = _load_taylor_init_and_dynamics(
        m0, chol_P0, Fs, cs, chol_Qs
    )

    def get_observation_log_density(
        state: LinearizedKalmanFilterState, model_inputs: int
    ) -> tuple[LogConditionalDensity, Array, Array]:
        def observation_log_density(x, y):
            return multivariate_normal.logpdf(
                y,
                Hs[model_inputs - 1] @ x + ds[model_inputs - 1],
                chol_Rs[model_inputs - 1],
            )

        return (
            observation_log_density,
            jnp.zeros_like(m0),
            ys[model_inputs - 1],
        )

    filter = taylor.build_filter(
        get_init_log_density,
        get_dynamics_log_density,
        get_observation_log_density,
        associative=associative_filter,
        ignore_nan_dims=ignore_nan_dims,
    )
    smoother = taylor.build_smoother(
        get_dynamics_log_density,
        ignore_nan_dims=ignore_nan_dims,
        store_gain=True,
    )
    model_inputs = jnp.arange(len(ys) + 1)
    return filter, smoother, model_inputs


seeds = [0, 42, 99, 123, 456]
x_dims = [3]
y_dims = [1, 2]
num_time_steps = [1, 25]

common_params = list(itertools.product(seeds, x_dims, y_dims, num_time_steps))


@pytest.mark.parametrize("seed,x_dim,y_dim,num_time_steps", common_params)
def test_offline_filter(seed, x_dim, y_dim, num_time_steps):
    # Generate a linear-Gaussian state-space model.
    m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys = generate_lgssm(
        seed, x_dim, y_dim, num_time_steps
    )

    if num_time_steps > 1:
        # Set an observation to nan
        ys = ys.at[1, 0].set(jnp.nan)

    taylor_filter, _, model_inputs = load_taylor_inference(
        m0,
        chol_P0,
        Fs,
        cs,
        chol_Qs,
        Hs,
        ds,
        chol_Rs,
        ys,
        associative_filter=False,
        ignore_nan_dims=True,
    )

    # Run sequential sqrt filter
    seq_states = filter(taylor_filter, model_inputs, parallel=False)
    seq_means, seq_chol_covs, seq_ells = (
        seq_states.mean,
        seq_states.chol_cov,
        seq_states.log_normalizing_constant,
    )

    associative_taylor_filter, _, model_inputs = load_taylor_inference(
        m0,
        chol_P0,
        Fs,
        cs,
        chol_Qs,
        Hs,
        ds,
        chol_Rs,
        ys,
        associative_filter=True,
        ignore_nan_dims=True,
    )

    # Run associative filter with parallel=False§
    seq_ass_states = filter(associative_taylor_filter, model_inputs, parallel=False)
    seq_ass_means, seq_ass_chol_covs, seq_ass_ells = (
        seq_ass_states.mean,
        seq_ass_states.chol_cov,
        seq_ass_states.log_normalizing_constant,
    )

    # Run associative filter with parallel=True
    par_ass_states = filter(associative_taylor_filter, model_inputs, parallel=True)
    par_ass_means, par_ass_chol_covs, par_ass_ells = (
        par_ass_states.mean,
        par_ass_states.chol_cov,
        par_ass_states.log_normalizing_constant,
    )

    # Run the standard Kalman filter.
    P0 = chol_P0 @ chol_P0.T
    Qs = chol_Qs @ chol_Qs.transpose(0, 2, 1)
    Rs = chol_Rs @ chol_Rs.transpose(0, 2, 1)
    des_means, des_covs, des_ells = std_kalman_filter(
        m0, P0, Fs, cs, Qs, Hs, ds, Rs, ys
    )

    seq_covs = seq_chol_covs @ seq_chol_covs.transpose(0, 2, 1)
    seq_ass_covs = seq_ass_chol_covs @ seq_ass_chol_covs.transpose(0, 2, 1)
    par_ass_covs = par_ass_chol_covs @ par_ass_chol_covs.transpose(0, 2, 1)
    chex.assert_trees_all_close(
        (seq_means, seq_covs, seq_ells),
        (seq_ass_means, seq_ass_covs, seq_ass_ells),
        (par_ass_means, par_ass_covs, par_ass_ells),
        (des_means, des_covs, des_ells),
        rtol=1e-5,
        atol=1e-8,
    )


@pytest.mark.parametrize("seed", [1, 43, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [1, 10])
@pytest.mark.parametrize("y_dim", [1, 5])
@pytest.mark.parametrize("associative", [True, False])
def test_filter_noop(seed, x_dim, y_dim, associative):
    m0, chol_P0 = generate_lgssm(seed, x_dim, y_dim, 0)[:2]

    def get_init_log_density(model_inputs: int) -> tuple[LogDensity, Array]:
        def init_log_density(x):
            return multivariate_normal.logpdf(x, m0, chol_P0)

        return init_log_density, jnp.zeros_like(m0)

    def get_noop_dynamics_log_density(
        state: LinearizedKalmanFilterState, model_inputs: int
    ) -> tuple[LogConditionalDensity, Array, Array]:
        return (
            lambda x_prev, x: multivariate_normal.logpdf(x, x_prev, jnp.eye(x_dim)),
            jnp.full_like(m0, jnp.nan),
            jnp.full_like(m0, jnp.nan),
        )  # For Taylor we indicate noop dynamics by setting both linearization points to NaNs

    def get_noop_observation_log_density(
        state: LinearizedKalmanFilterState, model_inputs: int
    ) -> tuple[LogConditionalDensity, Array, Array]:
        return (
            lambda x, y: multivariate_normal.logpdf(
                y,
                x,
                jnp.zeros((y_dim, y_dim)),
            ),
            jnp.zeros_like(m0),
            jnp.full(y_dim, jnp.nan),
        )

    filter_obj = taylor.build_filter(
        get_init_log_density=get_init_log_density,
        get_dynamics_log_density=get_noop_dynamics_log_density,
        get_observation_func=get_noop_observation_log_density,
        associative=associative,
    )

    state = filter_obj.init_prepare(None)
    prep_state = filter_obj.filter_prepare(None)
    filtered_state = filter_obj.filter_combine(state, prep_state)

    chex.assert_trees_all_close(
        (state.mean, state.chol_cov @ state.chol_cov.T, state.log_normalizing_constant),
        (
            filtered_state.mean,
            filtered_state.chol_cov @ filtered_state.chol_cov.T,
            filtered_state.log_normalizing_constant,
        ),
        rtol=1e-10,
    )  # Test covs rather than chol_covs because signs can be different


@pytest.mark.parametrize("seed,x_dim,y_dim,num_time_steps", common_params)
def test_smoother(seed, x_dim, y_dim, num_time_steps):
    # Generate a linear-Gaussian state-space model.
    m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys = generate_lgssm(
        seed, x_dim, y_dim, num_time_steps
    )

    log_density_filter, log_density_smoother, model_inputs = load_taylor_inference(
        m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys
    )

    # Run the Kalman filter and the standard Kalman smoother.
    filt_states = filter(log_density_filter, model_inputs)
    filt_means, filt_chol_covs = filt_states.mean, filt_states.chol_cov
    filt_covs = filt_chol_covs @ filt_chol_covs.transpose(0, 2, 1)
    Qs = chol_Qs @ chol_Qs.transpose(0, 2, 1)
    (des_means, des_covs), des_cross_covs = std_kalman_smoother(
        filt_means, filt_covs, Fs, cs, Qs
    )

    # Run the sequential and parallel versions of the square root smoother.
    seq_smoother_states = smoother(
        log_density_smoother, filt_states, model_inputs, parallel=False
    )
    seq_means, seq_chol_covs, seq_gains = (
        seq_smoother_states.mean,
        seq_smoother_states.chol_cov,
        seq_smoother_states.gain,
    )

    par_smoother_states = smoother(
        log_density_smoother, filt_states, model_inputs, parallel=True
    )
    par_means, par_chol_covs, par_gains = (
        par_smoother_states.mean,
        par_smoother_states.chol_cov,
        par_smoother_states.gain,
    )

    seq_covs = seq_chol_covs @ seq_chol_covs.transpose(0, 2, 1)
    par_covs = par_chol_covs @ par_chol_covs.transpose(0, 2, 1)
    seq_cross_covs = seq_gains[:-1] @ seq_covs[1:]
    par_cross_covs = par_gains[:-1] @ par_covs[1:]
    chex.assert_trees_all_close(
        (seq_means, seq_covs, seq_cross_covs),
        (par_means, par_covs, par_cross_covs),
        (des_means, des_covs, des_cross_covs),
        rtol=1e-5,
        atol=1e-8,
    )

    seq_default_mi = smoother(log_density_smoother, filt_states, parallel=False)
    chex.assert_trees_all_close(seq_default_mi, seq_smoother_states, rtol=1e-10)

    par_default_mi = smoother(log_density_smoother, filt_states, parallel=True)
    chex.assert_trees_all_close(par_default_mi, par_smoother_states, rtol=1e-10)


def load_taylor_inference_potential(
    m0: Array,
    chol_P0: Array,
    Fs: Array,
    cs: Array,
    chol_Qs: Array,
    ms: Array,
    chol_Rs: Array,
    associative_filter: bool = False,
    ignore_nan_dims: bool = False,
) -> tuple[Filter, Smoother, Array]:
    """Builds linearized log density Kalman filter and smoother objects and model_inputs
    for a linear-Gaussian SSM.

    Uses Gaussian log potential for observations instead of conditional log density.
    """
    get_init_log_density, get_dynamics_log_density = _load_taylor_init_and_dynamics(
        m0, chol_P0, Fs, cs, chol_Qs
    )

    def get_observation_log_potential(
        state: LinearizedKalmanFilterState, model_inputs: int
    ) -> tuple[LogPotential, Array]:
        def observation_log_potential(x):
            return multivariate_normal.logpdf(
                x, ms[model_inputs - 1], chol_Rs[model_inputs - 1]
            )

        return (
            observation_log_potential,
            jnp.zeros_like(m0),
        )

    filter = taylor.build_filter(
        get_init_log_density,
        get_dynamics_log_density,
        get_observation_log_potential,
        associative=associative_filter,
        ignore_nan_dims=ignore_nan_dims,
    )
    smoother = taylor.build_smoother(
        get_dynamics_log_density, ignore_nan_dims=ignore_nan_dims
    )
    model_inputs = jnp.arange(len(ms) + 1)
    return filter, smoother, model_inputs


common_params_potential = list(itertools.product(seeds, x_dims, num_time_steps))


@pytest.mark.parametrize("seed,x_dim,num_time_steps", common_params_potential)
def test_offline_filter_potential(seed, x_dim, num_time_steps):
    # Generate a linear-Gaussian state-space model.
    m0, chol_P0, Fs, cs, chol_Qs, _, _, chol_Rs, ms = generate_lgssm(
        seed, x_dim, x_dim, num_time_steps
    )

    taylor_filter, _, model_inputs = load_taylor_inference_potential(
        m0, chol_P0, Fs, cs, chol_Qs, ms, chol_Rs, associative_filter=False
    )

    # Run sequential sqrt filter
    seq_states = filter(taylor_filter, model_inputs, parallel=False)
    seq_means, seq_chol_covs, seq_ells = (
        seq_states.mean,
        seq_states.chol_cov,
        seq_states.log_normalizing_constant,
    )

    associative_taylor_filter, _, model_inputs = load_taylor_inference_potential(
        m0, chol_P0, Fs, cs, chol_Qs, ms, chol_Rs, associative_filter=True
    )

    # Run associative filter with parallel=False
    seq_ass_states = filter(associative_taylor_filter, model_inputs, parallel=False)
    seq_ass_means, seq_ass_chol_covs, seq_ass_ells = (
        seq_ass_states.mean,
        seq_ass_states.chol_cov,
        seq_ass_states.log_normalizing_constant,
    )

    par_ass_states = filter(associative_taylor_filter, model_inputs, parallel=True)
    par_ass_means, par_ass_chol_covs, par_ass_ells = (
        par_ass_states.mean,
        par_ass_states.chol_cov,
        par_ass_states.log_normalizing_constant,
    )

    # Run the standard Kalman filter.
    Hs = jnp.eye(x_dim)[None].repeat(num_time_steps, axis=0)
    ds = jnp.zeros((num_time_steps, x_dim))
    P0 = chol_P0 @ chol_P0.T
    Qs = chol_Qs @ chol_Qs.transpose(0, 2, 1)
    Rs = chol_Rs @ chol_Rs.transpose(0, 2, 1)
    des_means, des_covs, des_ells = std_kalman_filter(
        m0, P0, Fs, cs, Qs, Hs, ds, Rs, ms
    )

    seq_covs = seq_chol_covs @ seq_chol_covs.transpose(0, 2, 1)
    seq_ass_covs = seq_ass_chol_covs @ seq_ass_chol_covs.transpose(0, 2, 1)
    par_ass_covs = par_ass_chol_covs @ par_ass_chol_covs.transpose(0, 2, 1)
    chex.assert_trees_all_close(
        (seq_means, seq_covs, seq_ells),
        (seq_ass_means, seq_ass_covs, seq_ass_ells),
        (par_ass_means, par_ass_covs, par_ass_ells),
        (des_means, des_covs, des_ells),
        rtol=1e-5,
        atol=1e-8,
    )
