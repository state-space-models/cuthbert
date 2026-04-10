from functools import partial

import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jax import random

from cuthbert import filter
from cuthbert.enkf import ensemble_kalman_filter
from cuthbertlib.kalman.generate import generate_lgssm
from cuthbertlib.linalg import tria
from tests.cuthbert.gaussian.test_kalman import std_kalman_filter


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def load_enkf_inference(m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys, noop=False):
    n_particles = 100_000
    x_dim = m0.shape[0]

    def init_sample(key):
        return m0 + chol_P0 @ random.normal(key, m0.shape)

    if noop:
        y_dim = ys.shape[1] if ys.ndim > 1 else 1

        def dynamics_fn(x, key):
            return x

        def get_dynamics(model_inputs):
            return dynamics_fn

        def observation_fn(x):
            return jnp.zeros(y_dim)

        def get_observations(model_inputs):
            return observation_fn, jnp.zeros((y_dim, y_dim)), jnp.full(y_dim, jnp.nan)

    else:

        def get_dynamics(model_inputs):
            idx = model_inputs - 1
            return (
                lambda x, key: Fs[idx] @ x
                + cs[idx]
                + chol_Qs[idx] @ random.normal(key, (x_dim,))
            )

        def get_observations(model_inputs):
            idx = model_inputs - 1
            return lambda x: Hs[idx] @ x + ds[idx], chol_Rs[idx], ys[idx]

    inference = ensemble_kalman_filter.build_filter(
        init_sample=init_sample,
        get_dynamics=get_dynamics,
        get_observations=get_observations,
        n_particles=n_particles,
    )

    model_inputs = jnp.arange(len(ys) + 1)
    return inference, model_inputs


class Test(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.product(
        seed=[0, 41, 99, 123, 456],
        x_dim=[3],
        y_dim=[2],
        num_time_steps=[20],
    )
    def test(self, seed, x_dim, y_dim, num_time_steps):
        m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys = generate_lgssm(
            seed, x_dim, y_dim, num_time_steps
        )

        # Run the EnKF.
        inference, model_inputs = load_enkf_inference(
            m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys
        )
        key = random.key(seed + 1)
        states = self.variant(filter, static_argnames=("filter_obj", "parallel"))(
            inference, model_inputs, parallel=False, key=key
        )
        means = states.mean
        chol_covs = states.chol_cov
        covs = chol_covs @ chol_covs.transpose(0, 2, 1)
        ells = states.log_normalizing_constant

        # Run the standard Kalman filter.
        P0 = chol_P0 @ chol_P0.T
        Qs = chol_Qs @ chol_Qs.transpose(0, 2, 1)
        Rs = chol_Rs @ chol_Rs.transpose(0, 2, 1)
        des_means, des_covs, des_ells = std_kalman_filter(
            m0, P0, Fs, cs, Qs, Hs, ds, Rs, ys
        )

        chex.assert_trees_all_close(
            (ells, means, covs),
            (des_ells, des_means, des_covs),
            rtol=1e-2,
            atol=1e-2,
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_nonlinear_dynamics(self):
        seed = 42
        x_dim = 3
        y_dim = 2
        num_time_steps = 5

        m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys = generate_lgssm(
            seed, x_dim, y_dim, num_time_steps
        )

        def init_sample(key):
            return m0 + chol_P0 @ random.normal(key, m0.shape)

        def dynamics_fn(x, key):
            return jnp.tanh(x)

        def get_dynamics(model_inputs):
            return dynamics_fn

        def get_observations(model_inputs):
            idx = model_inputs - 1
            return lambda x: Hs[idx] @ x + ds[idx], chol_Rs[idx], ys[idx]

        inference = ensemble_kalman_filter.build_filter(
            init_sample=init_sample,
            get_dynamics=get_dynamics,
            get_observations=get_observations,
            n_particles=1_000,
        )

        model_inputs = jnp.arange(num_time_steps + 1)
        key = random.key(seed + 1)

        states = self.variant(filter, static_argnames=("filter_obj", "parallel"))(
            inference, model_inputs, parallel=False, key=key
        )

        # Check shapes
        chex.assert_shape(states.mean, (num_time_steps + 1, x_dim))
        chex.assert_shape(states.chol_cov, (num_time_steps + 1, x_dim, x_dim))
        assert jnp.all(jnp.isfinite(states.log_normalizing_constant))

        # Check autodiff works (differentiate w.r.t. a parameter)
        def log_nc(m0_):
            def init_sample_(key):
                return m0_ + chol_P0 @ random.normal(key, m0_.shape)

            inference_ = ensemble_kalman_filter.build_filter(
                init_sample=init_sample_,
                get_dynamics=get_dynamics,
                get_observations=get_observations,
                n_particles=1_000,
            )
            states = filter(inference_, model_inputs, parallel=False, key=key)
            return states.log_normalizing_constant[-1]

        grad_val = jax.grad(log_nc)(m0)
        assert jnp.all(jnp.isfinite(grad_val))


@pytest.mark.parametrize("seed", [1, 43, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [1, 10])
@pytest.mark.parametrize("y_dim", [1, 5])
def test_filter_noop(seed, x_dim, y_dim):
    lgssm = generate_lgssm(seed, x_dim, y_dim, 0)

    inference, _ = load_enkf_inference(*lgssm, noop=True)

    init_state = inference.init_prepare(jnp.array(0), key=random.key(seed + 1))
    prep_state = inference.filter_prepare(jnp.array(1), key=random.key(seed + 2))
    filtered_state = inference.filter_combine(init_state, prep_state)

    filtered_cov = filtered_state.chol_cov @ filtered_state.chol_cov.T
    init_cov = init_state.chol_cov @ init_state.chol_cov.T

    # With identity dynamics, zero noise, and NaN observations,
    # the ensemble, covariance, and log-likelihood should be exactly preserved
    chex.assert_trees_all_close(
        (
            filtered_state.mean,
            filtered_cov,
            filtered_state.log_normalizing_constant,
        ),
        (
            init_state.mean,
            init_cov,
            init_state.log_normalizing_constant,
        ),
        rtol=1e-10,
        atol=1e-10,
    )


def test_build_filter_requires_at_least_two_particles():
    """EnKF should fail fast when configured with fewer than two particles."""

    def init_sample(key):
        return jnp.zeros(1) + jnp.eye(1) @ random.normal(key, (1,))

    with pytest.raises(ValueError, match="at least 2"):
        ensemble_kalman_filter.build_filter(
            init_sample=init_sample,
            get_dynamics=lambda _: (lambda x, key: x),
            get_observations=lambda _: (lambda x: x, jnp.eye(1), jnp.zeros(1)),
            n_particles=1,
        )
