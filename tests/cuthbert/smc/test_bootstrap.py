import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jax import random

from cuthbert import filter
from cuthbert.smc import bootstrap
from cuthbertlib.resampling import systematic
from cuthbertlib.stats.multivariate_normal import logpdf
from tests.cuthbert.gaussian.test_kalman import std_kalman_filter
from tests.cuthbertlib.kalman.utils import generate_lgssm


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def load_bootstrap_filter(m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys):
    def init_sample(key, model_inputs):
        return m0 + chol_P0 @ random.normal(key, m0.shape)

    def propagate_sample(key, state, model_inputs: int):
        idx = model_inputs - 1
        mean_sample = Fs[idx] @ state + cs[idx]
        return mean_sample + chol_Qs[idx] @ random.normal(key, mean_sample.shape)

    def log_potential(state_prev, state, model_inputs: int):
        idx = model_inputs - 1
        return logpdf(
            Hs[idx] @ state + ds[idx], ys[idx], chol_Rs[idx], nan_support=False
        )

    n_filter_particles = 1_000_000
    ess_threshold = 0.7
    bootstrap_filter = bootstrap.build_filter(
        init_sample,
        propagate_sample,
        log_potential,
        n_filter_particles,
        systematic.resampling,
        ess_threshold,
    )

    model_inputs = jnp.arange(len(ys) + 1)
    return bootstrap_filter, model_inputs


class BootstrapTest(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.product(
        seed=[1, 42, 99, 123, 455], x_dim=[3], y_dim=[2], num_time_steps=[20]
    )
    def test_bootstrap(self, seed, x_dim, y_dim, num_time_steps):
        m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys = generate_lgssm(
            seed, x_dim, y_dim, num_time_steps
        )

        # Run the bootstrap particle filter.
        bootstrap_filter, model_inputs = load_bootstrap_filter(
            m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys
        )
        key = random.key(seed + 1)
        bootstrap_states = self.variant(filter, static_argnames=("filter", "parallel"))(
            bootstrap_filter, model_inputs, parallel=False, key=key
        )
        weights = jax.nn.softmax(bootstrap_states.log_weights)
        bt_means = jnp.sum(bootstrap_states.particles * weights[..., None], axis=1)
        bt_covs = jax.vmap(lambda particles, w: jnp.cov(particles.T, aweights=w))(
            bootstrap_states.particles, weights
        )
        bt_ells = bootstrap_states.log_likelihood

        # Run the standard Kalman filter.
        P0 = chol_P0 @ chol_P0.T
        Qs = chol_Qs @ chol_Qs.transpose(0, 2, 1)
        Rs = chol_Rs @ chol_Rs.transpose(0, 2, 1)
        des_means, des_covs, des_ells = std_kalman_filter(
            m0, P0, Fs, cs, Qs, Hs, ds, Rs, ys
        )

        chex.assert_trees_all_close(
            (bt_ells[1:], bt_means, bt_covs),
            (des_ells, des_means, des_covs),
            atol=1e-2,
            rtol=1e-2,
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_bootstrap_pytree_particles(self):
        """Test that the bootstrap pf handles pytree states correctly."""

        def init_sample(key, model_inputs):
            keys = random.split(key, 2)
            position = random.normal(keys[0], (2,))
            velocity = random.normal(keys[1], (2,))
            return (position, velocity)

        def propagate_sample(key, state, model_inputs):
            position, velocity = state
            new_position = position + velocity * 0.1
            new_velocity = velocity + random.normal(key, (2,))
            return (new_position, new_velocity)

        def log_potential(state_prev, state, model_inputs):
            return jnp.zeros(())

        n_filter_particles = 1000
        ess_threshold = 0.7

        bootstrap_filter = bootstrap.build_filter(
            init_sample,
            propagate_sample,
            log_potential,
            n_filter_particles,
            systematic.resampling,
            ess_threshold,
        )

        key = random.key(0)
        num_time_steps = 5

        # Run the bootstrap particle filter
        model_inputs = jnp.empty(num_time_steps + 1)
        key, subkey = random.split(key)
        bootstrap_states = self.variant(filter, static_argnames=("filter", "parallel"))(
            bootstrap_filter, model_inputs, parallel=False, key=subkey
        )

        # Verify that the pytree structure is preserved
        particles = bootstrap_states.particles
        assert isinstance(particles, tuple) and len(particles) == 2
        expected_shape = (num_time_steps + 1, n_filter_particles, 2)
        chex.assert_shape(particles, expected_shape)
