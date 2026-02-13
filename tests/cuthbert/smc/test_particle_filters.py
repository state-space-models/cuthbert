from functools import partial

import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jax import random

from cuthbert import filter
from cuthbert.inference import Filter
from cuthbert.smc import marginal_particle_filter, particle_filter
from cuthbertlib.kalman.generate import generate_lgssm
from cuthbertlib.resampling import systematic
from cuthbertlib.stats.multivariate_normal import logpdf
from tests.cuthbert.gaussian.test_kalman import std_kalman_filter


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def load_inference(
    m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys, method, noop=False
):
    # Maybe make this more flexible in the future if we want to support other methods.
    if method == "bootstrap":
        n_filter_particles = 1_000_000
        algo = particle_filter
    elif method == "marginal":
        n_filter_particles = 3_000

        algo = marginal_particle_filter
    else:
        raise ValueError(f"Unknown method: {method}")

    def init_sample(key, model_inputs):
        return m0 + chol_P0 @ random.normal(key, m0.shape)

    if noop:

        def propagate_sample(key, state, model_inputs: int):
            return state

        def log_potential(state_prev, state, model_inputs: int):
            return jnp.zeros(())

    else:

        def propagate_sample(key, state, model_inputs: int):
            idx = model_inputs - 1
            mean_sample = Fs[idx] @ state + cs[idx]
            return mean_sample + chol_Qs[idx] @ random.normal(key, mean_sample.shape)

        def log_potential(state_prev, state, model_inputs: int):
            idx = model_inputs
            return logpdf(
                Hs[idx] @ state + ds[idx], ys[idx], chol_Rs[idx], nan_support=False
            )

    ess_threshold = 0.7
    inference = Filter(
        init_prepare=partial(
            algo.init_prepare,
            init_sample=init_sample,
            log_potential=log_potential,
            n_filter_particles=n_filter_particles,
        ),
        filter_prepare=partial(
            algo.filter_prepare,
            init_sample=init_sample,
            n_filter_particles=n_filter_particles,
        ),
        filter_combine=partial(
            algo.filter_combine,
            propagate_sample=propagate_sample,
            log_potential=log_potential,
            resampling_fn=systematic.resampling,
            ess_threshold=ess_threshold,
        ),
        associative=False,
    )

    model_inputs = jnp.arange(len(ys))
    return inference, model_inputs


class Test(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.product(
        seed=[0, 42, 99, 123, 455],
        x_dim=[3],
        y_dim=[2],
        num_time_steps=[20],
        method=["bootstrap", "marginal"],
    )
    def test(self, seed, x_dim, y_dim, num_time_steps, method):
        m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys = generate_lgssm(
            seed, x_dim, y_dim, num_time_steps
        )

        # Run the particle filter.
        inference, model_inputs = load_inference(
            m0, chol_P0, Fs, cs, chol_Qs, Hs, ds, chol_Rs, ys, method
        )
        key = random.key(seed + 1)
        states = self.variant(filter, static_argnames=("filter_obj", "parallel"))(
            inference, model_inputs, parallel=False, key=key
        )
        weights = jax.nn.softmax(states.log_weights)
        means = jnp.sum(states.particles * weights[..., None], axis=1)
        covs = jax.vmap(lambda particles, w: jnp.cov(particles.T, aweights=w))(
            states.particles, weights
        )
        ells = states.log_normalizing_constant

        # Run the standard Kalman filter.
        P0 = chol_P0 @ chol_P0.T
        Qs = chol_Qs @ chol_Qs.transpose(0, 2, 1)
        Rs = chol_Rs @ chol_Rs.transpose(0, 2, 1)
        des_means, des_covs, des_ells = std_kalman_filter(
            m0, P0, Fs, cs, Qs, Hs, ds, Rs, ys
        )
        if method == "marginal":
            chex.assert_trees_all_close(
                (ells, means), (des_ells, des_means), atol=4e-1, rtol=0.25
            )
            chex.assert_trees_all_close(covs, des_covs, atol=6e-1, rtol=0.25)

        else:
            chex.assert_trees_all_close(
                (ells, means, covs),
                (des_ells, des_means, des_covs),
                rtol=2e-2,
                atol=2e-2,
            )

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.product(method=["bootstrap", "marginal"])
    def test_pytree_particles(self, method):
        """Test that the pf handles pytree states correctly."""

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

        ess_threshold = 0.7

        if method == "bootstrap":
            n_filter_particles = 1_000
            algo = particle_filter
        elif method == "marginal":
            n_filter_particles = 100
            algo = marginal_particle_filter
        else:
            raise ValueError(f"Unknown method: {method}")

        inference = Filter(
            init_prepare=partial(
                algo.init_prepare,
                init_sample=init_sample,
                log_potential=log_potential,
                n_filter_particles=n_filter_particles,
            ),
            filter_prepare=partial(
                algo.filter_prepare,
                init_sample=init_sample,
                n_filter_particles=n_filter_particles,
            ),
            filter_combine=partial(
                algo.filter_combine,
                propagate_sample=propagate_sample,
                log_potential=log_potential,
                resampling_fn=systematic.resampling,
                ess_threshold=ess_threshold,
            ),
            associative=False,
        )

        key = random.key(0)
        num_time_steps = 5

        # Run the particle filter
        model_inputs = jnp.empty(num_time_steps + 1)
        key, subkey = random.split(key)
        states = self.variant(filter, static_argnames=("filter_obj", "parallel"))(
            inference, model_inputs, parallel=False, key=subkey
        )

        # Verify that the pytree structure is preserved
        particles = states.particles
        assert isinstance(particles, tuple) and len(particles) == 2
        expected_shape = (num_time_steps + 1, n_filter_particles, 2)
        chex.assert_shape(particles, expected_shape)


@pytest.mark.parametrize("seed", [1, 43, 99, 123, 456])
@pytest.mark.parametrize("x_dim", [1, 10])
@pytest.mark.parametrize("y_dim", [1, 5])
@pytest.mark.parametrize("method", ["bootstrap", "marginal"])
def test_filter_noop(seed, x_dim, y_dim, method):
    lgssm = generate_lgssm(seed, x_dim, y_dim, 0)

    inference, _ = load_inference(*lgssm, method=method, noop=True)

    init_state = inference.init_prepare(None, key=random.key(seed + 1))
    prep_state = inference.filter_prepare(None, key=random.key(seed + 2))
    filtered_state = inference.filter_combine(init_state, prep_state)

    chex.assert_trees_all_close(
        filtered_state._replace(key=None),
        init_state._replace(key=None),
        rtol=1e-10,
        atol=1e-10,
    )
