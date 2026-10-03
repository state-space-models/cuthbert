import chex
import jax.numpy as jnp
from jax import random

from cuthbert import filter
from cuthbert.npf import build_filter
from cuthbertlib.resampling import systematic


class TestNestedParticleFilter(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    def test_filter_runs_and_returns_expected_shapes(self):
        n_param_particles = 5
        n_state_particles = 7
        param_dim = 2
        state_dim = 2
        num_time_steps = 3

        def init_param_sample(key):
            return random.normal(key, (param_dim,))

        def init_sample(key, model_inputs):
            return jnp.full((state_dim,), model_inputs) + 0.1 * random.normal(
                key, (state_dim,)
            )

        def propagate_sample(key, state, param, model_inputs):
            noise = 0.1 * random.normal(key, (state_dim,))
            return 0.8 * state + 0.2 * param + model_inputs + noise

        def log_potential(state_prev, state, param, model_inputs):
            del state_prev, param
            residual = model_inputs - state
            return -0.5 * jnp.sum(residual**2)

        def kernel_fn(key, particle):
            return particle + 0.01 * random.normal(key, particle.shape)

        filter_obj = build_filter(
            init_param_sample=init_param_sample,
            init_sample=init_sample,
            propagate_sample=propagate_sample,
            log_potential=log_potential,
            n_param_particles=n_param_particles,
            n_state_particles=n_state_particles,
            resampling_fn=systematic.resampling,
            kernel_fn=kernel_fn,
        )

        model_inputs = jnp.arange(num_time_steps + 1, dtype=jnp.float32)

        states = self.variant(filter, static_argnames=("filter_obj", "parallel"))(
            filter_obj, model_inputs, parallel=False, key=random.key(0)
        )

        expected_time_shape = (num_time_steps + 1,)
        chex.assert_shape(states.key, expected_time_shape)
        chex.assert_shape(
            states.param_particles,
            expected_time_shape + (n_param_particles, param_dim),
        )
        chex.assert_shape(
            states.state_particles,
            expected_time_shape + (n_param_particles, n_state_particles, state_dim),
        )
        chex.assert_shape(
            states.param_log_weights, expected_time_shape + (n_param_particles,)
        )
        chex.assert_shape(
            states.state_log_weights,
            expected_time_shape + (n_param_particles, n_state_particles),
        )
        chex.assert_shape(states.model_inputs, expected_time_shape)
        chex.assert_shape(states.log_normalizing_constant, expected_time_shape)
        chex.assert_tree_all_finite(
            (
                states.param_particles,
                states.state_particles,
                states.param_log_weights,
                states.state_log_weights,
                states.model_inputs,
                states.log_normalizing_constant,
            )
        )
