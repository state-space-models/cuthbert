import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

from cuthbert.mcmc.csmc.conditional_particle_filter import build_csmc_filter
from cuthbert.filtering import filter as apply_filter
from cuthbertlib.resampling.systematic import conditional_resampling


# A simple linear Gaussian state-space model for testing
def f(x, _):  # state transition
    return 0.9 * x


def g(x, _):  # observation
    return 0.5 * x


def sample_init(key, _):
    return jax.random.normal(key, (1,))


def propagate_sample(key, prev_particles, _):
    return f(prev_particles, None) + jax.random.normal(key, prev_particles.shape)


def log_potential(_, particles, model_inputs):
    y_t, _, _ = model_inputs
    return jax.scipy.stats.norm.logpdf(y_t, loc=g(particles, None), scale=1.0)


class TestConditionalParticleFilter(chex.TestCase):
    @chex.all_variants(with_pmap=False, without_jit=False)
    @parameterized.parameters(
        {"seed": 0, "n_particles": 100, "seq_len": 10, "conditional": True},
        {"seed": 42, "n_particles": 100, "seq_len": 10, "conditional": False},
    )
    def test_csmc_filter(self, seed, n_particles, seq_len, conditional):
        """Tests the conditional particle filter forward pass."""
        key = jax.random.key(seed)
        key_truth, key_obs, key_filter = jax.random.split(key, 3)

        # Generate a ground truth trajectory
        true_states = []
        x = 0.0
        keys = jax.random.split(key_truth, seq_len)
        for i in range(seq_len):
            x = f(x, None) + jax.random.normal(keys[i])
            true_states.append(x)
        true_states = jnp.array(true_states)

        # Generate observations
        observations = g(true_states, None) + jax.random.normal(
            key_obs, true_states.shape
        )

        # Define reference trajectory for the filter
        reference_particles = true_states
        reference_indices = jnp.zeros(seq_len, dtype=int)

        # The model_inputs will be a tuple of (observation, ref_particle, ref_index)
        model_inputs = (
            observations,
            reference_particles,
            reference_indices,
        )

        # Build the filter
        csmc_filter = self.variant(
            build_csmc_filter,
            static_argnums=(3, 5),
        )(
            init_sample=sample_init,
            propagate_sample=propagate_sample,
            log_potential=log_potential,
            n_particles=n_particles,
            resampling_fn=conditional_resampling,
            conditional=conditional,
        )

        # Run the filter
        filtered_states = apply_filter(csmc_filter, model_inputs, key=key_filter)

        # --- Assertions ---
        chex.assert_shape(
            filtered_states.particles, (seq_len, n_particles, 1)
        )
        chex.assert_shape(filtered_states.log_weights, (seq_len, n_particles))

        if conditional:
            # Check that the reference trajectory is correctly pinned
            pinned_particles = filtered_states.particles[
                jnp.arange(seq_len), reference_indices
            ]
            chex.assert_trees_all_close(
                pinned_particles, reference_particles, atol=1e-5
            )

        # Check that log-likelihood is not NaN or Inf
        log_likelihood = filtered_states.log_normalizing_constant[-1]
        chex.assert_scalar_not_nan(log_likelihood)
        chex.assert_scalar_not_inf(log_likelihood)
