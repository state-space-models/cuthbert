import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

from cuthbert.mcmc.csmc.conditional_particle_filter import build_csmc_filter
from cuthbert.mcmc.csmc.smoother import build_csmc_smoother
from cuthbert.filtering import filter as apply_filter
from cuthbertlib.mcmc import barker_move
from cuthbertlib.resampling.systematic import conditional_resampling

# Reuse the model from the filter test
from tests.cuthbert.mcmc.csmc.test_conditional_particle_filter import (
    f,
    g,
    log_potential,
    propagate_sample,
    sample_init,
)


class TestConditionalParticleSmoother(chex.TestCase):
    @chex.all_variants(with_pmap=False, without_jit=False)
    @parameterized.parameters(
        {"seed": 0, "n_particles": 50, "seq_len": 5, "do_sampling": True},
        {"seed": 42, "n_particles": 50, "seq_len": 5, "do_sampling": False},
    )
    def test_csmc_smoother(self, seed, n_particles, seq_len, do_sampling):
        """Tests the conditional particle smoother backward pass."""
        key = jax.random.key(seed)
        key_truth, key_obs, key_filter, key_smooth = jax.random.split(key, 4)

        # --- Forward Pass ---
        true_states = jnp.zeros((seq_len, 1))  # Dummy states
        observations = g(true_states, None) + jax.random.normal(
            key_obs, true_states.shape
        )
        reference_particles = true_states
        reference_indices = jnp.zeros(seq_len, dtype=int)
        model_inputs = (observations, reference_particles, reference_indices)

        csmc_filter = build_csmc_filter(
            init_sample=sample_init,
            propagate_sample=propagate_sample,
            log_potential=log_potential,
            n_particles=n_particles,
            resampling_fn=conditional_resampling,
            conditional=True,
        )
        forward_states = apply_filter(csmc_filter, model_inputs, key=key_filter)

        # --- Backward Pass ---
        csmc_smoother = self.variant(build_csmc_smoother)(
            log_potential_fn=log_potential,
            ancestor_move_fn=barker_move,
            conditional=True,
        )

        # Run the smoother
        smoothed_particles, smoothed_indices = csmc_smoother(
            forward_states, key_smooth, do_sampling
        )

        # --- Assertions ---
        chex.assert_shape(smoothed_particles, (seq_len, 1))
        chex.assert_shape(smoothed_indices, (seq_len,))

        if not do_sampling:
            # For backward trace, verify the trajectory is consistent
            # Reconstruct the trajectory manually and compare
            reconstructed_particles = []
            current_idx = smoothed_indices[-1]
            reconstructed_particles.append(forward_states.particles[-1, current_idx])
            for t in reversed(range(seq_len - 1)):
                current_idx = forward_states.ancestor_indices[t, current_idx]
                reconstructed_particles.append(forward_states.particles[t, current_idx])

            reconstructed_particles = jnp.array(reconstructed_particles)[::-1]
            chex.assert_trees_all_close(
                smoothed_particles, reconstructed_particles, atol=1e-5
            )
