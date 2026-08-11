import chex
import jax.numpy as jnp
from jax import random, vmap

from cuthbert import factorial
from cuthbert.discrete.filter import build_filter as build_discrete_filter
from cuthbert.gaussian import kalman
from cuthbert.smc.particle_filter import build_filter as build_particle_filter
from cuthbertlib.resampling import no_resampling


def test_synchronize_kalman():
    means = jnp.array([[1.0], [-2.0]])
    chol_covs = jnp.array([[[1.0]], [[2.0]]])
    transitions = jnp.array([[[2.0]], [[-1.5]]])
    offsets = jnp.array([[0.5], [1.0]])
    transition_chol_covs = jnp.array([[[3.0]], [[4.0]]])

    def predict(mean, chol_cov, transition, offset, transition_chol_cov):
        predicted_mean = transition @ mean + offset
        predicted_cov = (
            transition @ (chol_cov @ chol_cov.T) @ transition.T
            + transition_chol_cov @ transition_chol_cov.T
        )
        return predicted_mean, predicted_cov

    true_means, true_covs = vmap(predict)(
        means, chol_covs, transitions, offsets, transition_chol_covs
    )

    def get_factorial_indices(x):
        return x

    def get_init_params(_):
        return means, chol_covs

    def get_dynamics_params(factor_index):
        return (
            transitions[factor_index],
            offsets[factor_index],
            transition_chol_covs[factor_index],
        )

    def get_observation_params(_):
        # A NaN observation represents a dynamics-only update.
        return jnp.zeros((1, 1)), jnp.zeros(1), jnp.eye(1), jnp.array([jnp.nan])

    filter_obj = kalman.build_filter(
        get_init_params, get_dynamics_params, get_observation_params
    )
    factorializer = factorial.gaussian.build_factorializer(get_factorial_indices)
    factorial_state = filter_obj.init_prepare(None)

    synchronized = factorial.synchronize(
        filter_obj, factorializer, jnp.arange(len(means)), factorial_state
    )

    covs = synchronized.chol_cov @ synchronized.chol_cov.transpose(0, 2, 1)
    chex.assert_trees_all_close(
        (synchronized.mean, covs, synchronized.log_normalizing_constant),
        (true_means, true_covs, 0.0),
    )
    chex.assert_shape(synchronized.log_normalizing_constant, ())


def test_synchronize_discrete():
    init_dist = jnp.array([[0.6, 0.4], [0.25, 0.75]])
    transitions = jnp.array([[[0.7, 0.3], [0.2, 0.8]], [[0.4, 0.6], [0.9, 0.1]]])
    initial_model_inputs = jnp.array(0)
    model_inputs = jnp.arange(len(init_dist))
    true_dists = vmap(lambda dist, transition: dist @ transition)(
        init_dist, transitions
    )

    filter_obj = build_discrete_filter(
        lambda _: init_dist,
        lambda factor_index: transitions[factor_index],
        lambda _: jnp.zeros(2),
    )

    def get_factorial_indices(x):
        return x

    factorializer = factorial.discrete.build_factorializer(get_factorial_indices)

    synchronized = factorial.synchronize(
        filter_obj,
        factorializer,
        model_inputs,
        filter_obj.init_prepare(initial_model_inputs),
    )

    chex.assert_trees_all_close(
        (synchronized.dist, synchronized.log_normalizing_constant),
        (true_dists, jnp.zeros(len(init_dist))),
    )


def test_synchronize_particle_filter():
    initial_model_inputs = jnp.array([1.0, 10.0])
    model_inputs = jnp.array([0.5, -1.0])

    filter_obj = build_particle_filter(
        init_sample=lambda key, initial_value: initial_value,
        propagate_sample=lambda key, particle, increment: particle + increment,
        log_potential=lambda previous, particle, increment: jnp.array(0.0),
        n_filter_particles=3,
        resampling_fn=no_resampling.resampling,
    )

    def get_factorial_indices(x):
        return x

    factorializer = factorial.smc.build_factorializer(
        get_factorial_indices, no_resampling.resampling
    )
    initial_state = factorializer.factorialize_init_state(
        filter_obj.init_prepare(initial_model_inputs, key=random.key(0)),
        initial_model_inputs,
    )
    true_particles = vmap(lambda particles, increment: particles + increment)(
        initial_state.particles, model_inputs
    )

    synchronized = factorial.synchronize(
        filter_obj,
        factorializer,
        model_inputs,
        initial_state,
        key=random.key(1),
    )

    chex.assert_trees_all_close(
        (
            synchronized.particles,
            synchronized.log_weights,
            synchronized.log_normalizing_constant,
        ),
        (true_particles, jnp.zeros_like(initial_state.log_weights), 0.0),
    )
    chex.assert_shape(synchronized.log_normalizing_constant, ())
