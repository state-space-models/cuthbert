import chex
import jax
import jax.numpy as jnp
import pytest
from jax import random

from cuthbert import filter
from cuthbert.gaussian import kalman
from cuthbert.smc import nested_particle_filter, particle_filter
from cuthbertlib.resampling import systematic
from cuthbertlib.smc.ess import log_ess

# Scalar AR(1) model with unknown coefficient a:
#   x_t = a x_{t-1} + N(0, sigma^2),  y_t = x_t + N(0, sigma^2)
SIGMA = 0.3
NUM_TIME_STEPS = 30


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def simulate(key, a=0.8):
    def step(x, k):
        k1, k2 = random.split(k)
        x = a * x + SIGMA * random.normal(k1)
        return x, x + SIGMA * random.normal(k2)

    _, ys = jax.lax.scan(step, 0.0, random.split(key, NUM_TIME_STEPS))
    return ys


def build_inner_kalman(a=None):
    """Inner Kalman filter, `a` is read from model inputs unless given."""

    def get_dynamics_params(model_inputs):
        coef = model_inputs[0] if a is None else a
        return coef * jnp.eye(1), jnp.zeros(1), SIGMA * jnp.eye(1)

    def get_observation_params(model_inputs):
        y = model_inputs[1] if a is None else model_inputs
        return jnp.eye(1), jnp.zeros(1), SIGMA * jnp.eye(1), jnp.atleast_1d(y)

    return kalman.build_filter(
        m0=jnp.zeros(1),
        chol_P0=SIGMA * jnp.eye(1),
        get_dynamics_params=get_dynamics_params,
        get_observation_params=get_observation_params,
    )


def build_inner_pf(n_filter_particles):
    def init_sample(key):
        return SIGMA * random.normal(key)

    def propagate_sample(key, state, model_inputs):
        a, _ = model_inputs
        return a * state + SIGMA * random.normal(key)

    def log_potential(state_prev, state, model_inputs):
        _, y = model_inputs
        return jax.scipy.stats.norm.logpdf(y, state, SIGMA)

    return particle_filter.build_filter(
        init_sample,
        propagate_sample,
        log_potential,
        n_filter_particles,
        systematic.resampling,
    )


def init_param_sample(key):
    return random.uniform(key, minval=-1.0, maxval=1.0)


def jitter(key, param, model_inputs):
    del model_inputs
    return param + 0.01 * random.normal(key)


def run(filter_obj, ys, seed=0):
    init_state = filter_obj.init_prepare(key=random.key(seed))
    return jax.jit(filter, static_argnames=("filter_obj", "parallel"))(
        filter_obj, ys, init_state, key=random.key(seed + 1)
    )


def test_particle_filter_inner_with_resampling():
    """Nested PF with a particle filter inner, adaptive resampling and a pytree param.

    Checks output shapes, and that parameters are jittered exactly at the steps
    where the outer particles were resampled.
    """
    n_outer, n_inner, ess_threshold = 50, 10, 0.5

    def init_param(key):
        k1, k2 = random.split(key)
        return {
            "a": init_param_sample(k1),
            "log_sigma": jnp.log(SIGMA) + 0.1 * random.normal(k2),
        }

    def jitter_param(key, param, model_inputs):
        k1, k2 = random.split(key)
        return {
            "a": jitter(k1, param["a"], model_inputs),
            "log_sigma": jitter(k2, param["log_sigma"], model_inputs),
        }

    def init_sample(key):
        return SIGMA * random.normal(key)

    def propagate_sample(key, state, model_inputs):
        param, _ = model_inputs
        return param["a"] * state + SIGMA * random.normal(key)

    def log_potential(state_prev, state, model_inputs):
        param, y = model_inputs
        return jax.scipy.stats.norm.logpdf(y, state, jnp.exp(param["log_sigma"]))

    inner_filter = particle_filter.build_filter(
        init_sample, propagate_sample, log_potential, n_inner, systematic.resampling
    )
    filter_obj = nested_particle_filter.build_filter(
        inner_filter=inner_filter,
        init_param_sample=init_param,
        propagate_param_sample=jitter_param,
        n_filter_particles=n_outer,
        resampling_fn=systematic.resampling,
        ess_threshold=ess_threshold,
    )
    states = run(filter_obj, simulate(random.key(0)))
    params = states.particles.param
    inner_state = states.particles.inner_state

    # Shapes
    T = NUM_TIME_STEPS + 1
    for leaf in jax.tree.leaves(params):
        chex.assert_shape(leaf, (T, n_outer))
    chex.assert_shape(inner_state.particles, (T, n_outer, n_inner))
    chex.assert_shape(inner_state.log_weights, (T, n_outer, n_inner))
    chex.assert_shape(states.particles.resampled, (T, n_outer))
    chex.assert_shape(states.log_weights, (T, n_outer))
    chex.assert_shape(states.log_normalizing_constant, (T,))
    assert inner_state.model_inputs is None
    chex.assert_tree_all_finite(
        (
            params,
            inner_state.particles,
            inner_state.log_weights,
            states.log_weights,
            states.log_normalizing_constant,
        )
    )

    # Parameters move exactly at the steps where the outer particles were resampled
    resampled = jax.vmap(log_ess)(states.log_weights[:-1]) < jnp.log(
        ess_threshold * n_outer
    )
    assert 0 < resampled.sum() < NUM_TIME_STEPS  # both cases must occur
    for leaf in jax.tree.leaves(params):
        ancestors = jnp.take_along_axis(leaf[:-1], states.ancestor_indices[1:], 1)
        assert jnp.all(jnp.all(leaf[1:] != ancestors, axis=1) == resampled)
        assert jnp.all(jnp.all(leaf[1:] == ancestors, axis=1) == ~resampled)


def test_matches_exact_likelihood_without_resampling():
    """Without resampling or jittering, outer weights are exact Kalman likelihoods."""
    n_outer = 50
    ys = simulate(random.key(0))
    filter_obj = nested_particle_filter.build_filter(
        inner_filter=build_inner_kalman(),
        init_param_sample=init_param_sample,
        propagate_param_sample=jitter,
        n_filter_particles=n_outer,
        resampling_fn=systematic.resampling,
        ess_threshold=0.0,  # never resample
    )
    states = run(filter_obj, ys)
    params = states.particles.param

    # Outer states are never moved since resampling never happens
    chex.assert_trees_all_close(params, jnp.broadcast_to(params[0], params.shape))
    assert not jnp.any(states.particles.resampled)

    def exact_log_likelihood(a):
        kf = build_inner_kalman(a)
        return filter(kf, ys, kf.init_prepare()).log_normalizing_constant

    expected_ll = jax.vmap(exact_log_likelihood)(params[0])  # (n_outer, T + 1)
    chex.assert_trees_all_close(states.log_weights, expected_ll.T, rtol=1e-8)
    chex.assert_trees_all_close(
        states.log_normalizing_constant,
        jax.nn.logsumexp(expected_ll, axis=0) - jnp.log(n_outer),
        rtol=1e-8,
    )


def test_particle_filter_inner_likelihood_unbiased():
    """Without resampling, outer weights are unbiased inner PF likelihood estimates.

    Each outer particle runs an independent inner particle filter with a fixed
    parameter a_i, so exp(log_weight_i - exact_log_likelihood(a_i)) has
    expectation 1. We check the sample mean over outer particles is within a few
    standard errors of 1 at every time step.
    """
    n_outer, n_inner = 200, 50
    ys = simulate(random.key(0))
    filter_obj = nested_particle_filter.build_filter(
        inner_filter=build_inner_pf(n_inner),
        init_param_sample=init_param_sample,
        propagate_param_sample=jitter,
        n_filter_particles=n_outer,
        resampling_fn=systematic.resampling,
        ess_threshold=0.0,  # never resample
    )
    states = run(filter_obj, ys)
    params = states.particles.param
    chex.assert_trees_all_close(params, jnp.broadcast_to(params[0], params.shape))

    def exact_log_likelihood(a):
        kf = build_inner_kalman(a)
        return filter(kf, ys, kf.init_prepare()).log_normalizing_constant

    expected_ll = jax.vmap(exact_log_likelihood)(params[0]).T  # (T + 1, n_outer)
    ratios = jnp.exp(states.log_weights - expected_ll)

    # At t = 0 all weights are exactly zero
    chex.assert_trees_all_close(states.log_weights[0], jnp.zeros(n_outer))
    mean = ratios[1:].mean(axis=1)
    std_err = ratios[1:].std(axis=1) / jnp.sqrt(n_outer)
    assert jnp.all(jnp.abs(mean - 1.0) < 4.0 * std_err)


def test_posterior_close_to_exact():
    ys = simulate(random.key(0))
    filter_obj = nested_particle_filter.build_filter(
        inner_filter=build_inner_kalman(),
        init_param_sample=init_param_sample,
        propagate_param_sample=jitter,
        n_filter_particles=1_000,
        resampling_fn=systematic.resampling,
        ess_threshold=0.5,
    )
    states = run(filter_obj, ys)
    weights = jax.nn.softmax(states.log_weights[-1])
    npf_mean = jnp.sum(weights * states.particles.param[-1])

    # Exact posterior under the uniform prior on a grid
    def exact_log_likelihood(a):
        kf = build_inner_kalman(a)
        return filter(kf, ys, kf.init_prepare()).log_normalizing_constant[-1]

    grid = jnp.linspace(-1.0, 1.0, 401)
    lls = jax.vmap(exact_log_likelihood)(grid)
    exact_mean = jnp.sum(jax.nn.softmax(lls) * grid)
    exact_sd = jnp.sqrt(jnp.sum(jax.nn.softmax(lls) * (grid - exact_mean) ** 2))
    exact_log_evidence = jax.nn.logsumexp(lls) - jnp.log(grid.shape[0])

    assert jnp.abs(npf_mean - exact_mean) < 0.3 * exact_sd
    assert jnp.abs(states.log_normalizing_constant[-1] - exact_log_evidence) < 0.5
