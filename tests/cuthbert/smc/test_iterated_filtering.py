"""IF2 reference calculations, parameter trees and failure handling."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cuthbert.smc import iterated_filtering
from cuthbertlib.resampling import ess_decorator, multinomial, systematic

Y = [0.3, -0.5, 0.7, 1.0, 0.2]


@pytest.fixture(autouse=True)
def precision():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def model(theta):
    phi, log_q, initial_mean = theta

    def init(k):
        return initial_mean + jax.random.normal(k, (), dtype=theta.dtype)

    def propagate(k, x, y):
        return jnp.tanh(phi) * x + jnp.exp(log_q) * jax.random.normal(
            k, (), dtype=theta.dtype
        )

    def potential(old, x, y):
        return -0.5 * ((y - x) / 0.7) ** 2 - jnp.log(0.7 * jnp.sqrt(2 * jnp.pi))

    return init, propagate, potential


def manual_if2(theta, ys, key, n, iterations, scale, init_scale, cooling, resample):
    """Plain Python-loop IF2 reference including terminal resampling."""
    swarm = jnp.broadcast_to(theta, (n, theta.size))
    likelihoods, means = [], [theta]
    for m, mk in enumerate(jax.random.split(key, iterations)):
        nk, ik, fk, tk = jax.random.split(mk, 4)
        rate = cooling**m
        swarm = swarm + rate * init_scale * jax.random.normal(
            jax.random.split(nk, 1)[0], swarm.shape
        )
        x = jax.vmap(lambda p, k: model(p)[0](k))(swarm, jax.random.split(ik, n))
        weights, ll = jnp.zeros(n), jnp.array(0.0)
        step_keys = jnp.concatenate([ik[None], jax.random.split(fk, len(ys))[:-1]])
        for y, sk in zip(ys, step_keys, strict=True):
            keys = jax.random.split(sk, n + 1)
            idx, _, _ = resample(keys[0], weights, x, n)
            x, swarm = x[idx], swarm[idx]
            split = jax.vmap(jax.random.split)(keys[1:])
            noise_keys = jax.vmap(lambda k: jax.random.split(k, 1)[0])(split[:, 0])
            eps = jax.vmap(lambda k: jax.random.normal(k, theta.shape))(noise_keys)
            swarm = swarm + rate * scale * eps
            old = x
            x = jax.vmap(lambda k, s, p: model(p)[1](k, s, y))(split[:, 1], x, swarm)
            weights = jax.vmap(lambda a, b, p: model(p)[2](a, b, y))(old, x, swarm)
            ll += jax.scipy.special.logsumexp(weights) - jnp.log(n)
        idx, _, _ = resample(tk, weights, x, n)
        swarm = swarm[idx]
        means.append(jnp.mean(swarm, axis=0))
        likelihoods.append(ll)
    return swarm, jnp.stack(means), jnp.stack(likelihoods)


@pytest.mark.parametrize("scale", [jnp.zeros(3), jnp.array([0.1, 0.04, 0.0])])
@pytest.mark.parametrize("initial_scale", [[0.0, 0.0, 0.0], [0.02, 0.0, 0.3]])
@pytest.mark.parametrize(
    "resampling_fn", [systematic.resampling, multinomial.resampling]
)
def test_if2_cooling_and_initial_parameters_against_reference(
    scale, initial_scale, resampling_fn
):
    theta, ys, key = jnp.array([0.4, -0.7, 0.2]), jnp.array(Y), jax.random.key(5)
    init_scale = jnp.array(initial_scale)
    expected, means, logliks = manual_if2(
        theta, ys, key, 16, 3, scale, init_scale, 0.6, resampling_fn
    )
    result = iterated_filtering.fit(
        model,
        theta,
        ys,
        key=key,
        n_particles=16,
        n_iterations=3,
        perturbation_scale=scale,
        initial_perturbation_scale=init_scale,
        cooling_rate=0.6,
        resampling_fn=resampling_fn,
    )
    assert result.success and result.completed_iterations == 3
    np.testing.assert_allclose(result.parameter_swarm, expected, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(result.parameter_history, means, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(
        result.perturbed_log_likelihoods, logliks, rtol=2e-6, atol=1e-7
    )


def tree_model(parameters):
    return model(jnp.concatenate([parameters["dynamic"], parameters["initial"][None]]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cooling_rate": 0},
        {"perturbation_scale": -1},
        {"initial_perturbation_scale": [1.0, 2.0]},
        {"n_iterations": -1},
        {"resampling_fn": ess_decorator(systematic.resampling, 0.7)},
    ],
)
def test_invalid_if2_configuration(kwargs):
    options = dict(
        key=jax.random.key(0), n_particles=8, n_iterations=2, perturbation_scale=0.1
    )
    options.update(kwargs)
    with pytest.raises(ValueError):
        iterated_filtering.fit(
            model, jnp.array([0.4, -0.7, 0.2]), jnp.array(Y), **options
        )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize(
    "resampling_fn", [systematic.resampling, multinomial.resampling]
)
def test_parameter_tree_jit_and_vmap(dtype, resampling_fn):
    theta = {"dynamic": jnp.array([0.4, -0.7], dtype), "initial": jnp.array(0.2, dtype)}

    def fit(key):
        return iterated_filtering.fit(
            tree_model,
            theta,
            jnp.array(Y, dtype),
            key=key,
            n_particles=16,
            n_iterations=2,
            perturbation_scale={"dynamic": 0.01, "initial": 0.0},
            initial_perturbation_scale={"dynamic": 0.0, "initial": 0.05},
            resampling_fn=resampling_fn,
        )

    keys = jax.random.split(jax.random.key(8), 2)
    batched = jax.jit(jax.vmap(fit))(keys)
    assert np.asarray(batched.success).all()
    assert batched.parameters["dynamic"].dtype == dtype
    for i, key in enumerate(keys):
        single = jax.jit(fit)(key)
        for name in theta:
            np.testing.assert_allclose(
                batched.parameter_swarm[name][i],
                single.parameter_swarm[name],
                rtol=2e-6,
                atol=1e-7,
            )


def test_failed_pass_retains_initial_population():
    theta = jnp.array([0.4, -0.7, 0.2])

    def impossible(parameters):
        init, propagate, _ = model(parameters)
        return init, propagate, lambda old, x, y: jnp.array(-jnp.inf)

    result = iterated_filtering.fit(
        impossible,
        theta,
        jnp.array(Y),
        key=jax.random.key(1),
        n_particles=8,
        n_iterations=3,
        perturbation_scale=0.1,
    )
    assert not result.success and result.completed_iterations == 0
    assert not np.asarray(result.step_accepted).any()
    np.testing.assert_array_equal(result.parameters, theta)
    np.testing.assert_array_equal(
        result.parameter_swarm, np.broadcast_to(theta, (8, 3))
    )
    assert np.isnan(np.asarray(result.perturbed_log_likelihoods)[1:]).all()


def test_invalid_traced_scale_reports_failure():
    theta = jnp.array([0.4, -0.7, 0.2])

    @jax.jit
    def fit(scale):
        return iterated_filtering.fit(
            model,
            theta,
            jnp.array(Y),
            key=jax.random.key(2),
            n_particles=8,
            n_iterations=2,
            perturbation_scale=scale,
        )

    result = fit(jnp.array(-0.1))
    assert not result.success and result.completed_iterations == 0
    np.testing.assert_array_equal(result.parameters, theta)


def test_zero_passes_preserves_parameters():
    theta = jnp.array([0.4, -0.7, 0.2])
    result = iterated_filtering.fit(
        model,
        theta,
        jnp.array(Y),
        key=jax.random.key(3),
        n_particles=8,
        n_iterations=0,
        perturbation_scale=0.1,
    )
    assert result.success and result.completed_iterations == 0
    np.testing.assert_array_equal(result.parameter_history, theta[None])


def test_zero_perturbations_match_original_particle_filter():
    from cuthbert import filter as run_filter
    from cuthbert.smc import particle_filter

    theta, observations, key = (
        jnp.array([0.4, -0.7, 0.2]),
        jnp.array(Y),
        jax.random.key(4),
    )
    result = iterated_filtering.fit(
        model,
        theta,
        observations,
        key=key,
        n_particles=16,
        n_iterations=1,
        perturbation_scale=0.0,
    )
    _, init_key, filter_key, _ = jax.random.split(jax.random.split(key, 1)[0], 4)
    init, propagate, potential = model(theta)

    def same_process_keys(key, state, observation):
        # IF2 splits perturbation and process keys even when the scale is zero.
        return propagate(jax.random.split(key)[1], state, observation)

    pf = particle_filter.build_filter(
        init, same_process_keys, potential, 16, systematic.resampling
    )
    expected = run_filter(
        pf, observations, pf.init_prepare(key=init_key), key=filter_key
    )
    np.testing.assert_allclose(
        result.perturbed_log_likelihoods[0],
        expected.log_normalizing_constant[-1],
        rtol=2e-6,
        atol=1e-7,
    )
