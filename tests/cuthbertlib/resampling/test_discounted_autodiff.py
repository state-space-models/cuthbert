"""Check discounted resampling and the full bootstrap-filter recursion."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cuthbert import filter as run_filter
from cuthbert.smc import particle_filter
from cuthbertlib.resampling import (
    autodiff,
    discounted_stop_gradient_decorator,
    multinomial,
    systematic,
)


@pytest.fixture(autouse=True)
def precision():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize(
    "alpha", [-0.1, 1.1, float("nan"), float("inf"), -float("inf")]
)
def test_invalid_discount(alpha):
    with pytest.raises(ValueError, match="finite and in"):
        discounted_stop_gradient_decorator(systematic.resampling, alpha)


def test_alpha_one_delegates_to_existing_decorator(monkeypatch):
    sentinel = object()
    calls = []

    def original(func):
        calls.append(func)
        return sentinel

    monkeypatch.setattr(autodiff, "stop_gradient_decorator", original)
    assert discounted_stop_gradient_decorator(systematic.resampling, 1.0) is sentinel
    assert calls == [systematic.resampling]


@pytest.mark.parametrize("base", [systematic.resampling, multinomial.resampling])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 0.97, 1.0])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pytree_forward_and_weight_derivatives(base, alpha, dtype):
    # Distinct input/output counts exercise repeated and omitted ancestors.
    logits = jnp.array([-1.0, 0.4, 0.8, -0.2], dtype=dtype)
    positions = {"x": jnp.arange(8, dtype=dtype).reshape(4, 2), "z": (logits,)}
    keys = jax.random.split(jax.random.key(73), 3)
    wrapped = discounted_stop_gradient_decorator(base, alpha)
    actual = jax.jit(jax.vmap(lambda k: wrapped(k, logits, positions, 7)))(keys)
    expected = jax.vmap(lambda k: base(k, logits, positions, 7))(keys)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(a, b)
        assert a.dtype == b.dtype

    indices = actual[0][0]
    jacobian = jax.jacrev(lambda w: wrapped(keys[0], w, positions, 7)[1])(logits)
    np.testing.assert_allclose(
        jacobian, alpha * jnp.eye(4, dtype=dtype)[indices], rtol=1e-6, atol=0
    )
    # State derivatives survive even when alpha is zero.
    state_jacobian = jax.jacrev(
        lambda x: wrapped(keys[0], logits, {"x": x}, 7)[2]["x"]
    )(logits)
    np.testing.assert_array_equal(state_jacobian, jnp.eye(4, dtype=dtype)[indices])


def model(theta):
    phi, log_q, initial_mean = theta

    def init(key):
        return initial_mean + jax.random.normal(key, (), dtype=theta.dtype)

    def propagate(key, x, y):
        return jnp.tanh(phi) * x + jnp.exp(log_q) * jax.random.normal(
            key, (), dtype=theta.dtype
        )

    def potential(old, x, y):
        return -0.5 * ((y - x) / 0.7) ** 2 - jnp.log(0.7 * jnp.sqrt(2 * jnp.pi))

    return init, propagate, potential


def filter_run(theta, observations, key, resampler, n):
    obj = particle_filter.build_filter(*model(theta), n, resampler)
    init_key, filter_key = jax.random.split(key)
    return run_filter(obj, observations, obj.init_prepare(key=init_key), key=filter_key)


@pytest.mark.parametrize("base", [systematic.resampling, multinomial.resampling])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 0.97, 1.0])
def test_filter_against_fixed_reference_recursion(base, alpha):
    theta = jnp.array([0.4, -0.7, 0.2])
    observations = jnp.array([0.3, -0.5, 0.7, 1.0, 0.2])
    key, n = jax.random.key(13), 32
    stock = filter_run(theta, observations, key, base, n)
    wrapped = discounted_stop_gradient_decorator(base, alpha)
    corrected = filter_run(theta, observations, key, wrapped, n)
    for field in ("particles", "ancestor_indices", "log_weights"):
        np.testing.assert_array_equal(getattr(stock, field), getattr(corrected, field))
    # Wrapping the resampler can change XLA reduction rounding in the filter.
    np.testing.assert_allclose(
        stock.log_normalizing_constant,
        corrected.log_normalizing_constant,
        rtol=1e-14,
        atol=1e-14,
    )

    # Independently evaluate the off-parameter MOP recursion with FIXED ancestry,
    # base noises and reference potentials. This reference has no stop_gradient.
    # Finite differences of the ordinary forward PF would test a different target.
    init_key, filter_key = jax.random.split(key)
    initial_noise = jax.vmap(jax.random.normal)(jax.random.split(init_key, n))
    prepared_keys = jax.random.split(filter_key, len(observations))
    step_keys = jnp.concatenate([init_key[None], prepared_keys[:-1]])
    noise = jax.vmap(
        lambda k: jax.vmap(jax.random.normal)(jax.random.split(k, n + 1)[1:])
    )(step_keys)
    ancestry = stock.ancestor_indices[1:]
    fixed_potentials = jax.vmap(model(theta)[2])(
        stock.particles[:-1], stock.particles[1:], observations
    )

    def reference(candidate):
        x = (candidate[2] + initial_noise)[ancestry[0]]
        log_ratios, log_likelihood = jnp.zeros(n), jnp.array(0.0)
        for t in range(len(observations)):
            prior = alpha * log_ratios
            x = jnp.tanh(candidate[0]) * x + jnp.exp(candidate[1]) * noise[t]
            potential = model(candidate)[2](x, x, observations[t])
            log_likelihood += jax.scipy.special.logsumexp(
                prior + potential
            ) - jax.scipy.special.logsumexp(prior)
            if t + 1 < len(observations):
                indices = ancestry[t + 1]
                x = x[indices]
                log_ratios = (prior + potential - fixed_potentials[t])[indices]
        return log_likelihood

    def objective(candidate):
        return filter_run(
            candidate, observations, key, wrapped, n
        ).log_normalizing_constant[-1]

    value, gradient = jax.jit(jax.value_and_grad(objective))(theta)
    reference_value, reference_gradient = jax.value_and_grad(reference)(theta)
    np.testing.assert_allclose(value, reference_value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gradient, reference_gradient, rtol=1e-12, atol=1e-12)
    epsilon = 1e-5
    finite_difference = jnp.array(
        [
            (
                reference(theta + epsilon * direction)
                - reference(theta - epsilon * direction)
            )
            / (2 * epsilon)
            for direction in jnp.eye(len(theta))
        ]
    )
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-7, atol=2e-8)
    if alpha in (0.0, 1.0):
        baseline = base if alpha == 0 else autodiff.stop_gradient_decorator(base)
        baseline_gradient = jax.grad(
            lambda candidate: filter_run(
                candidate, observations, key, baseline, n
            ).log_normalizing_constant[-1]
        )(theta)
        np.testing.assert_allclose(gradient, baseline_gradient, rtol=1e-12, atol=1e-12)
