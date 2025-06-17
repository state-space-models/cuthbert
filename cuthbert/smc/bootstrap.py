from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
from jax import Array, random, tree

from cuthbertlib.resampling import Resampling
from cuthbertlib.types import ArrayTree, ArrayTreeLike, KeyArray, ScalarArray


class M0(Protocol):
    """Get a sample from the initial distribution $M_0(x_0)$."""

    def __call__(self, key: KeyArray, model_inputs: ArrayTreeLike) -> ArrayTree: ...


class Mt(Protocol):
    """Sample from the Markov kernel $M_t(x_t \\mid x_{t-1})$."""

    def __call__(
        self, key: KeyArray, state: ArrayTreeLike, model_inputs: ArrayTreeLike
    ) -> ArrayTree: ...


class LogG0(Protocol):
    """Compute the log potential function $\\log G_0(x_0)$."""

    def __call__(
        self, state: ArrayTreeLike, model_inputs: ArrayTreeLike
    ) -> ScalarArray: ...


class LogGt(Protocol):
    """Compute the log potential function $\\log G_t(x_{t-1}, x_t)$."""

    def __call__(
        self,
        state_prev: ArrayTreeLike,
        state: ArrayTreeLike,
        model_inputs: ArrayTreeLike,
    ) -> ScalarArray: ...


class BootstrapFilterState(NamedTuple):
    key: KeyArray
    particles: ArrayTree | None
    log_weights: Array | None
    ancestor_indices: Array | None
    model_inputs: ArrayTreeLike
    log_likelihood_incr: ScalarArray | None


def init_prepare(
    model_inputs: ArrayTreeLike,
    m0: M0,
    log_g0: LogG0,
    num_samples: int,
    key: KeyArray,
) -> BootstrapFilterState:
    keys = random.split(key, num_samples)
    particles = jax.vmap(m0, (0, None))(keys, model_inputs)
    log_weights = jax.vmap(log_g0, (0, None))(particles, model_inputs)
    return BootstrapFilterState(
        key, particles, log_weights, jnp.array(0.0), model_inputs, None
    )


def filter_prepare(
    model_inputs: ArrayTreeLike,
    num_samples: int,
    key: KeyArray,
) -> BootstrapFilterState:
    key, sub_key = random.split(key)
    return BootstrapFilterState(key, None, None, None, model_inputs, None)


def filter_combine(
    state_1: BootstrapFilterState,
    state_2: BootstrapFilterState,
    mt: Mt,
    log_gt: LogGt,
    resampling_fn: Resampling,
    ess_threshold: float,
) -> BootstrapFilterState:
    if state_1.particles is None or state_1.log_weights is None:
        raise ValueError("state_1 must have particles and log_weights")

    N = state_1.particles.shape[0]
    keys = random.split(state_1.key, N + 1)

    # Resample
    ancestor_indices, log_weights = jax.lax.cond(
        log_ess(state_1.log_weights) < jnp.log(ess_threshold),
        lambda: (resampling_fn(keys[0], state_1.log_weights, N), jnp.zeros(N)),  # type: ignore
        lambda: (jnp.arange(N), state_1.log_weights),
    )
    ancestors = tree.map(lambda x: x[ancestor_indices], state_1.particles)

    # Move
    next_particles = jax.vmap(mt, (0, 0, None))(
        keys[1:], ancestors, state_1.model_inputs
    )

    # Reweight
    log_potentials = jax.vmap(log_gt, (0, 0, None))(
        ancestors, next_particles, state_1.model_inputs
    )
    next_log_weights = log_potentials + log_weights

    # Compute the log likelihood increment
    logsum_weights = jax.nn.logsumexp(next_log_weights)
    log_likelihood_incr = logsum_weights - jax.nn.logsumexp(log_weights)

    return BootstrapFilterState(
        state_2.key,
        next_particles,
        next_log_weights,
        ancestor_indices,
        state_2.model_inputs,
        log_likelihood_incr,
    )


def log_ess(log_weights: Array) -> Array:
    return 2 * jax.nn.logsumexp(log_weights) - jax.nn.logsumexp(2 * log_weights)
