"""Implements the nested particle filter of Crisan and Miguez (2018) for parameter estimation.

Reference:
    Crisan and Miguez (2018) -  https://doi.org/10.3150/17-BEJ954
"""

from functools import partial
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
from jax import random, tree

from cuthbert.inference import Filter
from cuthbert.npf.types import (
    JitteringKernel,
    LogPotential,
    PropagateSample,
    SampleParam,
)
from cuthbert.smc.types import InitSample
from cuthbert.utils import dummy_tree_like
from cuthbertlib.resampling import Resampling
from cuthbertlib.types import Array, ArrayTree, ArrayTreeLike, KeyArray, ScalarArray


class NPFState(NamedTuple):
    """Nested particle filter state.

    Attributes:
        key: JAX PRNG key.
        param_particles: Parameter particles for the outer filter.
        state_particles: State particles for the inner filters. The leading
            axes index parameter particles and state particles, respectively.
        param_log_weights: Log weights for the parameter particles.
        state_log_weights: Log weights for the state particles, with shape
            `(n_param_particles, n_state_particles)`.
        model_inputs: Model inputs associated with this filter state.
        log_normalizing_constant: Current estimate of the log normalizing
            constant.
    """

    key: KeyArray
    param_particles: ArrayTree
    state_particles: ArrayTree
    param_log_weights: Array
    state_log_weights: Array
    model_inputs: ArrayTree
    log_normalizing_constant: ScalarArray


def build_filter(
    init_param_sample: SampleParam,
    init_sample: InitSample,
    propagate_sample: PropagateSample,
    log_potential: LogPotential,
    n_param_particles: int,
    n_state_particles: int,
    resampling_fn: Resampling,
    kernel_fn: JitteringKernel,
) -> Filter:
    r"""Builds a nested particle filter object.

    Args:
        init_param_sample: Function to sample from the initial parameter
            distribution $\mu(\theta_0)$.
        init_sample: Function to sample from the initial state distribution
            $M_0(x_0)$.
        propagate_sample: Function to sample from the Markov kernel $M_t(x_t \mid x_{t-1}, \theta)$.
        log_potential: Function to compute the log potential $\log G_t(x_{t-1}, x_t, \theta)$.
        n_param_particles: Number of parameter particles for the outer filter.
        n_state_particles: Number of state particles for the inner filters.
        resampling_fn: Resampling algorithm to use (e.g., systematic, multinomial).
            The resampling function may be decorated with adaptive behaviour
            (using cuthbertlib.resampling.adaptive.adaptive_resampling_decorator)
            before being passed to the filter. The same resampling function is
            used for both the outer and inner filters.
        kernel_fn: The jittering kernel to use for the parameter particles.
            See Section 4.2 in Crisan and Miguez (2018) for different choices.

    Returns:
        Filter object for the nested particle filter.
    """
    return Filter(
        init_prepare=partial(
            init_prepare,
            init_param_sample=init_param_sample,
            init_state_sample=init_sample,
            n_param_particles=n_param_particles,
            n_state_particles=n_state_particles,
        ),
        filter_prepare=partial(
            filter_prepare,
            init_param_sample=init_param_sample,
            init_state_sample=init_sample,
            n_param_particles=n_param_particles,
            n_state_particles=n_state_particles,
        ),
        filter_combine=partial(
            filter_combine,
            propagate_sample=propagate_sample,
            log_potential=log_potential,
            resampling_fn=resampling_fn,
            kernel_fn=kernel_fn,
        ),
    )


def init_prepare(
    model_inputs: ArrayTreeLike,
    init_param_sample: SampleParam,
    init_state_sample: InitSample,
    n_param_particles: int,
    n_state_particles: int,
    key: KeyArray | None = None,
) -> NPFState:
    r"""Prepares the initial state for the nested particle filter.

    Args:
        model_inputs: Model inputs for the initial state distribution.
        init_param_sample: Function to sample from the initial parameter
            distribution $\mu(\theta_0)$.
        init_state_sample: Function to sample from the initial state
            distribution $M_0(x_0)$.
        n_param_particles: Number of parameter particles for the outer filter.
        n_state_particles: Number of state particles for each inner filter.
        key: JAX PRNG key.

    Returns:
        Initial nested particle filter state.

    Raises:
        ValueError: If `key` is None.
    """
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)
    if key is None:
        raise ValueError("A JAX PRNG key must be provided.")

    param_key, state_key = random.split(key)

    # Sample parameters
    param_keys = random.split(param_key, n_param_particles)
    param_particles = jax.vmap(init_param_sample)(param_keys)
    param_log_weights = jnp.zeros(n_param_particles)

    # Sample states
    state_keys = random.split(state_key, n_param_particles * n_state_particles)
    state_particles = jax.vmap(init_state_sample, (0, None))(state_keys, model_inputs)
    state_particles = tree.map(
        lambda x: x.reshape((n_param_particles, n_state_particles) + x.shape[1:]),
        state_particles,
    )
    state_log_weights = jnp.zeros((n_param_particles, n_state_particles))

    return NPFState(
        key=key,
        param_particles=param_particles,
        state_particles=state_particles,
        param_log_weights=param_log_weights,
        state_log_weights=state_log_weights,
        model_inputs=model_inputs,
        log_normalizing_constant=jnp.array(0.0),
    )


def filter_prepare(
    model_inputs: ArrayTreeLike,
    init_param_sample: SampleParam,
    init_state_sample: InitSample,
    n_param_particles: int,
    n_state_particles: int,
    key: KeyArray | None = None,
) -> NPFState:
    r"""Prepares a state for a nested particle filter step.

    Args:
        model_inputs: Model inputs for the current filtering step.
        init_param_sample: Function to sample from the initial parameter
            distribution $\mu(\theta_0)$.
        init_state_sample: Function to sample from the initial state
            distribution $M_0(x_0)$.
        n_param_particles: Number of parameter particles for the outer filter.
        n_state_particles: Number of state particles for each inner filter.
        key: JAX PRNG key.

    Returns:
        Prepared nested particle filter state with placeholder parameter and
        state particles.

    Raises:
        ValueError: If `key` is None.
    """
    model_inputs = tree.map(lambda x: jnp.asarray(x), model_inputs)
    if key is None:
        raise ValueError("A JAX PRNG key must be provided.")

    param_key, state_key = random.split(key)
    dummy_param_particle = jax.eval_shape(init_param_sample, param_key)
    particles = tree.map(
        lambda x: jnp.empty((n_param_particles,) + x.shape), dummy_param_particle
    )
    param_particles = dummy_tree_like(particles)

    dummy_state_particle = jax.eval_shape(init_state_sample, state_key, model_inputs)
    state_particles = tree.map(
        lambda x: jnp.empty((n_param_particles, n_state_particles) + x.shape),
        dummy_state_particle,
    )
    state_particles = dummy_tree_like(state_particles)

    return NPFState(
        key=key,
        param_particles=param_particles,
        state_particles=state_particles,
        param_log_weights=jnp.zeros(n_param_particles),
        state_log_weights=jnp.zeros((n_param_particles, n_state_particles)),
        model_inputs=model_inputs,
        log_normalizing_constant=jnp.array(0.0),
    )


def _pf_step(
    key: KeyArray,
    particles: ArrayTree,
    log_weights: Array,
    param: ArrayTree,
    model_inputs: ArrayTree,
    propagate_sample: PropagateSample,
    log_potential: LogPotential,
    resampling_fn: Resampling,
):
    """Performs a single particle filter step."""
    N = log_weights.shape[0]
    keys = random.split(key, N + 1)

    # Resample - resampling_fn is expected to handle adaptivity if desired
    _, log_weights, ancestors = resampling_fn(keys[0], log_weights, particles, N)

    # Propagate
    next_particles = jax.vmap(propagate_sample, (0, 0, None, None))(
        keys[1:], ancestors, param, model_inputs
    )

    # Reweight
    log_potentials = jax.vmap(log_potential, (0, 0, None, None))(
        ancestors, next_particles, param, model_inputs
    )
    next_log_weights = log_potentials + log_weights

    # Compute the log normalizing constant
    logsum_weights = jax.nn.logsumexp(next_log_weights)
    log_normalizing_constant_incr = logsum_weights - jax.nn.logsumexp(log_weights)

    return {
        "particles": next_particles,
        "log_weights": next_log_weights,
        "log_normalizing_constant_incr": log_normalizing_constant_incr,
    }


def filter_combine(
    state_1: NPFState,
    state_2: NPFState,
    propagate_sample: PropagateSample,
    log_potential: LogPotential,
    resampling_fn: Resampling,
    kernel_fn: JitteringKernel,
) -> NPFState:
    r"""Combines previous filter state with the state prepared for the current step.

    This is Algorithm 3 from Crisan and Miguez (2018) with one difference: we
    perform resampling before jittering and return weighted particles.

    Args:
        state_1: Nested particle filter state from the previous time step.
        state_2: Nested particle filter state prepared for the current time
            step.
        propagate_sample: Function to sample from the state Markov kernel
            $M_t(x_t \mid x_{t-1}, \theta)$.
        log_potential: Function to compute the log potential
            $\log G_t(x_{t-1}, x_t, \theta)$.
        resampling_fn: Resampling algorithm to use for the outer and inner
            filters.
        kernel_fn: Jittering kernel applied to resampled parameter particles.

    Returns:
        Nested particle filter state for the current time step.
    """
    N, M = state_1.state_log_weights.shape

    # Resample
    key, sub_key = random.split(state_1.key)
    ancestor_indices, log_weights, ancestors = resampling_fn(
        sub_key, state_1.param_log_weights, state_1.param_particles, N
    )
    state_particles, state_log_weights = tree.map(
        lambda x: x[ancestor_indices],
        (state_1.state_particles, state_1.state_log_weights),
    )

    # Jitter
    keys = random.split(key, N + 1)
    # TODO: We should only jitter if the particles were resampled.
    param_particles = jax.vmap(kernel_fn)(keys[1:], ancestors)

    # Perform the inner particle filter step for each parameter particle
    keys = random.split(keys[0], N)
    next_inner_state = jax.vmap(_pf_step, (0, 0, 0, 0, None, None, None, None))(
        keys,
        state_particles,
        state_log_weights,
        param_particles,
        state_2.model_inputs,
        propagate_sample,
        log_potential,
        resampling_fn,
    )

    # The log potentials are the log normalizing constants increments from the inner particle filters
    next_log_weights = log_weights + next_inner_state["log_normalizing_constant_incr"]
    logsum_weights = jax.nn.logsumexp(next_log_weights)
    log_normalizing_constant_incr = logsum_weights - jax.nn.logsumexp(log_weights)
    log_normalizing_constant = (
        log_normalizing_constant_incr + state_1.log_normalizing_constant
    )

    return NPFState(
        key=state_2.key,
        param_particles=param_particles,
        state_particles=next_inner_state["particles"],
        param_log_weights=next_log_weights,
        state_log_weights=next_inner_state["log_weights"],
        model_inputs=state_2.model_inputs,
        log_normalizing_constant=log_normalizing_constant,
    )
