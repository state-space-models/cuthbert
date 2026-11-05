"""Iterated filtering (IF2) with Gaussian parameter perturbations."""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from cuthbert.smc import _estimation as util
from cuthbert.smc import particle_filter as pf
from cuthbert.smc._estimation import ModelFactory
from cuthbertlib.resampling import Resampling, multinomial, systematic


class IteratedFilteringResult(NamedTuple):
    """IF2 results on the estimation scale.

    ``parameters`` is the mean of the final parameter population.
    ``perturbed_log_likelihoods`` gives the log likelihood estimate for the
    perturbed model at each iteration. If an iteration fails, IF2 keeps the
    last valid parameter population.
    """

    parameters: Any
    parameter_swarm: Any
    parameter_history: Any
    perturbed_log_likelihoods: Any
    step_accepted: Any
    completed_iterations: Any
    success: Any


def fit(
    model_factory: ModelFactory,
    initial_parameters: Any,
    model_inputs: Any,
    *,
    key: Any,
    n_particles: int,
    n_iterations: int,
    perturbation_scale: Any,
    initial_perturbation_scale: Any = None,
    cooling_rate: float = 0.9,
    resampling_fn: Resampling = systematic.resampling,
) -> IteratedFilteringResult:
    """Estimate parameters by iterated filtering (IF2).

    At iteration ``m`` (counting from zero), IF2 multiplies both perturbation
    scales by ``cooling_rate**m``.
    To perturb a parameter only at initialization, give it a positive initial
    scale and a zero scale for the observation intervals.

    Args:
        model_factory: Function that takes parameters and returns the initial
            sampler, transition sampler and observation log density function.
        initial_parameters: Starting parameters on the estimation scale, as a
            PyTree of floating-point values.
        model_inputs: Model inputs with time along the first axis.
        key: JAX random key.
        n_particles: Number of particles.
        n_iterations: Number of IF2 iterations.
        perturbation_scale: Standard deviations for parameter perturbations
            before each observation interval. Use a nonnegative scalar or a
            PyTree matching the parameters.
        initial_perturbation_scale: Standard deviations before sampling the
            initial state. Defaults to ``perturbation_scale``.
        cooling_rate: Geometric cooling factor in (0, 1].
        resampling_fn: ``systematic.resampling`` or ``multinomial.resampling``.

    Returns:
        An ``IteratedFilteringResult`` with the final parameter population
        and the parameter means and log likelihood estimates for each iteration.
    """
    util.positive_int("n_particles", n_particles)
    util.positive_int("n_iterations", n_iterations, allow_zero=True)
    util.scalar_range("cooling_rate", cooling_rate, 1e-30, 1)
    if resampling_fn not in (systematic.resampling, multinomial.resampling):
        raise ValueError("IF2 requires systematic.resampling or multinomial.resampling")
    initial = util.parameters_tree(initial_parameters)
    inputs = util.inputs_tree(model_inputs)
    scale = util.broadcast_tree(perturbation_scale, initial, name="perturbation_scale")
    initial_scale = util.broadcast_tree(
        perturbation_scale
        if initial_perturbation_scale is None
        else initial_perturbation_scale,
        initial,
        name="initial_perturbation_scale",
    )
    initial_swarm = jax.tree.map(
        lambda p: jnp.broadcast_to(p, (n_particles,) + p.shape), initial
    )
    iteration_keys = jax.random.split(key, n_iterations)

    def one_pass(swarm, pass_key, cooling):
        perturb_key, init_key, filter_key, terminal_key = jax.random.split(pass_key, 4)
        swarm = util.perturb(swarm, initial_scale, cooling, perturb_key)
        state_keys = jax.random.split(init_key, n_particles)
        states = jax.vmap(lambda p, k: model_factory(p)[0](k))(swarm, state_keys)

        # This callback is used only for shape inference by filter_prepare.
        def shape_init(key):
            return {"state": model_factory(initial)[0](key), "parameters": initial}

        def propagate(key, state, model_inputs):
            noise_key, process_key = jax.random.split(key)
            parameters = util.perturb(state["parameters"], scale, cooling, noise_key)
            next_state = model_factory(parameters)[1](
                process_key, state["state"], model_inputs
            )
            return {"state": next_state, "parameters": parameters}

        def potential(state_prev, state, model_inputs):
            return model_factory(state["parameters"])[2](
                state_prev["state"], state["state"], model_inputs
            )

        obj = pf.build_filter(
            shape_init, propagate, potential, n_particles, resampling_fn
        )
        start = pf.ParticleFilterState(
            init_key,
            {"state": states, "parameters": swarm},
            jnp.zeros(n_particles),
            jnp.arange(n_particles),
            None,
            jnp.array(0.0),
        )
        final = util.scan_final(obj, inputs, start, filter_key)
        finite = (
            jnp.isfinite(final.log_normalizing_constant)
            & util.all_finite(final.particles)
            & jnp.any(jnp.isfinite(final.log_weights))
        )

        def finish(_):
            # cuthbert resamples BEFORE propagation, leaving weighted particles
            # after the last observation. Explicitly resample the terminal swarm.
            _, _, particles = resampling_fn(
                terminal_key, final.log_weights, final.particles, n_particles
            )
            return particles["parameters"]

        terminal_swarm = jax.lax.cond(finite, finish, lambda _: swarm, operand=None)
        return terminal_swarm, final.log_normalizing_constant, finite

    loglik_dtype = jax.eval_shape(one_pass, initial_swarm, key, jnp.asarray(1.0))[
        1
    ].dtype

    def step(carry, item):
        swarm, means, active = carry
        m, pass_key = item

        def advance(_):
            proposal, loglik, finite = one_pass(swarm, pass_key, cooling_rate**m)
            proposal_means = jax.tree.map(lambda p: jnp.mean(p, axis=0), proposal)
            finite = finite & util.all_finite(proposal_means)
            next_swarm = util.select_tree(finite, proposal, swarm)
            next_means = util.select_tree(finite, proposal_means, means)
            return (next_swarm, next_means, finite), (next_means, loglik, finite)

        def skip(_):
            return carry, (
                means,
                jnp.asarray(jnp.nan, dtype=loglik_dtype),
                jnp.array(False),
            )

        return jax.lax.cond(active, advance, skip, operand=None)

    valid = (
        util.all_finite(initial_swarm)
        & util.all_finite(scale)
        & util.all_finite(initial_scale)
    )
    (swarm, means, active), (history, logliks, accepted) = jax.lax.scan(
        step,
        (initial_swarm, initial, valid),
        (jnp.arange(n_iterations), iteration_keys),
    )
    history = jax.tree.map(lambda p, h: jnp.concatenate((p[None], h)), initial, history)
    return IteratedFilteringResult(
        means, swarm, history, logliks, accepted, jnp.sum(accepted), active
    )
