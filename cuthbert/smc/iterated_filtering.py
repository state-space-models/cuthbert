"""Iterated filtering with Gaussian perturbations on the estimation scale.

Particles carry a model state and a parameter PyTree with shared ancestry.
Each pass starts from the previous pass's terminal, equally weighted parameter
swarm. This module uses unconditional bootstrap resampling at every observation.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from cuthbert.smc import _estimation as util
from cuthbert.smc import particle_filter as pf
from cuthbert.smc._estimation import ModelFactory


class IteratedFilteringResult(NamedTuple):
    """IF2 result on the estimation scale.

    Parameters are the arithmetic mean of the terminal, equally weighted swarm.
    ``perturbed_log_likelihoods`` belong to the perturbed models, NOT to the
    static model at these parameters. Use independent unperturbed PF evaluation
    to compare endpoints. On failure the last valid swarm is retained.
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
    resampling: str = "systematic",
) -> IteratedFilteringResult:
    """Run IF2 with Gaussian parameter random walks and geometric cooling.

    Pass m uses ``cooling_rate**m`` for both initial and within-pass noise.
    Initial noise is applied before rinit; process noise is applied before each
    observation interval, including the first. To perturb an IVP only at rinit,
    give it a nonzero initial scale and a zero process scale.

    Args:
        model_factory: Estimation-scale parameters to bootstrap callbacks.
        initial_parameters: Floating-point parameter PyTree. Use vmap for starts.
        model_inputs: Observation inputs sharing a leading time axis.
        key: Root key for independent initial, process and terminal draws.
        n_particles: Positive static size of the state/parameter swarm.
        n_iterations: Nonnegative static number of passes through the data.
        perturbation_scale: Nonnegative scalar or parameter-shaped PyTree of
            standard deviations on the estimation scale.
        initial_perturbation_scale: Initial standard deviations, defaulting to
            perturbation_scale. Distinct scales support initial-value parameters.
        cooling_rate: Static geometric multiplier in (0, 1].
        resampling: Unconditional 'systematic' or 'multinomial'.

    Returns:
        Parameter swarm, means, perturbed likelihoods and failure diagnostics.
    """
    util.positive_int("n_particles", n_particles)
    util.positive_int("n_iterations", n_iterations, allow_zero=True)
    util.scalar_range("cooling_rate", cooling_rate, 1e-30, 1)
    base_resampling = util.resampler(resampling)
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
            shape_init, propagate, potential, n_particles, base_resampling
        )
        start = pf.ParticleFilterState(
            init_key,
            {"state": states, "parameters": swarm},
            jnp.zeros(n_particles),
            jnp.arange(n_particles),
            None,
            jnp.array(0.0),
        )
        final = util.scan_final(obj, inputs, start, filter_key, checkpoint=False)
        finite = (
            jnp.isfinite(final.log_normalizing_constant)
            & util.all_finite(final.particles)
            & jnp.any(jnp.isfinite(final.log_weights))
        )

        def finish(_):
            # cuthbert resamples BEFORE propagation, leaving weighted particles
            # after the last observation. Explicitly resample the terminal swarm.
            _, _, particles = base_resampling(
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
