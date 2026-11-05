"""Run a small Dacca IF2 example and evaluate the unperturbed model."""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import model
import numpy as np

from cuthbert import filter as run_filter
from cuthbert.smc import iterated_filtering, particle_filter
from cuthbertlib.resampling import systematic


def run(months=12, particles=32, iterations=2, seed=20261103):
    """Return IF2 output and two independent ordinary-PF evaluations.

    Defaults exercise the code on the first twelve observations. They are not
    intended to produce a converged estimate or a precise likelihood estimate.
    """
    observations = model.inputs(months)
    parameters = jnp.asarray(model.PUBLISHED)
    fit_key, evaluation_key = jax.random.split(jax.random.key(seed))

    @jax.jit
    def fit(theta, key):
        return iterated_filtering.fit(
            model.factory,
            theta,
            observations,
            key=key,
            n_particles=particles,
            n_iterations=iterations,
            perturbation_scale=jnp.asarray(model.PROCESS_SCALE),
            initial_perturbation_scale=jnp.asarray(model.INITIAL_SCALE),
            cooling_rate=0.9,
            resampling="systematic",
        )

    started = time.perf_counter()
    result = jax.block_until_ready(fit(parameters, fit_key))
    fit_seconds = time.perf_counter() - started
    if not bool(result.success) or int(result.completed_iterations) != iterations:
        raise RuntimeError("IF2 stopped before completing the requested iterations")

    @jax.jit
    def evaluate(theta, key):
        initial_sampler, transition, log_potential = model.factory(theta)
        pf = particle_filter.build_filter(
            initial_sampler, transition, log_potential, particles, systematic.resampling
        )
        init_key, filter_key = jax.random.split(key)
        states = run_filter(
            pf, observations, pf.init_prepare(key=init_key), key=filter_key
        )
        return states.log_normalizing_constant[-1]

    evaluation_keys = jax.random.split(evaluation_key, 2)
    started = time.perf_counter()
    log_likelihoods = np.asarray(
        [evaluate(result.parameters, key) for key in evaluation_keys]
    )
    evaluation_seconds = time.perf_counter() - started
    if not np.isfinite(log_likelihoods).all():
        raise RuntimeError("An ordinary-PF evaluation returned a nonfinite value")
    summary = {
        "purpose": "Small runnable example, not an optimization comparison",
        "months": months,
        "euler_steps_per_month": 20,
        "particles": particles,
        "if2_iterations": int(result.completed_iterations),
        "estimation_coordinates": int(parameters.size),
        "seed": seed,
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "parameter_dtype": str(parameters.dtype),
        "success": bool(result.success),
        "fit_seconds_including_compilation": fit_seconds,
        "evaluation_seconds_including_compilation": evaluation_seconds,
        "perturbed_log_likelihoods": np.asarray(
            result.perturbed_log_likelihoods
        ).tolist(),
        "ordinary_pf_log_likelihoods": log_likelihoods.tolist(),
        "ordinary_pf_log_mean_likelihood": float(
            jax.scipy.special.logsumexp(log_likelihoods) - jnp.log(2.0)
        ),
        "evaluation_precision_assessed": False,
        "density_floor_evaluations": int(
            np.count_nonzero(np.isclose(log_likelihoods, months * model.LTOL))
        ),
    }
    arrays = {
        "parameters": np.asarray(result.parameters),
        "parameter_swarm": np.asarray(result.parameter_swarm),
        "parameter_history": np.asarray(result.parameter_history),
        "perturbed_log_likelihoods": np.asarray(result.perturbed_log_likelihoods),
        "step_accepted": np.asarray(result.step_accepted),
        "ordinary_pf_log_likelihoods": log_likelihoods,
        "fit_key": np.asarray(jax.random.key_data(fit_key)),
        "evaluation_keys": np.asarray(jax.random.key_data(evaluation_keys)),
    }
    return summary, arrays


def main():
    """Run the example, optionally saving its parameter population and history."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--particles", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--output", type=Path, help="New output directory")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("--output must name a new directory")
    summary, arrays = run(args.months, args.particles, args.iterations, args.seed)
    if args.output is not None:
        args.output.mkdir(parents=True)
        np.savez_compressed(args.output / "result.npz", **arrays)
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
