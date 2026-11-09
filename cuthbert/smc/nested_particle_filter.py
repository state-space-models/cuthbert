r"""Implements the nested particle filter (NPF) of Crisan and Míguez (2018).

Link to paper: https://www.jstor.org/stable/26492094

The NPF targets the joint filtering distribution of a parameter $\theta$ and
the state $x_t$. It is implemented as a bootstrap particle filter on an
augmented state, where each outer particle is a parameter value $\theta^{(i)}$
paired with the state of an inner filter for $x_t$ conditional on
$\theta^{(i)}$. The weight of each outer particle at time $t$ is the inner
filter's estimate of the incremental likelihood
$p(y_t \mid y_{1:t-1}, \theta^{(i)})$.

This implementation is more flexible than the algorithm from the original paper:

- The inner filter does not need to be a particle filter, and can be any
  `cuthbert` filter whose state is a `NamedTuple` with a `model_inputs` field
  and a cumulative `log_normalizing_constant`. For example, if conditionally on
  a parameter the model is linear-Gaussian, then we could use a Kalman filter
  for the inner filter to yield a nested hybrid filter, in the style of
  https://doi.org/10.1103/PhysRevE.98.063305.
- The paper jitters the parameters at every step. Here we only jitter if the
  parameter particles have been resampled, and we allow this to be done adaptively
  (when the ESS goes below a threshold).
"""

from typing import Callable, NamedTuple

import jax.numpy as jnp
from jax import random, tree

from cuthbert.inference import Filter
from cuthbert.smc import particle_filter
from cuthbert.smc.types import InitSample, PropagateSample
from cuthbertlib.resampling import Resampling
from cuthbertlib.resampling.adaptive import ess_decorator
from cuthbertlib.types import Array, ArrayLike, ArrayTree, ArrayTreeLike, KeyArray


class NestedParticle(NamedTuple):
    r"""An outer particle of the nested particle filter.

    The fields below describe a single outer particle, as built inside the
    filter, but they may have leading batch dimensions (e.g. a particle axis).

    Attributes:
        param: The parameter value $\theta_t$, an arbitrary pytree (e.g. a
            scalar, a vector or a dict of parameters).
        inner_state: Inner filter state conditional on `param`.
        resampled: Whether this particle was produced by a resampling operation
            immediately before the current propagation.
    """

    param: ArrayTree
    inner_state: ArrayTree
    resampled: Array


def build_filter(
    inner_filter: Filter,
    init_param_sample: InitSample,
    propagate_param_sample: PropagateSample,
    add_param_to_model_inputs: Callable[[ArrayTree, ArrayTreeLike], ArrayTreeLike],
    n_filter_particles: int,
    resampling_fn: Resampling,
    ess_threshold: float | None = None,
) -> Filter:
    r"""Builds a nested particle filter object.

    Args:
        inner_filter: Filter for the state $x_t$ conditional on a parameter,
            e.g., a particle filter or a Kalman filter. Its model functions read the
            parameter from their model inputs, as constructed by
            `add_param_to_model_inputs`. Its state must have a
            `log_normalizing_constant` attribute.
        init_param_sample: Function of a JAX random key only, generates a single
            sample from the prior on the parameter $\theta$.
        propagate_param_sample: Function to jitter the parameter, i.e., sample
            from a kernel $\kappa_t(\theta_t \mid \theta_{t-1})$.
        add_param_to_model_inputs: Function of a parameter and the model inputs
            at time $t$, returns the model inputs passed to `inner_filter` at
            time $t$. For example, `lambda param, y: (param, y)`.
        n_filter_particles: Number of outer (parameter) particles.
        resampling_fn: Resampling algorithm to use (e.g., systematic, multinomial).
            Unlike the other particle filters, if you want adaptive resampling, do
            *not* wrap this with the `ess_decorator`. Provide an
            `ess_threshold` instead.
        ess_threshold: Fraction of `n_filter_particles` below which the effective
            sample size triggers resampling (and jittering) of the outer
            particles. If `None`, resampling happens at every step.

    Returns:
        Filter object for the nested particle filter. Its state is a particle
            filter state whose particles are `NestedParticle`s, with a leading
            outer particle axis on every leaf.
    """

    def init_sample(key: KeyArray) -> NestedParticle:
        outer_key, inner_key = random.split(key)
        return NestedParticle(
            param=init_param_sample(outer_key),
            inner_state=inner_filter.init_prepare(key=inner_key),
            resampled=jnp.array(False),
        )

    def propagate_sample(
        key: KeyArray, state: NestedParticle, model_inputs: ArrayTreeLike
    ) -> NestedParticle:
        outer_key, inner_key = random.split(key)
        param = propagate_param_sample(outer_key, state.param, model_inputs)
        param = tree.map(
            lambda new, old: jnp.where(state.resampled, new, old), param, state.param
        )

        inner_state_prepared = inner_filter.filter_prepare(
            add_param_to_model_inputs(param, model_inputs), key=inner_key
        )
        inner_state = inner_filter.filter_combine(
            state.inner_state, inner_state_prepared
        )
        return NestedParticle(
            param=param,
            inner_state=inner_state._replace(model_inputs=None),
            resampled=jnp.array(False),
        )

    def log_potential(
        state_prev: NestedParticle, state: NestedParticle, model_inputs: ArrayTreeLike
    ) -> Array:
        del model_inputs
        return (
            state.inner_state.log_normalizing_constant
            - state_prev.inner_state.log_normalizing_constant
        )

    # Wrap the resampling function
    resampling_fn = _mark_resampled(resampling_fn)
    if ess_threshold is not None:
        resampling_fn = ess_decorator(resampling_fn, ess_threshold)

    return particle_filter.build_filter(
        init_sample, propagate_sample, log_potential, n_filter_particles, resampling_fn
    )


def _mark_resampled(resampling_fn: Resampling) -> Resampling:
    """Wraps a resampling function to flag when a resampling has occurred.

    This is the key to making the nested particle filter work without modifying
    code elsewhere in `cuthbert`. Essentially, we only want to jitter (i.e., add
    noise to) the parameters adaptively, when the ESS has fallen below a
    threshold (just like for adaptive resampling). This means we need to keep
    track of whether we have resampled or not, which the resampling functions in
    `cuthbert` currently do not tell us. What we do here is attach a flag of
    whether or not resampling happened to every particle by wrapping the
    resampling function.
    """

    def wrapped(
        key: KeyArray, logits: ArrayLike, positions: ArrayTreeLike, n: int
    ) -> tuple[Array, Array, ArrayTree]:
        ancestor_indices, log_weights, ancestors = resampling_fn(
            key, logits, positions, n
        )
        ancestors = ancestors._replace(
            resampled=jnp.full_like(ancestors.resampled, True)
        )
        return ancestor_indices, log_weights, ancestors

    return wrapped
