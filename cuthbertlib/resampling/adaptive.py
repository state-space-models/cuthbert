"""Adaptive resampling decorator.

Provides a decorator to turn any Resampling function into an adaptive resampling
function which performs resampling only when the effective sample size (ESS)
falls below a threshold.
"""

from functools import wraps

import jax
import jax.numpy as jnp

from cuthbertlib.resampling.protocols import Resampling
from cuthbertlib.smc.ess import log_ess
from cuthbertlib.types import Array, ArrayLike, ArrayTree, ArrayTreeLike


def ess_decorator(func: Resampling, threshold: float) -> Resampling:
    """Wrap a Resampling function so that it only resamples when ESS < threshold.

    The returned function is jitted and has `n` as a static argument. The
    original resampler's docstring is appended to this wrapper's docstring so
    IDEs and users can see the underlying algorithm documentation.

    Args:
        func: A resampling function with signature
              (key, logits, positions, n) -> (indices, logits_out, positions_out).
        threshold: Fraction of particle count specifying when to resample.
            Resampling is triggered when ESS < ess_threshold * n.

    Returns:
        A Resampling function implementing adaptive resampling.
    """
    # Build a descriptive docstring that includes the wrapped function doc
    wrapped_doc = func.__doc__ or ""
    doc = f"""
    Adaptive resampling decorator (threshold={threshold}).

    This wrapper will call the provided resampling function only when the
    effective sample size (ESS) is below `ess_threshold * n`.

    Wrapped resampler documentation:
    {wrapped_doc}
    """

    @wraps(func)
    def _wrapped(
        key: Array, logits: ArrayLike, positions: ArrayTreeLike, n: int
    ) -> tuple[Array, Array, ArrayTree]:
        logits_arr = jnp.asarray(logits)
        N = logits_arr.shape[0]
        if n != N:
            raise AssertionError(
                "The number of sampled indices must be equal to the number of "
                f"particles for `adaptive` resampling. Got {n} instead of {N}."
            )

        def _do_resample():
            return func(key, logits_arr, positions, n)

        def _no_resample():
            return jnp.arange(n), logits_arr, positions

        return jax.lax.cond(
            log_ess(logits_arr) < jnp.log(threshold * n),
            _do_resample,
            _no_resample,
        )

    # Attach the composed docstring and return a jitted version
    _wrapped.__doc__ = doc
    return jax.jit(_wrapped, static_argnames=("n",))
