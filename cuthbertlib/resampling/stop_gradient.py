"""Stop gradient resampling decorator.

The stop_gradient resampling scheme provides the classical Fisher estimates for
the score function via automatic differentiation. This can be wrapped around a
resampling scheme such as multinomial or systematic resampling.

See [Scibior and Wood (2021)](https://arxiv.org/abs/2106.10314) for more details.
"""

from functools import wraps

import jax
import jax.numpy as jnp

from cuthbertlib.resampling.protocols import Resampling
from cuthbertlib.resampling.utils import apply_resampling_indices
from cuthbertlib.smc.ess import log_ess
from cuthbertlib.types import Array, ArrayLike, ArrayTree, ArrayTreeLike


def stop_gradient_decorator(func: Resampling) -> Resampling:
    """Wrap a Resampling function to use stop gradient resampling.

    Args:
        func: A resampling function with signature
              (key, logits, positions, n) -> (indices, logits_out, positions_out).

    Returns:
        A Resampling function implementing stop gradient resampling.
    """
    # Build a descriptive docstring that includes the wrapped function doc
    wrapped_doc = func.__doc__ or ""
    doc = f"""
    Stop gradient resampling decorator.

    This wrapper will call the provided resampling function, and then apply 
    the stop gradient trick of [Scibior and Wood (2021)](https://arxiv.org/abs/2106.10314). 
    Resulting estimates of the score function (i.e., the gradient of the 
    log-likelihood with respect to model parameters) are unbiased, 
    corresponding to the classical Fisher estimate.

    Wrapped resampler documentation:
    {wrapped_doc}
    """

    @wraps(func)
    def _wrapped(
        key: Array, logits: ArrayLike, positions: ArrayTreeLike, n: int
    ) -> tuple[Array, Array, ArrayTree]:
        idx_base, logits_base, positions_base = func(
            key, jax.lax.stop_gradient(logits), positions, n
        )

        logits = jnp.asarray(
            logits_base
            + apply_resampling_indices(logits, idx_base)
            - jax.lax.stop_gradient(apply_resampling_indices(logits, idx_base))
        )
        return idx_base, logits, positions_base

    # Attach the composed docstring and return a jitted version
    _wrapped.__doc__ = doc
    return jax.jit(_wrapped, static_argnames=("n",))
