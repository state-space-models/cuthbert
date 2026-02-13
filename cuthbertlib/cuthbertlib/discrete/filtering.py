"""Implements the discrete HMM filtering associative operator."""

from typing import NamedTuple

import jax.numpy as jnp

from cuthbertlib.types import Array


class FilterScanElement(NamedTuple):
    """Elements carried through the discrete HMM filtering scan."""

    f: Array
    log_g: Array


def condition_on_obs(state_probs: Array, log_likelihoods: Array) -> tuple[Array, Array]:
    r"""Conditions a state distribution on an observation.

    Args:
        state_probs: Either the state transition probabilities or the initial distribution.
        log_likelihoods: Vector of $\log p(y_t \mid x_t)$ for each possible state $x_t$.

    Returns:
        The conditioned state and the log normalizing constant.
    """
    ll_max = log_likelihoods.max(axis=-1)
    A_cond = state_probs * jnp.exp(log_likelihoods - ll_max)
    norm = A_cond.sum(axis=-1)
    A_cond /= jnp.expand_dims(norm, axis=-1)
    return A_cond, jnp.log(norm) + ll_max


def filtering_operator(
    elem_ij: FilterScanElement, elem_jk: FilterScanElement
) -> FilterScanElement:
    """Binary associative operator for filtering in discrete HMMs.

    Args:
        elem_ij: Filter scan element.
        elem_jk: Filter scan element.

    Returns:
        The output of the associative operator applied to the input elements.
    """
    f, lognorm = condition_on_obs(elem_ij.f, elem_jk.log_g)
    f = f @ elem_jk.f
    log_g = elem_ij.log_g + lognorm
    return FilterScanElement(f, log_g)
