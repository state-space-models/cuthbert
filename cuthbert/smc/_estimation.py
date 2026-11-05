"""Helpers for iterated filtering with bootstrap model callbacks."""

import math
from collections.abc import Callable
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax import core

from cuthbert.smc.types import InitSample, LogPotential, PropagateSample
from cuthbert.utils import dummy_leading_element
from cuthbertlib.resampling import multinomial, systematic


class ModelFactory(Protocol):
    """Build bootstrap callbacks from an estimation-scale parameter PyTree.

    Apply constraints (exp, sigmoid, etc.) inside this pure JAX function. The
    returned functions have the same signatures as ``particle_filter.build_filter``.
    The potential must be the observation log density, not a guided-proposal
    correction. IF2 does not require derivatives of the model callbacks.
    """

    def __call__(
        self, parameters: Any, /
    ) -> tuple[InitSample, PropagateSample, LogPotential]:
        """Return initial sampler, transition sampler and observation potential."""
        ...


def positive_int(name, value, *, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < (0 if allow_zero else 1):
        raise ValueError(f"{name} is out of range")


def scalar_range(name, value, lo, hi):
    if not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f"{name} must be finite and in [{lo}, {hi}]")


def parameters_tree(parameters):
    parameters = jax.tree.map(jnp.asarray, parameters)
    leaves = jax.tree.leaves(parameters)
    if not leaves or any(x.size == 0 for x in leaves):
        raise ValueError("parameters must be a nonempty PyTree of nonempty arrays")
    if any(not jnp.issubdtype(x.dtype, jnp.floating) for x in leaves):
        raise ValueError("parameters must have real floating-point dtypes")
    return parameters


def inputs_tree(model_inputs):
    model_inputs = jax.tree.map(jnp.asarray, model_inputs)
    leaves = jax.tree.leaves(model_inputs)
    if not leaves or any(x.ndim == 0 for x in leaves):
        raise ValueError("model_inputs must have a leading observation axis")
    length = leaves[0].shape[0]
    if length < 1 or any(x.shape[0] != length for x in leaves):
        raise ValueError("model_inputs must share a nonempty observation axis")
    return model_inputs


def broadcast_tree(value, template, *, name, boolean=False):
    # A scalar can control every coordinate; otherwise require the exact tree.
    if np.isscalar(value) or (hasattr(value, "ndim") and value.ndim == 0):
        value = jax.tree.map(lambda _: value, template)
    if jax.tree.structure(value) != jax.tree.structure(template):
        raise ValueError(f"{name} must be scalar or match the parameter PyTree")

    def convert(v, p):
        if boolean and not jnp.issubdtype(jnp.asarray(v).dtype, jnp.bool_):
            raise ValueError(f"{name} must contain boolean values")
        v = jnp.asarray(v, dtype=bool if boolean else p.dtype)
        try:
            v = jnp.broadcast_to(v, p.shape)
        except ValueError as exc:
            raise ValueError(f"{name} has an incompatible leaf shape") from exc
        if not boolean:
            valid = jnp.isfinite(v) & (v >= 0)
            all_valid = jnp.all(valid)
            # A captured array may be concrete while this predicate is traced.
            if not isinstance(all_valid, core.Tracer) and not bool(all_valid):
                raise ValueError(f"{name} must be finite and nonnegative")
            # Invalid traced scales produce a reported failure, not silent use.
            v = jnp.where(valid, v, jnp.nan)
        return v

    return jax.tree.map(convert, value, template)


def resampler(method):
    if method == "systematic":
        return systematic.resampling
    if method == "multinomial":
        return multinomial.resampling
    raise ValueError("resampling must be 'systematic' or 'multinomial'")


def all_finite(value):
    leaves = jax.tree.leaves(value)
    if not leaves:
        return jnp.array(True)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in leaves]))


def select_tree(condition, new, old):
    return jax.tree.map(lambda a, b: jnp.where(condition, a, b), new, old)


def perturb(parameters, scale, cooling, key):
    leaves, structure = jax.tree.flatten(parameters)
    scales = jax.tree.leaves(scale)
    keys = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(
        structure,
        [
            p
            + jnp.asarray(cooling, dtype=p.dtype)
            * s
            * jax.random.normal(k, p.shape, dtype=p.dtype)
            for p, s, k in zip(leaves, scales, keys, strict=True)
        ],
    )


def scan_final(filter_obj, model_inputs, init_state, key, *, checkpoint):
    """Match cuthbert.filter's key order, returning only its terminal state."""
    length = jax.tree.leaves(model_inputs)[0].shape[0]
    prepare_keys = jax.random.split(key, length)
    init_state = init_state._replace(model_inputs=dummy_leading_element(model_inputs))

    def step(previous, inp_and_key):
        inp, k = inp_and_key
        prepared = filter_obj.filter_prepare(inp, key=k)
        state = filter_obj.filter_combine(previous, prepared)
        return state, None

    body: Callable = jax.checkpoint(step) if checkpoint else step
    return jax.lax.scan(body, init_state, (model_inputs, prepare_keys))[0]
