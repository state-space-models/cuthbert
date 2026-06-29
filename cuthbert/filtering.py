"""Unified cuthbert filtering interface."""

import warnings

from jax import numpy as jnp
from jax import random, tree, vmap
from jax.lax import associative_scan, scan

from cuthbert.inference import Filter
from cuthbertlib.types import ArrayTree, ArrayTreeLike, KeyArray


def filter(
    filter_obj: Filter,
    model_inputs: ArrayTreeLike,
    *,
    init_state: ArrayTreeLike | None = None,
    parallel: bool = False,
    key: KeyArray | None = None,
) -> ArrayTree:
    """Applies offline filtering given a filter object and model inputs.

    `model_inputs` should have leading temporal dimension of
    length T if `init_state` is provided, or T + 1 if `init_state` is not provided,
    where T is the number of time steps excluding the initial state.
    The output will therefore have leading temporal dimension of length T + 1.

    If `init_state` is provided, `filter_obj.init_prepare` will not be called.

    Args:
        filter_obj: The filter inference object.
        model_inputs: The model inputs (with leading temporal dimension of length T if init_state is provided, or T + 1 otherwise).
        init_state: The initial state for the filter. If not provided,
            `filter_obj.init_prepare` will be called on the first `model_inputs` to
            generate the initial state.
        parallel: Whether to run the filter in parallel.
            Requires `filter.associative_filter` to be `True`.
        key: The key for the random number generator.

    Returns:
        The filtered states (NamedTuple with leading temporal dimension of length T + 1).
        `init_state` is included as the first element of the output.
    """
    if parallel and not filter_obj.associative:
        warnings.warn(
            f"Parallel filtering attempted but filter.associative is False for {filter_obj}"
        )

    len_model_inputs = tree.leaves(model_inputs)[0].shape[0]
    T = len_model_inputs if init_state is not None else len_model_inputs - 1

    if key is None:
        # This will throw error if used as a key, which is desired behavior
        # (albeit not a useful error, we could improve this)
        prepare_keys = jnp.empty(T + 1)
    else:
        prepare_keys = random.split(key, T + 1)

    if init_state is not None:
        prep_model_inputs = model_inputs
    else:
        init_model_input = tree.map(lambda x: x[0], model_inputs)
        init_state = filter_obj.init_prepare(init_model_input, key=prepare_keys[0])

        prep_model_inputs = tree.map(lambda x: x[1:], model_inputs)

    if parallel:
        other_prep_states = vmap(lambda inp, k: filter_obj.filter_prepare(inp, key=k))(
            prep_model_inputs, prepare_keys[1:]
        )
        prep_states = tree.map(
            lambda x, y: jnp.concatenate([x[None], y]), init_state, other_prep_states
        )
        states = associative_scan(
            vmap(filter_obj.filter_combine),
            prep_states,
        )
    else:

        def body(prev_state, prep_inp_and_k):
            prep_inp, k = prep_inp_and_k
            prep_state = filter_obj.filter_prepare(prep_inp, key=k)
            state = filter_obj.filter_combine(prev_state, prep_state)
            return state, state

        _, states = scan(
            body,
            init_state,
            (prep_model_inputs, prepare_keys[1:]),
        )
        states = tree.map(
            lambda x, y: jnp.concatenate([x[None], y]), init_state, states
        )

    return states
