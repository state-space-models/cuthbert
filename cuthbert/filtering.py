import warnings

from jax import numpy as jnp
from jax import random, tree, vmap
from jax.lax import associative_scan, scan

from cuthbert.inference import Filter
from cuthbertlib.kalman.utils import append_tree
from cuthbertlib.types import ArrayTree, ArrayTreeLike, KeyArray


def filter(
    filter: Filter,
    model_inputs: ArrayTreeLike,
    parallel: bool = False,
    key: KeyArray | None = None,
) -> ArrayTree:
    """
    Applies offlines filtering given a filter object and model inputs
    (with leading temporal dimension of len T + 1, where T is the number of time steps
    excluding the initial state).

    Args:
        filter: The filter inference object.
        model_inputs: The model inputs (with leading temporal dimension of len T + 1).
        parallel: Whether to run the filter in parallel.
            Requires inference.associative_filter to be True.
        key: The key for the random number generator.

    Returns:
        The filtered states (NamedTuple with leading temporal dimension of len T + 1).
    """

    if parallel and not filter.associative:
        warnings.warn(
            f"Parallel filtering attempted but filter.associative is False for {filter}"
        )

    T = tree.leaves(model_inputs)[0].shape[0] - 1

    if key is None:
        # This will throw error if used as a key, which is desired behavior
        # (albeit not a useful error, we could improve this)
        prepare_keys = jnp.empty(T + 1)
    else:
        prepare_keys = random.split(key, T + 1)

    init_model_input = tree.map(lambda x: x[0], model_inputs)
    init_state = filter.init_prepare(init_model_input, key=prepare_keys[0])

    prep_model_inputs = tree.map(lambda x: x[1:], model_inputs)

    if parallel:
        other_prep_states = vmap(lambda inp, k: filter.filter_prepare(inp, key=k))(
            prep_model_inputs, prepare_keys[1:]
        )
        prep_states = append_tree(other_prep_states, init_state, prepend=True)
        states = associative_scan(
            vmap(filter.filter_combine),
            prep_states,
        )
    else:

        def body(prev_state, prep_inp_and_k):
            prep_inp, k = prep_inp_and_k
            prep_state = filter.filter_prepare(prep_inp, key=k)
            state = filter.filter_combine(prev_state, prep_state)
            return state, state

        _, states = scan(
            body,
            init_state,
            (prep_model_inputs, prepare_keys[1:]),
        )
        states = append_tree(states, init_state, prepend=True)

    return states
