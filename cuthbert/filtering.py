from jax import vmap, tree, random, numpy as jnp
from jax.lax import scan, associative_scan
import warnings

from cuthbert.inference import SSMInference
from cuthbertlib.types import ArrayTreeLike, KeyArray, ArrayTree
from cuthbertlib.kalman.utils import append_tree


def filter(
    inference: SSMInference,
    model_inputs: ArrayTreeLike,
    parallel: bool = False,
    key: KeyArray | None = None,
) -> ArrayTree:
    """
    Applies offlines filtering given a inference object and model inputs
    (with leading temporal dimension of len T + 1, where T is the number of time steps
    excluding the initial state).

    Args:
        inference: The inference object.
        model_inputs: The model inputs (with leading temporal dimension of len T + 1).
        parallel: Whether to run the filter in parallel.
            Requires inference.associative_filter to be True.
        key: The key for the random number generator.

    Returns:
        The filtered states (NamedTuple with leading temporal dimension of len T + 1).
    """

    if parallel and not inference.associative_filter:
        warnings.warn(
            "Parallel filtering attempted but inference.associative_filter is False "
            f"for {inference}"
        )

    T = tree.leaves(model_inputs)[0].shape[0] - 1

    if key is None:
        # This will throw error if used as a key, which is desired behavior
        # (albeit not a useful error, we could improve this)
        prepare_keys = jnp.empty(T + 1)
    else:
        prepare_keys = random.split(key, T + 1)

    init_model_input = tree.map(lambda x: x[0], model_inputs)
    init_state = inference.init_prepare(init_model_input, key=prepare_keys[0])

    prep_model_inputs = tree.map(lambda x: x[1:], model_inputs)

    if parallel:
        other_prep_states = vmap(lambda inp, k: inference.filter_prepare(inp, key=k))(
            prep_model_inputs, prepare_keys[1:]
        )
        prep_states = append_tree(other_prep_states, init_state, prepend=True)
        states = associative_scan(
            vmap(inference.filter_combine),
            prep_states,
        )
    else:

        def body(prev_state, prep_inp_and_k):
            prep_inp, k = prep_inp_and_k
            prep_state = inference.filter_prepare(prep_inp, key=k)
            state = inference.filter_combine(prev_state, prep_state)
            return state, state

        _, states = scan(
            body,
            init_state,
            (prep_model_inputs, prepare_keys[1:]),
        )
        states = append_tree(states, init_state, prepend=True)

    return states
