import warnings

from jax import numpy as jnp
from jax import random, tree, vmap
from jax.lax import associative_scan, scan

from cuthbert.inference import Inference
from cuthbertlib.kalman.utils import append_tree
from cuthbertlib.types import ArrayTree, ArrayTreeLike, KeyArray


def smoother(
    inference: Inference,
    filter_states: ArrayTreeLike,
    model_inputs: ArrayTreeLike,
    parallel: bool = False,
    key: KeyArray | None = None,
) -> ArrayTree:
    """
    Applies offline smoothing given a inference object, output from filter, and model
    inputs (both with leading temporal dimension of len T + 1, where T is the number of
    time steps excluding the initial state).

    Args:
        inference: The inference object.
        filter_states: The filtered states (with leading temporal dimension of len T + 1).
        model_inputs: The model inputs (with leading temporal dimension of len T + 1).
        parallel: Whether to run the smoother in parallel.
            Requires inference.associative_smoother to be True.
        key: The key for the random number generator.

    Returns:
        The smoothed states (NamedTuple with leading temporal dimension of len T + 1).
    """
    if parallel and not inference.associative_smoother:
        warnings.warn(
            "Parallel smoothing attempted but inference.associative_smoother is False "
            f"for {inference}"
        )

    T = tree.leaves(model_inputs)[0].shape[0] - 1

    if key is None:
        # This will throw error if used as a key, which is desired
        # (albeit not a useful error, we could improve this)
        prepare_keys = jnp.empty(T)
    else:
        prepare_keys = random.split(key, T)

    final_filter_state = tree.map(lambda x: x[-1], filter_states)
    other_filter_states = tree.map(lambda x: x[:-1], filter_states)

    final_smoother_state = inference.convert_filter_to_smoother_state(
        final_filter_state
    )

    # Model inputs for dynamics distribution from t to t+1 is stored
    # in the (t+1)th model_inputs i.e. model_inputs[t] thus we need model_inputs[1:]
    other_model_inputs = tree.map(lambda x: x[1:], model_inputs)

    if parallel:
        prep_states = vmap(
            lambda fs, inp, k: inference.smoother_prepare(fs, inp, key=k)
        )(other_filter_states, other_model_inputs, prepare_keys)
        prep_states = append_tree(prep_states, final_smoother_state)

        states = associative_scan(
            vmap(lambda current, next: inference.smoother_combine(next, current)),
            # TODO: Maybe change cuthbertlib direction so that this lambda isn't needed
            prep_states,
            reverse=True,
        )
    else:

        def body(next_state, filt_state_and_prep_inp_and_k):
            filt_state, prep_inp, k = filt_state_and_prep_inp_and_k
            prep_state = inference.smoother_prepare(filt_state, prep_inp, key=k)
            state = inference.smoother_combine(prep_state, next_state)
            return state, state

        _, states = scan(
            body,
            final_smoother_state,
            (other_filter_states, other_model_inputs, prepare_keys),
            reverse=True,
        )

        states = append_tree(states, final_smoother_state)

    return states
