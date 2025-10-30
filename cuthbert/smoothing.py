import warnings

from jax import numpy as jnp
from jax import random, tree, vmap
from jax.lax import associative_scan, scan

from cuthbert.inference import Smoother
from cuthbert.utils import dummy_tree_like
from cuthbertlib.types import ArrayTree, ArrayTreeLike, KeyArray


def smoother(
    smoother_obj: Smoother,
    filter_states: ArrayTreeLike,
    model_inputs: ArrayTreeLike | None = None,
    parallel: bool = False,
    key: KeyArray | None = None,
) -> ArrayTree:
    """Applies offline smoothing given a smoother object, output from filter, and model inputs.

    `filter_states` should have leading temporal dimension of length T + 1, where
    T is the number of time steps excluding the initial state.

    Each element of `model_inputs` refers to the transition from t to t+1, except for the
    first element which refers to the initial state. The initial state `model_inputs`
    are not used for smoothing. Thus the `model_inputs` used here have length T.
    By default, `filter_states.model_inputs[1:]` are used (i.e. the `model_inputs`
    used for the initial state is ignored).

    Args:
        smoother_obj: The smoother inference object.
        filter_states: The filtered states (with leading temporal dimension of length T + 1).
        model_inputs: The model inputs (with leading temporal dimension of length T).
            Optional, if None then `filter_states.model_inputs[1:]` are used.
        parallel: Whether to run the smoother in parallel.
            Requires `smoother_obj.associative_smoother` to be `True`.
        key: The key for the random number generator.

    Returns:
        The smoothed states (NamedTuple with leading temporal dimension of length T + 1).
    """
    if parallel and not smoother_obj.associative:
        warnings.warn(
            "Parallel smoothing attempted but smoother.associative is False "
            f"for {smoother}"
        )

    if model_inputs is None:
        model_inputs = filter_states.model_inputs

    T = tree.leaves(filter_states)[0].shape[0] - 1

    # model_inputs for the dynamics distribution from t-1 to t is stored
    # in model_inputs[t] thus we need model_inputs[1:]
    # model_inputs[0] is only used for init_prepare and not for smoothing.
    # Therefore, we allow model_inputs to be either of length T + 1 or T
    # where if length is T + 1 then we simply discard model_inputs[0]
    model_inputs_length = tree.leaves(model_inputs)[0].shape[0]
    if model_inputs_length == T + 1:
        model_inputs = tree.map(lambda x: x[1:], model_inputs)
    elif model_inputs_length != T:
        raise ValueError(
            "model_inputs must have length T + 1 or T, got length "
            f"{model_inputs_length}"
        )

    if key is None:
        # This will throw error if used as a key, which is desired
        # (albeit not a useful error, we could improve this)
        prepare_keys = jnp.empty(T + 1)
    else:
        prepare_keys = random.split(key, T + 1)

    final_filter_state = tree.map(lambda x: x[-1], filter_states)
    other_filter_states = tree.map(lambda x: x[:-1], filter_states)

    # Final smoother state doesn't need model inputs, so we create a dummy one
    # with the same structure as model_inputs but with all values set to dummy values.
    dummy_single_model_inputs = dummy_tree_like(tree.map(lambda x: x[0], model_inputs))

    final_smoother_state = smoother_obj.convert_filter_to_smoother_state(
        final_filter_state, model_inputs=dummy_single_model_inputs, key=prepare_keys[0]
    )

    if parallel:
        prep_states = vmap(
            lambda fs, inp, k: smoother_obj.smoother_prepare(
                fs, model_inputs=inp, key=k
            )
        )(other_filter_states, model_inputs, prepare_keys[1:])
        prep_states = tree.map(
            lambda x, y: jnp.concatenate([x, y[None]]),
            prep_states,
            final_smoother_state,
        )

        states = associative_scan(
            vmap(lambda current, next: smoother_obj.smoother_combine(next, current)),
            # TODO: Maybe change cuthbertlib direction so that this lambda isn't needed
            prep_states,
            reverse=True,
        )
    else:

        def body(next_state, filt_state_and_prep_inp_and_k):
            filt_state, prep_inp, k = filt_state_and_prep_inp_and_k
            prep_state = smoother_obj.smoother_prepare(
                filt_state, model_inputs=prep_inp, key=k
            )
            state = smoother_obj.smoother_combine(prep_state, next_state)
            return state, state

        _, states = scan(
            body,
            final_smoother_state,
            (other_filter_states, model_inputs, prepare_keys[1:]),
            reverse=True,
        )

        states = tree.map(
            lambda x, y: jnp.concatenate([x, y[None]]),
            states,
            final_smoother_state,
        )

    return states
