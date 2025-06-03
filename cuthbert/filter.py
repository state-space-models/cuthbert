from jax import vmap, tree, random, numpy as jnp
from jax.lax import scan, associative_scan

from cuthbert.inference import SSMInference
from cuthbertlib.types import ArrayTreeLike, KeyArray, ArrayTree


def filter_update(
    inference: SSMInference,
    state: ArrayTreeLike,
    model_inputs: ArrayTreeLike,
    key: KeyArray | None = None,
) -> ArrayTree:
    return inference.FilterCombine(state, inference.FilterPrepare(model_inputs, key))


def filter(
    inference: SSMInference,
    model_inputs: ArrayTreeLike,
    parallel: bool = False,
    key: KeyArray | None = None,
) -> ArrayTree:
    if parallel and not inference.associative_filter:
        raise ValueError(
            f"Parallel filtering attempted but inference.associative_filter is False for {inference}"
        )

    T = tree.leaves(model_inputs)[0].shape[0]

    if key is None:
        # This will throw error if used as a key, which is desired
        # (albeit not a useful error, we could improve this)
        prepare_keys = jnp.empty(T)
    else:
        prepare_keys = random.split(key, T)

    prep_states = vmap(lambda inp, k: inference.FilterPrepare(inp, key=key))(
        model_inputs, prepare_keys
    )

    if parallel:
        states = associative_scan(
            vmap(inference.FilterCombine, in_axes=(0, 0)),
            prep_states,
        )
    else:
        init_state = tree.map(lambda x: x[0], prep_states)
        other_states = tree.map(lambda x: x[1:], prep_states)

        def body(prev_state, prep_state):
            state = inference.FilterCombine(prev_state, prep_state)
            return state, state

        _, states = scan(
            body,
            init_state,
            other_states,
        )

        states = tree.map(
            lambda i, ss: jnp.concatenate([i[None, ...], ss], axis=0),
            init_state,
            states,
        )

    return states
