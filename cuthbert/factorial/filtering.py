"""cuthbert factorial filtering interface."""

from jax import numpy as jnp
from jax import random, tree, vmap
from jax.lax import associative_scan, scan

from cuthbert.inference import Filter
from cuthbert.factorial.types import ExtractAndJoin, MarginalizeAndInsert
from cuthbertlib.types import ArrayTree, ArrayTreeLike, KeyArray


def filter(
    filter_obj: Filter,
    extract_and_join: ExtractAndJoin,
    marginalize_and_insert: MarginalizeAndInsert,
    model_inputs: ArrayTreeLike,
    key: KeyArray | None = None,
) -> ArrayTree:
    """Applies offline factorial filtering for given model inputs.

    `model_inputs` should have leading temporal dimension of length T + 1,
    where T is the number of time steps excluding the initial state.

    Parallel associative filtering is not supported for factorial filtering.

    Note that this function will output a factorial state with first temporal dimension
    of length T + 1 and second factorial dimension of length F. Many of the factors
    will be unchanged across timesteps where they aren't relevant. So some memory
    can be saved with more sophisticated data structures although this is left to the
    user for maximum flexibility (and jax.lax.scan can be hard to work with varliable
    sized arrays).

    Args:
        filter_obj: The filter inference object.
        extract_and_join: Function to extract and join the relevant factors into
            a single joint state.
        marginalize_and_insert: Function to marginalize and insert the updated factors
            back into the factorial state.
        model_inputs: The model inputs (with leading temporal dimension of length T + 1).
        key: The key for the random number generator.

    Returns:
        The filtered states (NamedTuple with leading temporal dimension of length T + 1).
    """
    T = tree.leaves(model_inputs)[0].shape[0] - 1

    if key is None:
        # This will throw error if used as a key, which is desired behavior
        # (albeit not a useful error, we could improve this)
        prepare_keys = jnp.empty(T + 1)
    else:
        prepare_keys = random.split(key, T + 1)

    init_model_input = tree.map(lambda x: x[0], model_inputs)
    init_factorial_state = filter_obj.init_prepare(
        init_model_input, key=prepare_keys[0]
    )

    prep_model_inputs = tree.map(lambda x: x[1:], model_inputs)

    def body(prev_factorial_state, prep_inp_and_k):
        prep_inp, k = prep_inp_and_k
        local_state = extract_and_join(prev_factorial_state, prep_inp)
        prep_state = filter_obj.filter_prepare(prep_inp, key=k)
        filtered_joint_state = filter_obj.filter_combine(local_state, prep_state)
        factorial_state = marginalize_and_insert(
            prev_factorial_state, filtered_joint_state, prep_inp
        )
        return factorial_state, factorial_state

    _, factorial_states = scan(
        body,
        init_factorial_state,
        (prep_model_inputs, prepare_keys[1:]),
    )
    factorial_states = tree.map(
        lambda x, y: jnp.concatenate([x[None], y]),
        init_factorial_state,
        factorial_states,
    )

    return factorial_states
