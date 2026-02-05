"""cuthbert factorial filtering interface."""

from jax import numpy as jnp
from jax import random, tree
from jax.lax import scan

from cuthbert.factorial.types import Factorializer
from cuthbert.inference import Filter
from cuthbertlib.types import ArrayTree, ArrayTreeLike, KeyArray


def filter(
    filter_obj: Filter,
    factorializer: Factorializer,
    model_inputs: ArrayTreeLike,
    output_factorial: bool = False,
    key: KeyArray | None = None,
) -> (
    ArrayTree | tuple[ArrayTree, ArrayTree]
):  # TODO: Can overload this function so the type checker knows that the output is a ArrayTree if output_factorial is True and a tuple[ArrayTree, ArrayTree] if output_factorial is False
    """Applies offline factorial filtering for given model inputs.

    `model_inputs` should have leading temporal dimension of length T + 1,
    where T is the number of time steps excluding the initial state.

    Parallel associative filtering is not supported for factorial filtering.

    Note that if output_factorial is True, this function will output a factorial state
    with first temporal dimension of length T + 1 and second factorial dimension of
    length F. Many of the factors will be unchanged across timesteps where they aren't
    relevant.

    Args:
        filter_obj: The filter inference object.
        factorializer: The factorializer object for the inference method.
        model_inputs: The model inputs (with leading temporal dimension of length T + 1).
        output_factorial: If True, return a single state with first temporal dimension
            of length T + 1 and second factorial dimension of length F.
            If False, return a tuple of states. The first being the initial state
            with first dimension of length F and temporal dimension.
            The second being the local states for each time step, i.e. first
            dimension of length T and no factorial dimension.
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

    def body_local(prev_factorial_state, prep_inp_and_k):
        prep_inp, k = prep_inp_and_k
        factorial_inds = factorializer.get_factorial_indices(prep_inp)
        local_state = factorializer.extract_and_join(
            prev_factorial_state, factorial_inds
        )
        prep_state = filter_obj.filter_prepare(prep_inp, key=k)
        filtered_joint_state = filter_obj.filter_combine(local_state, prep_state)
        factorial_state = factorializer.marginalize_and_insert(
            filtered_joint_state, prev_factorial_state, factorial_inds
        )

        def extract(arr):
            if arr.ndim >= 2:
                return arr[factorial_inds]
            else:
                return arr

        factorial_state_fac_inds = tree.map(extract, factorial_state)
        return factorial_state, factorial_state_fac_inds

    if output_factorial:

        def body_factorial(prev_factorial_state, prep_inp_and_k):
            factorial_state, _ = body_local(prev_factorial_state, prep_inp_and_k)
            return factorial_state, factorial_state

        _, factorial_states = scan(
            body_factorial,
            init_factorial_state,
            (prep_model_inputs, prepare_keys[1:]),
        )
        factorial_states = tree.map(
            lambda x, y: jnp.concatenate([x[None], y]),
            init_factorial_state,
            factorial_states,
        )

        return factorial_states

    else:
        _, local_states = scan(
            body_local,
            init_factorial_state,
            (prep_model_inputs, prepare_keys[1:]),
        )
        return init_factorial_state, local_states
