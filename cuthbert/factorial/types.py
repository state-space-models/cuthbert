"""Provides types for factorial state-space models."""

from typing import NamedTuple, Protocol

from cuthbertlib.types import ArrayLike, ArrayTree, ArrayTreeLike


class GetFactorialIndices(Protocol):
    """Protocol for getting the factorial indices."""

    def __call__(self, model_inputs: ArrayTreeLike) -> ArrayLike:
        """Extract the factorial indices from model inputs.

        Args:
            model_inputs: Model inputs.

        Returns:
            Indices of the factors to extract. Integer array.
        """
        ...


class ExtractAndJoin(Protocol):
    """Protocol for extracting and joining the relevant factors."""

    def __call__(
        self,
        factorial_state: ArrayTreeLike,
        factorial_inds: ArrayLike,
    ) -> ArrayTree:
        """Extract factors from factorial state and combine into a joint local state.

        E.g. factorial_state might encode factorial `means` with shape (F, d) and
        `chol_covs` with shape (F, d, d). Then `model_inputs` tells us factors `i` and
        `j` are relevant, so we extract `means[i]` and `means[j]` and `chol_covs[i]` and
        `chol_covs[j]`. Then combine them into `joint_mean` with shape (2 * d,)
        and block diagonal `joint_chol_cov` with shape (2 * d, 2 * d).

        Args:
            factorial_state: Factorial state with factorial index as the first dimension.
            factorial_inds: Indices of the factors to extract. Integer array.

        Returns:
            Joint local state with no factorial index dimension.
        """
        ...


class MarginalizeAndInsert(Protocol):
    """Protocol for marginalizing and inserting the updated factors."""

    def __call__(
        self,
        local_state: ArrayTree,
        factorial_state: ArrayTree,
        factorial_inds: ArrayLike,
    ) -> ArrayTree:
        """Marginalize joint state into factored state and insert into factorial state.

        E.g. `local_state` might have shape (2 * d,) and `joint_chol_cov`
        with shape (2 * d, 2 * d). Then we marginalize out the joint local state into
        two factorial `means` with shape (2, d) and `chol_covs` with shape (2, d, d).
        If `model_inputs` tells us we're working with factors `i` and `j`, then we
        insert `means[0]` and `means[1]` into `state[i]` and `state[j]` respectively.
        Similarly, we insert `chol_covs[0]` and `chol_covs[1]`. In both cases, we
        overwrite the existing factors in the factorial state for `i` and `j`,
        leaving the other factors unchanged.

        Args:
            factorial_state: Factorial state with factorial index as the first dimension.
            local_state: Joint local state with no factorial index dimension.
            factorial_inds: Indices of the factors to insert. Integer array.

        Returns:
            Factorial state with factorial index as the first dimension.
            The updated factors are inserted into the factorial state.
            The remaining factors are left unchanged.
        """
        ...


class Factorializer(NamedTuple):
    """Factorializer object.

    Attributes:
        get_factorial_indices: Function to get the factorial indices.
            Model inputs dependent.
        extract_and_join: Function to extract and join the relevant factors.
            Inference method dependent (e.g. Gaussian/SMC etc)
        marginalize_and_insert: Function to marginalize and insert the updated factors.
            Inference method dependent (e.g. Gaussian/SMC etc).
    """

    get_factorial_indices: GetFactorialIndices
    extract_and_join: ExtractAndJoin
    marginalize_and_insert: MarginalizeAndInsert
