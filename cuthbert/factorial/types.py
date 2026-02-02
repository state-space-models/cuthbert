"""Provides types for factorial state-space models."""

from typing import Protocol

from cuthbertlib.types import ArrayTree, ArrayTreeLike


class ExtractAndJoin(Protocol):
    """Protocol for extracting and joining the relevant factors."""

    def __call__(
        self, factorial_state: ArrayTree, model_inputs: ArrayTreeLike
    ) -> ArrayTree:
        """Extract factors from factorial state and combine into a joint local state.

        E.g. state might encode factorial `means` with shape (F, d) and `chol_covs`
        with shape (F, d, d). Then `model_inputs` tells us factors `i` and `j` are
        relevant, so we extract `means[i]` and `means[j]` and `chol_covs[i]` and
        `chol_covs[j]`. Then combine them into `joint_mean` with shape (2 * d,)
        and block diagonal `joint_chol_cov` with shape (2 * d, 2 * d).

        Args:
            factorial_state: Factorial state with factorial index as the first dimension.
            model_inputs: Model inputs including information required to determine
                the relevant factors (e.g. factor indices).

        Returns:
            Joint local state with no factorial index dimension.
        """
        ...


class MarginalizeAndInsert(Protocol):
    """Protocol for marginalizing and inserting the updated factors."""

    def __call__(
        self,
        factorial_state: ArrayTree,
        local_state: ArrayTree,
        model_inputs: ArrayTreeLike,
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
            model_inputs: Model inputs including information required to determine
                the relevant factors (e.g. factor indices).

        Returns:
            Factorial state with factorial index as the first dimension.
            The updated factors are inserted into the factorial state.
            The remaining factors are left unchanged.
        """
        ...
