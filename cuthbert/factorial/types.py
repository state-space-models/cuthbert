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


class Extract(Protocol):
    """Protocol for extracting the relevant factors."""

    def __call__(
        self,
        factorial_state: ArrayTreeLike,
        factorial_inds: ArrayLike,
    ) -> ArrayTree:
        """Extract factors from factorial state.

        E.g. factorial_state might encode factorial `means` with shape (F, d) and
        `chol_covs` with shape (F, d, d). Then `model_inputs` tells us factors `i` and
        `j` are relevant, so we extract `means[i]` and `means[j]` and `chol_covs[i]` and
        `chol_covs[j]`. Thus we return `means` with shape (2, d) and `chol_covs` with
        shape (2, d, d).

        Args:
            factorial_state: Factorial state with factorial index as the first dimension.
            factorial_inds: Indices of the factors to extract. Integer array.

        Returns:
            Local factorial state with factorial dimension of length len(factorial_inds).
        """
        ...


class Join(Protocol):
    """Protocol for combining factorial states into a joint state."""

    def __call__(
        self,
        local_factorial_state: ArrayTreeLike,
    ) -> ArrayTree:
        """Extract factors from factorial state and combine into a joint local state.

        E.g. local_factorial_state might encode factorial `means` with shape (2, d) and
        `chol_covs` with shape (2, d, d).
        Which is then combined into a joint state with shape (2 * d,)
        and block diagonal `joint_chol_cov` with shape (2 * d, 2 * d).

        Args:
            local_factorial_state: Factorial state with factorial index as the first
                dimension. Typically contains only a small number of factors, as it's
                applied after an `Extract` operation.

        Returns:
            Joint state with no factorial index dimension.
        """
        ...


class Marginalize(Protocol):
    """Protocol for marginalizing a joint state into a factored state."""

    def __call__(
        self,
        local_state: ArrayTree,
        num_factors: int,
    ) -> ArrayTree:
        """Marginalize joint state into factored state.

        E.g. `local_state` might have shape (2 * d,) and `joint_chol_cov`
        with shape (2 * d, 2 * d). Then we marginalize out the joint local state into
        two factorial `means` with shape (2, d) and `chol_covs` with shape (2, d, d).

        Args:
            local_state: Joint local state with no factorial index dimension.
            num_factors: Number of factors to marginalize out. Integer.
                This is typically equal to len(factorial_inds).

        Returns:
            Factorial state with factorial index as the first dimension and
                `num_factors` factors (length of first dimension).
        """
        ...


class Insert(Protocol):
    """Protocol for inserting a local factorial state into a factorial state."""

    def __call__(
        self,
        local_factorial_state: ArrayTree,
        factorial_state: ArrayTree,
        factorial_inds: ArrayLike,
    ) -> ArrayTree:
        """Marginalize joint state into factored state and insert into factorial state.

        E.g. `local_factorial_state` might have shape (2, d) and `joint_chol_cov`
        with shape (2, d, d). Then we insert `means[0]` and `means[1]` into
        `state[i]` and `state[j]` respectively. Similarly, we insert `chol_covs[0]` and
        `chol_covs[1]`. In both cases, we overwrite the existing factors in the
        factorial state for `i` and `j`, leaving the other factors unchanged.
        Here `i` and `j` are determined from `factorial_inds`.

        Args:
            local_factorial_state: Local factorial state with factorial index as the first
                dimension and `len(factorial_inds)` factors (length of first dimension).
            factorial_state: Factorial state with factorial index as the first dimension.
            factorial_inds: Indices of the factors to insert. Integer array.

        Returns:
            Factorial state with factorial index as the first dimension.
                The updated factors are inserted into the factorial state.
                The remaining factors are left unchanged.
        """
        ...


class Factorializer(NamedTuple):
    """Factorializer object.

    All functions are inference method dependent (e.g. Gaussian/SMC etc),
    aside from the `get_factorial_indices` function which acts purely on `model_inputs`.

    Attributes:
        get_factorial_indices: Function to extract factorial indices from model inputs.
        extract: Function to extract the relevant factors.
        join: Function to combine factorial states into a joint state.
        marginalize: Function to marginalize a joint state into a factored state.
        insert: Function to insert a local factorial state into a factorial state.
    """

    get_factorial_indices: GetFactorialIndices
    extract: Extract
    join: Join
    marginalize: Marginalize
    insert: Insert
