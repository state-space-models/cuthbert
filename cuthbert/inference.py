from typing import NamedTuple, Protocol

from cuthbertlib.types import ArrayTree, ArrayTreeLike, KeyArray


class InitPrepare(Protocol):
    def __call__(
        self,
        model_inputs: ArrayTreeLike,
        key: KeyArray | None = None,
    ) -> ArrayTree:
        """Prepare the initial state for the inference.

        The state at the first time point, prior to any observations.

        Args:
            model_inputs: The model inputs at the first time point.
            key: The key for the random number generator.
                Optional, as only used for stochastic inference methods

        Returns:
            The initial state, a NamedTuple with inference-specific fields.
        """
        ...


class FilterPrepare(Protocol):
    def __call__(
        self,
        model_inputs: ArrayTreeLike,
        key: KeyArray | None = None,
    ) -> ArrayTree:
        """Prepare the state for the filter at the next time point.

        Converts the model inputs (and any stochasticity) into a unified state
        object which can be combined with a state (of the same form) from the
        previous time point with FilterCombine.

        state = FilterCombine(prev_state, FilterPrepare(model_inputs, key))

        Args:
            model_inputs: The model inputs at the next time point.
            key: The key for the random number generator.
                Optional, as only used for stochastic inference methods

        Returns:
            The state prepared for FilterCombine,
                a NamedTuple with inference-specific fields.
        """
        ...


class FilterCombine(Protocol):
    def __call__(
        self,
        state_1: ArrayTreeLike,
        state_2: ArrayTreeLike,
    ) -> ArrayTree:
        """Combine state from previous time point with state from FilterPrepare.

        state = FilterCombine(prev_state, FilterPrepare(model_inputs, key))

        Args:
            state_1: The state from the previous time point.
            state_2: The state from FilterPrepare for the current time point.

        Returns:
            The combined filter state, a NamedTuple with inference-specific fields.
        """
        ...


class SmootherPrepare(Protocol):
    def __call__(
        self,
        filter_state: ArrayTreeLike,
        model_inputs: ArrayTreeLike,
        key: KeyArray | None = None,
    ) -> ArrayTree:
        """Prepare the state for the smoother at the next time point.

        Converts the model inputs (and any stochasticity) into a unified state
        object which can be combined with a state (of the same form) from the
        next time point with SmootherCombine.

        Remember smoothing iterates backwards in time.

        state = SmootherCombine(
            SmootherPrepare(filter_state, model_inputs, key), next_smoother_state
        )

        Args:
            filter_state: The state from the filter at the previous time point.
            model_inputs: The model inputs at the next time point.
            key: The key for the random number generator.
                Optional, as only used for stochastic inference methods

        Returns:
            The state prepared for SmootherCombine,
                a NamedTuple with inference-specific fields.
        """
        ...


class SmootherCombine(Protocol):
    def __call__(
        self,
        state_1: ArrayTreeLike,
        state_2: ArrayTreeLike,
    ) -> ArrayTree:
        """Combine state from next time point with state from SmootherPrepare.

        Remember smoothing iterates backwards in time.

        state = SmootherCombine(
            SmootherPrepare(filter_state, model_inputs, key), next_smoother_state
        )

        Args:
            state_1: The state from SmootherPrepare for the current time point.
            state_2: The state from the next time point.

        Returns:
            The combined smoother state, a NamedTuple with inference-specific fields.
        """
        ...


class ConvertFilterToSmootherState(Protocol):
    def __call__(
        self,
        filter_state: ArrayTreeLike,
    ) -> ArrayTree:
        """Convert the filter state to a smoother state.

        Useful for offline smoothing where the final filter state is equivalent
        statistically to the final smoother state.
        This function converts the filter state to the smoother state data structure.

        Args:
            filter_state: The filter state.

        Returns:
            The smoother state.
        """
        ...


class Inference(NamedTuple):
    init_prepare: InitPrepare
    filter_prepare: FilterPrepare
    filter_combine: FilterCombine
    smoother_prepare: SmootherPrepare
    smoother_combine: SmootherCombine
    convert_filter_to_smoother_state: ConvertFilterToSmootherState
    associative_filter: bool = False
    associative_smoother: bool = False
