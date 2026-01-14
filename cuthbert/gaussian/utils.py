from cuthbert.gaussian.types import LinearizedKalmanFilterState
from cuthbert.utils import dummy_tree_like
from cuthbertlib.kalman import filtering
from cuthbertlib.types import Array, ArrayTree


def linearized_kalman_filter_state_dummy_elem(
    mean: Array,
    chol_cov: Array,
    log_normalizing_constant: Array,
    model_inputs: ArrayTree,
    mean_prev: Array,
) -> LinearizedKalmanFilterState:
    """Create a LinearizedKalmanFilterState with a dummy element
    I.e. when associated scan is not used.

    Args:
        mean: Mean of the state.
        chol_cov: Cholesky covariance of the state.
        log_normalizing_constant: Log normalizing constant of the state.
        model_inputs: Model inputs.
        mean_prev: Mean of the previous state.

    Returns:
        LinearizedKalmanFilterState with a dummy elem attribute.
    """
    return LinearizedKalmanFilterState(
        elem=filtering.FilterScanElement(
            A=dummy_tree_like(chol_cov),
            b=mean,
            U=chol_cov,
            eta=dummy_tree_like(mean),
            Z=dummy_tree_like(chol_cov),
            ell=log_normalizing_constant,
        ),
        model_inputs=model_inputs,
        mean_prev=mean_prev,
    )
