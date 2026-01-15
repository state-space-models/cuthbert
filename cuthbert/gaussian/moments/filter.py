r"""Linearized moments Kalman filter.

Takes a user provided conditional `mean` and `chol_cov` functions to define a
conditionally linear Gaussian state space model.

I.e., we approximate conditional densities as

$$
p(y \mid x) \approx N(y \mid \mathrm{mean}(x), \mathrm{chol\_cov}(x) @ \mathrm{chol\_cov}(x)^\top).
$$

See `cuthbertlib.linearize` for more details.

Parallelism via `associative_scan` is supported, but requires the `state` argument
to be ignored in `get_dynamics_params` and `get_observation_params`.
I.e. the linearization points are pre-defined or extracted from model inputs.
"""

from functools import partial

from cuthbert.gaussian.moments import associative_filter, non_associative_filter
from cuthbert.gaussian.moments.types import GetDynamicsMoments, GetObservationMoments
from cuthbert.gaussian.types import GetInitParams
from cuthbert.inference import Filter


def build_filter(
    get_init_params: GetInitParams,
    get_dynamics_params: GetDynamicsMoments,
    get_observation_params: GetObservationMoments,
    associative: bool = False,
) -> Filter:
    """Build linearized moments Kalman inference filter.

    If `associative` is True all filtering linearization points are pre-defined or
    extracted from model inputs. The `state` argument should be ignored in
    `get_dynamics_params` and `get_observation_params`.

    If `associative` is False the linearization points can be extracted from the
    previous filter state for dynamics parameters and the predict state for
    observation parameters.

    Args:
        get_init_params: Function to get m0, chol_P0 from model inputs.
        get_dynamics_params: Function to get dynamics conditional mean and
            (generalised) Cholesky covariance from linearization point and model inputs.
            and linearization points (for the previous and current time points)
            If `associative` is True, the `state` argument should be ignored.
        get_observation_params: Function to get observation conditional mean,
            (generalised) Cholesky covariance and observation from linearization point
            and model inputs.
            If `associative` is True, the `state` argument should be ignored.
        associative: If True, then the filter is suitable for associative scan, but
            assumes that the `state` is ignored in `get_dynamics_params` and
            `get_observation_params`.
            If False, then the filter is suitable for non-associative scan, but
            the user is free to use the `state` to extract the linearization points.

    Returns:
        Linearized moments Kalman filter object.
    """
    if associative:
        return Filter(
            init_prepare=partial(
                associative_filter.init_prepare,
                get_init_params=get_init_params,
                get_observation_params=get_observation_params,
            ),
            filter_prepare=partial(
                associative_filter.filter_prepare,
                get_init_params=get_init_params,
                get_dynamics_params=get_dynamics_params,
                get_observation_params=get_observation_params,
            ),
            filter_combine=associative_filter.filter_combine,
            associative=True,
        )
    else:
        return Filter(
            init_prepare=partial(
                non_associative_filter.init_prepare,
                get_init_params=get_init_params,
                get_observation_params=get_observation_params,
            ),
            filter_prepare=partial(
                non_associative_filter.filter_prepare,
                get_init_params=get_init_params,
            ),
            filter_combine=partial(
                non_associative_filter.filter_combine,
                get_dynamics_params=get_dynamics_params,
                get_observation_params=get_observation_params,
            ),
            associative=False,
        )
