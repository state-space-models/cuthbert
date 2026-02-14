r"""Linearized Taylor Kalman filter.

Uses automatic differentiation to extract conditionally Gaussian parameters from log
densities of the dynamics and observation distributions.

This differs from `gaussian/moments`, which requires `mean` and `chol_cov`
functions as input rather than log densities.

I.e., we approximate conditional densities as

$$
p(y \mid x) \approx N(y \mid H x + d, L L^T),
$$

and potentials as

$$
G(x) \approx N(x \mid m, L L^T),
$$

where $L$ is the Cholesky factor of the covariance matrix.

See `cuthbertlib.linearize` for more details.

Parallelism via `associative_scan` is supported, but requires the `state` argument
to be ignored in `get_dynamics_log_density` and `get_observation_func`.
I.e. the linearization points are pre-defined or extracted from model inputs.
"""

from functools import partial

from cuthbert.gaussian.taylor import associative_filter, non_associative_filter
from cuthbert.gaussian.taylor.types import (
    GetDynamicsLogDensity,
    GetInitLogDensity,
    GetObservationFunc,
)
from cuthbert.inference import Filter


def build_filter(
    get_init_log_density: GetInitLogDensity,
    get_dynamics_log_density: GetDynamicsLogDensity,
    get_observation_func: GetObservationFunc,
    associative: bool = False,
    rtol: float | None = None,
    ignore_nan_dims: bool = False,
) -> Filter:
    """Build linearized Taylor Kalman inference filter.

    If `associative` is True all filtering linearization points are pre-defined or
    extracted from model inputs. The `state` argument should be ignored in
    `get_dynamics_log_density` and `get_observation_func`.

    If `associative` is False the linearization points can be extracted from the
    previous filter state for dynamics parameters and the predict state for
    observation parameters.

    Args:
        get_init_log_density: Function to get log density log p(x_0)
            and linearization point.
            Only takes `model_inputs` as input.
        get_dynamics_log_density: Function to get dynamics log density log p(x_t+1 | x_t)
            and linearization points (for the previous and current time points)
            If `associative` is True, the `state` argument should be ignored.
        get_observation_func: Function to get observation function (either conditional
            log density or log potential), linearization point and optional observation
            (not required for log potential functions).
            If `associative` is True, the `state` argument should be ignored.
        associative: If True, then the filter is suitable for associative scan, but
            assumes that the `state` is ignored in `get_dynamics_log_density` and
            `get_observation_func`.
            If False, then the filter is suitable for non-associative scan, but
            the user is free to use the `state` to extract the linearization points.
        rtol: The relative tolerance for the singular values of precision matrices
            when passed to `symmetric_inv_sqrt` during linearization.
            Cutoff for small singular values; singular values smaller than
            `rtol * largest_singular_value` are treated as zero.
            The default is determined based on the floating point precision of the dtype.
            See https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.pinv.html.
        ignore_nan_dims: Whether to treat dimensions with NaN on the diagonal of the
            precision matrices (found via linearization) as missing and ignore all rows
            and columns associated with them.

    Returns:
        Linearized Taylor Kalman filter object.
    """
    if associative:
        return Filter(
            init_prepare=partial(
                associative_filter.init_prepare,
                get_init_log_density=get_init_log_density,
                rtol=rtol,
                ignore_nan_dims=ignore_nan_dims,
            ),
            filter_prepare=partial(
                associative_filter.filter_prepare,
                get_init_log_density=get_init_log_density,
                get_dynamics_log_density=get_dynamics_log_density,
                get_observation_func=get_observation_func,
                rtol=rtol,
                ignore_nan_dims=ignore_nan_dims,
            ),
            filter_combine=associative_filter.filter_combine,
            associative=True,
        )
    else:
        return Filter(
            init_prepare=partial(
                non_associative_filter.init_prepare,
                get_init_log_density=get_init_log_density,
                rtol=rtol,
                ignore_nan_dims=ignore_nan_dims,
            ),
            filter_prepare=partial(
                non_associative_filter.filter_prepare,
                get_init_log_density=get_init_log_density,
            ),
            filter_combine=partial(
                non_associative_filter.filter_combine,
                get_dynamics_log_density=get_dynamics_log_density,
                get_observation_func=get_observation_func,
                rtol=rtol,
                ignore_nan_dims=ignore_nan_dims,
            ),
            associative=False,
        )
