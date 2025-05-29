import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array
from typing import Sequence, NamedTuple

from cuthbertlib.kalman import filtering, smoothing, sampling
from cuthbertlib.kalman.utils import append_tree
from cuthbertlib.types import ScalarArray


class KalmanState(NamedTuple):
    """Gaussian state with mean and square root of covariance.

    Attributes:
        mean: Mean of the Gaussian state.
        chol_cov: Generalized Cholesky factor of the covariance of the Gaussian state.
    """

    mean: Array
    chol_cov: Array


class KalmanFilterInfo(NamedTuple):
    """Additional output from the Kalman filter.

    Attributes:
        log_likelihood: Log marginal likelihoods.
    """

    log_likelihood: ScalarArray


class KalmanSmootherInfo(NamedTuple):
    """Additional output from the Kalman smoother.

    Attributes:
        gains: Smoothing Kalman gain matrices.
    """

    gains: Array


def filter(
    m0: ArrayLike,
    chol_P0: ArrayLike,
    F: ArrayLike,
    c: ArrayLike,
    chol_Q: ArrayLike,
    H: ArrayLike,
    d: ArrayLike,
    chol_R: ArrayLike,
    y: ArrayLike,
    parallel: bool = True,
) -> tuple[KalmanState, KalmanFilterInfo]:
    """The square root Kalman filter.

    The square root Kalman filter is more numerically stable than the standard implementation that
    uses full covariance matrices, especially when using single-precision floating point numbers.
    It also ensures that covariance matrices remain positive semi-definite.

    Matrices and vectors that define the transition and observation models for
    every time step, along with the observations, must be batched along the first axis.

    chol_P0, chol_Q and chol_R must be generalized Cholesky factors. A generalized Cholesky factor
    of a positive semi-definite matrix A is a lower triangular matrix L such that L @ L.T = A.

    Args:
        m0: Mean of the initial state.
        chol_P0: Generalized Cholesky factor of the covariance of the initial state.
        F: State transition matrices.
        c: State transition shift vectors.
        chol_Q: Generalized Cholesky factors of the transition noise covariance.
        H: Observation matrices.
        d: Observation shift vectors.
        chol_R: Generalized Cholesky factors of the observation noise covariance.
        y: Observations.
        parallel: Whether to use temporal parallelization.

    Returns:
        A tuple of the filtered states at every time step and the log marginal likelihood.

    References:
        Paper: Yaghoobi, Corenflos, Hassan and Särkkä (2022) - https://arxiv.org/pdf/2207.00426
        Code: https://github.com/EEA-sensors/sqrt-parallel-smoothers/blob/main/parsmooth/parallel
    """
    m0, chol_P0 = jnp.asarray(m0), jnp.asarray(chol_P0)
    F, c, chol_Q = jnp.asarray(F), jnp.asarray(c), jnp.asarray(chol_Q)
    H, d, chol_R, y = (
        jnp.asarray(H),
        jnp.asarray(d),
        jnp.asarray(chol_R),
        jnp.asarray(y),
    )
    associative_params = filtering.sqrt_associative_params(
        m0, chol_P0, F, c, chol_Q, H, d, chol_R, y
    )

    if parallel:
        all_prefix_sums = jax.lax.associative_scan(
            jax.vmap(filtering.sqrt_filtering_operator), associative_params
        )
    else:
        init_carry = jax.tree.map(lambda x: x[0], associative_params)
        inputs = jax.tree.map(lambda x: x[1:], associative_params)

        def body(carry, inp):
            next_elem = filtering.sqrt_filtering_operator(carry, inp)
            return next_elem, next_elem

        _, all_prefix_sums = jax.lax.scan(body, init_carry, inputs)
        all_prefix_sums = jax.tree.map(
            lambda x, y: jnp.concatenate([x[None, ...], y]), init_carry, all_prefix_sums
        )

    _, filt_means, filt_chol_covs, _, _, ells = all_prefix_sums
    filt_means = jnp.vstack([m0[None, ...], filt_means])
    filt_chol_covs = jnp.vstack([chol_P0[None, ...], filt_chol_covs])
    return KalmanState(filt_means, filt_chol_covs), KalmanFilterInfo(-ells)


def smoother(
    filter_ms: ArrayLike,
    filter_chol_Ps: ArrayLike,
    Fs: ArrayLike,
    cs: ArrayLike,
    chol_Qs: ArrayLike,
    parallel: bool = True,
) -> tuple[KalmanState, KalmanSmootherInfo]:
    r"""The square root Rauch–Tung–Striebel (RTS) smoother, also known
    colloquially as the Kalman smoother.

    All ArrayLike inputs must be batched over time along the first axis.

    Args:
        filter_ms: The means of the filtered states.
        filter_chol_Ps: The generalized Cholesky factors of the covariances of the filtered states.
        Fs: State transition matrices.
        cs: State transition shift vectors.
        chol_Qs: Generalized Cholesky factors of the state transition noise covariances.
        parallel: Whether to use temporal parallelization.

    Returns:
        A tuple `(smooth_states, info)`.
        `smooth_states` contains the smoothed states for :math:`t \in \{0, \dots, T\}`
        `info` contains the smoothing gain matrices for :math:`t \in \{0, \dots, T-1\}`.

        The cross-covariance matrices :math:`Cov[x_{t}, x_{t + 1} \mid y_{1:T}]` for
        :math:`t \in \{0, \dots, T-1\}` can be computed as
        ``gains @ chol_covs[1:] @ chol_covs[1:].transpose(0, 2, 1)``.

    References:
            Paper: Yaghoobi, Corenflos, Hassan and Särkkä (2022) - https://arxiv.org/pdf/2207.00426
            Code: https://github.com/EEA-sensors/sqrt-parallel-smoothers/blob/main/parsmooth/parallel
    """
    ms, Ps = jnp.asarray(filter_ms), jnp.asarray(filter_chol_Ps)
    Fs, cs, chol_Qs = jnp.asarray(Fs), jnp.asarray(cs), jnp.asarray(chol_Qs)
    associative_params = smoothing.sqrt_associative_params(ms, Ps, Fs, cs, chol_Qs)

    if parallel:
        all_prefix_sums = jax.lax.associative_scan(
            jax.vmap(smoothing.sqrt_smoothing_operator),
            associative_params,
            reverse=True,
        )
    else:
        final_element = jax.tree.map(lambda x: x[-1], associative_params)
        inputs = jax.tree.map(lambda x: x[:-1], associative_params)

        def body(carry, inp):
            next_elem = smoothing.sqrt_smoothing_operator(carry, inp)
            return next_elem, next_elem

        _, all_prefix_sums = jax.lax.scan(body, final_element, inputs, reverse=True)
        all_prefix_sums = append_tree(all_prefix_sums, final_element)

    smoothed_means, _, smoothed_chol_covs = all_prefix_sums
    return KalmanState(smoothed_means, smoothed_chol_covs), KalmanSmootherInfo(
        associative_params.E[:-1]
    )


def sampler(
    key: ArrayLike,
    ms: Array,
    chol_Ps: Array,
    Fs: Array,
    cs: Array,
    chol_Qs: Array,
    shape: Sequence[int] = (),
    parallel: bool = True,
) -> Array:
    """Sample from the smoothing distribution of a linear-Gaussian state-space model (LGSSM).

    Args:
        key: A PRNG key.
        ms: Filtering means.
        chol_Ps: Generalized Cholesky factors of the filtering covariances.
        Fs: State transition matrices.
        cs: State transition shift vectors.
        chol_Qs: Generalized Cholesky factors of the state transition noise covariances.
        shape: The shape of the samples to draw. This represents the prefix of the
            output shape which will have an additional two axes representing the
            number of time steps and the state dimension.
        parallel: Whether to use temporal parallelization.

    Returns:
        An array of shape `shape + (num_time_steps, x_dim)` containing the samples.
    """

    associative_params = sampling.sqrt_associative_params(
        key, ms, chol_Ps, Fs, cs, chol_Qs, shape
    )
    if parallel:
        all_prefix_sums = jax.lax.associative_scan(
            jax.vmap(sampling.sampling_operator), associative_params, reverse=True
        )
    else:
        final_element = jax.tree.map(lambda x: x[-1], associative_params)
        inputs = jax.tree.map(lambda x: x[:-1], associative_params)

        def body(carry, inp):
            next_elem = sampling.sampling_operator(carry, inp)
            return next_elem, next_elem

        _, all_prefix_sums = jax.lax.scan(body, final_element, inputs, reverse=True)
        all_prefix_sums = append_tree(all_prefix_sums, final_element)

    return jnp.moveaxis(all_prefix_sums.sample, 0, -2)
