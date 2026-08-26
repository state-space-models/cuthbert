# Ensemble Kalman

The high-level interfaces built on these functions are in
[`cuthbert.ensemble_kalman.ensemble_kalman_filter`](../../cuthbert/ensemble_kalman/ensemble_kalman_filter.py)
and
[`cuthbert.ensemble_kalman.ensemble_rts_smoother`](../../cuthbert/ensemble_kalman/ensemble_rts_smoother.py).

<!-- --8<-- [start:overview] -->
This sub-repository provides modular functions for ensemble Kalman methods, including the ensemble Kalman filter (EnKF) and ensemble RTS smoother (EnRTS smoother).

The core functions are:

- `predict`: Propagate ensemble members through nonlinear dynamics with additive Gaussian noise.
- `filter_update`: Update ensemble members with an observation using the EnKF update equation.
- `smoother_update`: Apply one Ensemble Rauch-Tung-Striebel smoother update.
- `construct_tapered_chol_innovation_covariance`: Construct a tapered innovation covariance factor without forming the covariance densely.
- `gaspari_cohn`: Construct covariance tapers with the Gaspari-Cohn correlation function.
- `gaussian`: Construct smooth, non-compact covariance tapers with the Gaussian correlation function.
<!-- --8<-- [end:overview] -->

## Ensemble Kalman filtering

<!-- --8<-- [start:filtering] -->
Together, `predict` and `filter_update` can be used to perform an online EnKF filtering step.

The EnKF uses an ensemble of particles with a Kalman-style measurement update based on empirical covariances. Unlike the EKF, it does not require Jacobians, while naturally handling nonlinear dynamics.
<!-- --8<-- [end:filtering] -->

## Ensemble Rauch-Tung-Striebel smoothing

<!-- --8<-- [start:smoothing] -->
The EnRTS algorithm provides a smoothing counterpart to the EnKF, based on the RTS smoother. It makes an empirical approximation of the RTS smoother gain, which is applied to the ensemble in a backwards pass to obtain a smoothing distribution. Note that, based on the outputs of the EnKF, the EnRTS step is entirely deterministic.
<!-- --8<-- [end:smoothing] -->

## Covariance localization

<!-- --8<-- [start:localization] -->
`filter_update` accepts two independent low-level hooks. `cross_covariance_modifier` receives one empirical state-observation cross-covariance and returns its modified value; it defaults to `no_covariance_modifier`. For observation-space covariance localization, one may optionally use `construct_chol_innovation_covariance`, which receives normalized observation deviations `Y` and a Cholesky factor of the observation noise covariance, `chol_R`, and returns a generalized Cholesky factor of the localized innovation covariance $C_{yy} + R$. All localization happens before handling of missing data, so localization happens in the original coordinates.

The most common form of localization is covariance tapering; cuthbertlib provides `gaspari_cohn` and `gaussian` correlation functions for this purpose. The Gaspari-Cohn correlation function is more classical, but has compact support which may cause difficulties in optimizing localization hyperparameters. The `gaussian` correlation function has infinite support, and may therefore have better gradient properties. For obsservation-space tapering, `cuthbertlib` provides a convenience `construct_tapered_chol_innovation_covariance`, which uses a Cholesky factor of the taper to construct a factor of the tapered innovation covariance. This comes at slightly higher cost, due to a larger QR solve.

<!-- --8<-- [end:localization] -->
