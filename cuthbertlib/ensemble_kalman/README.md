# Ensemble Kalman

This sub-repository provides modular functions for ensemble Kalman methods, including the ensemble Kalman filter (EnKF) and ensemble RTS smoother (EnRTS smoother).

The core functions are:

- `predict`: Propagate ensemble members through nonlinear dynamics with additive Gaussian noise.
- `filter_update`: Update ensemble members with an observation using the EnKF update equation.
- `smoother_update`: Apply one Ensemble Rauch-Tung-Striebel smoother update.
- `gaspari_cohn`: Construct covariance tapers with the Gaspari-Cohn correlation function.
- `gaussian`: Construct smooth, non-compact covariance tapers with the Gaussian correlation function.

Together, `predict` and `filter_update` can be used to perform an online EnKF filtering step.

The EnKF uses an ensemble of particles with a Kalman-style measurement update based on empirical covariances. Unlike the EKF, it does not require Jacobians, while naturally handling nonlinear dynamics.

The EnRTS algorithm provides a smoothing counterpart to the EnKF, based on the RTS smoother. It makes an empirical approximation of the RTS smoother gain, which is applied to the ensemble in a backwards pass to obtain a smoothing distribution. Note that, based on the outputs of the EnKF, the EnRTS step is entirely deterministic.

## Covariance localization

`filter_update` optionally accepts separate `cross_covariance_modifier` and `marginal_covariance_modifier` functions. Each function receives the corresponding empirical covariance matrix and the current model inputs, and returns the modified covariance matrix. These functions are applied before missingness is handled, so they can be applied in the original indexing.

The modifier functions may be arbitrary JAX-compatible code returning a PSD matrix. The most common form of this is covariance tapering; cuthbertlib provides `gaspari_cohn` and `gaussian` correlation functions for use as tapering functions. The Gaspari-Cohn correlation function is more classical, while the Gaussian has non-compact support and may therefore have better gradient properties.

When only `cross_covariance_modifier` is provided (i.e., when `marginal_covariance_modifier` is `None`), the normal square-root update is applied. When `marginal_covariance_modifier` is supplied, we fall back to an explicit computation of a Cholesky factor. 