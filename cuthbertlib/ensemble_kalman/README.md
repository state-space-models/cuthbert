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

`filter_update` optionally accepts `CovarianceTapers` to perform localization. Passing a custom covariance taper is possible, but cuthbertlib also provides `gaspari_cohn` and `gaussian` to construct tapers from predefined correlation functions. The Gaspari-Cohn correlation function is more classical, but the Gaussian has non-compact support and may therefore have better gradient properties.
