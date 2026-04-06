# Ensemble Kalman Filter (EnKF)

This sub-repository provides modular functions for the Ensemble Kalman Filter.

The core functions are:

- `predict`: Propagate ensemble members through nonlinear dynamics with additive Gaussian noise.
- `update`: Update ensemble members with an observation using the EnKF update equation.

Together, `predict` and `update` can be used to perform an online EnKF filtering step.

The EnKF uses an ensemble of particles with a Kalman-style measurement update based on
empirical covariances. Unlike the EKF, it does not require Jacobians, while naturally
handling nonlinear dynamics. 