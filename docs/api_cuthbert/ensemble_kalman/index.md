# Ensemble Kalman

Ensemble Kalman methods are a class of sequential Monte Carlo methods for filtering and smoothing in state-space models. They are particularly useful for high-dimensional systems where traditional Kalman filters are computationally infeasible. The core idea is to represent the state distribution with an ensemble of particles, which are propagated through the system dynamics and updated based on Gaussian approximations fitted to the ensemble. Unlike
sequential Monte Carlo methods, ensemble Kalman methods do not use importance weights.

- [Ensemble Kalman Filtering](ensemble_kalman_filter.md)
- [Ensemble Rauch-Tung-Striebel smoothing](ensemble_rts_smoother.md)
- [Callback types](types.md)
