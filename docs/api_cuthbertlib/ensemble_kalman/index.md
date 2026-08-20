# Ensemble Kalman

Ensemble Kalman methods are a class of sequential Monte Carlo methods for filtering and smoothing in state-space models. They are particularly useful for high-dimensional systems where traditional Kalman filters are computationally infeasible. This submodule implements atomic functions for ensemble Kalman filters and smoothers, as well as utilities of covariance localization.

- [Ensemble Kalman filtering](filtering.md)
- [Ensemble Rauch-Tung-Striebel smoothing](smoothing.md)
- [Covariance localization](localization.md)

The high-level interfaces built on these functions are in
[`cuthbert.ensemble_kalman`](../../api_cuthbert/ensemble_kalman/index.md).
