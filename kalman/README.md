# Kalman


This sub-repository provides modular functions for Kalman filtering and smoothing.


The core functions are:

- `offline_filter`: Perform offline Kalman filtering.
- `smoother`: Perform Rauch-Tung-Striebel smoothing.
- `predict`: Perform an online prediction step.
- `update`: Perform an online update step.

Together, `predict` and `update` can be used to perform an online filtering step.

In all cases, we operate on the **square-root form of the covariance matrix**, which is
more numerically stable as the outputs are always guaranteed to be positive-definite.
This means **we also require input covariance matrices to be provided in square-root
(Cholesky) form**.

The offline `offline_filter` and `smoother` functions both support temporal
parallelization via the `parallel` argument which defaults to `True`.


