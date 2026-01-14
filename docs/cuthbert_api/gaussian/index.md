# Gaussian Filters and Smoothers

- [Exact Kalman](kalman.md) - exact inference in linear Gaussian state-space models.
- [Moments](moments.md) - approximate Gaussian inference in state-space models, with
specified conditional mean and (Cholesky) covariance functions.
- [Taylor](taylor.md) - approximate Gaussian inference via linearization of log
densities.

Note that the moments and Taylor methods can be considered variants of the extended
Kalman filter.

The core atomic functions can be found in [`cuthbertlib.kalman`](../../cuthbertlib_api/kalman.md).

## Gaussian Filters and Smoothers in `cuthbert`

{%
    include-markdown "../../../cuthbert/gaussian/README.md"
    start="<!--gaussian-filters-and-smoothers-start-->"
    end="<!--gaussian-filters-and-smoothers-end-->"
%}
