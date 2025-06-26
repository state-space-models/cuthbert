# SMC

This sub-repository provides modular tools useful for constructing sequential Monte
Carlo (SMC) algorithms. It pairs with the `cuthbertlib.resampling` sub-repository.

### Backward simulation

`cuthbertlib.smc.backward` provides tools for backward simulation, i.e. converting
particles `x0` from the filter distribution at time t0 and particles `x1` from the smoothing
distribution at time t1 into joint particles from the smoothing distribution `(x0, x1)`.
Backward simulation requires a log conditional density `log_density(x0, x1)` and
has computational cost `O(N^2)` where `N` is the number of particles.