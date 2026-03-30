# Sequential Monte Carlo Methods

- [Particle filter](particle_filter.md) - flexible particle filter for general Feynman-Kac models.
- [Marginal particle filter](marginal_particle_filter.md) - $O(N^2)$ variant of the particle filter.
- [Backward sampler](backward_sampler.md) - flexible backward smoothing using approaches in [`cuthbertlib.smc.smoothing`](../../api_cuthbertlib/smc.md).

The core atomic functions can be found in [`cuthbertlib.smc`](../../api_cuthbertlib/smc.md).

## Feynman-Kac Models

--8<-- "cuthbert/smc/README.md:feynman_kac"
