# Sequential Monte Carlo Methods

- [Particle filter](particle_filter.md) - flexible particle filter for general Feynman-Kac models.
- [Marginal particle filter](marginal_particle_filter.md) - $O(N^2)$ variant of the particle filter.
- [Backward sampler](backward_sampler.md) - flexible backward smoothing using approaches in [`cuthbertlib.smc.smoothing`](../../cuthbertlib_api/smc.md).

The core atomic functions can be found in [`cuthbertlib.smc`](../../cuthbertlib_api/smc.md).

## Feynman-Kac Models

{%
    include-markdown "../../../cuthbert/smc/README.md"
    start="<!--feynman-kac-start-->"
    end="<!--feynman-kac-end-->"
%}
