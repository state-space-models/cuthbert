# Sequential Monte Carlo Methods

Sequential Monte Carlo methods provide particle approximations to general Feynman-Kac models:

$$
\mathbb{Q}_{t}(x_{0:t}) \propto \mathbb{M}_0(x_0) \, G_0(x_0) \prod_{s=1}^{t} M_s(x_s \mid x_{s-1}) \, G_s(x_{s-1}, x_s), \quad t \in \{ 0, \dots, T\},
$$

where $M_{s}$ are probability kernels and $G_{s}$ are positive and bounded functions.

See [Types](types.md) for how to define a Feynman-Kac model in code.

- [Particle filter](particle_filter.md) - flexible particle filter for general Feynman-Kac models.
- [Marginal particle filter](marginal_particle_filter.md) - $O(N^2)$ variant of the particle filter.
- [Backward sampler](backward_sampler.md) - flexible backward smoothing using approaches in [`cuthbertlib.smc.smoothing`](../../cuthbertlib_api/smc.md).

