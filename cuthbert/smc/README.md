# Sequential Monte Carlo Methods

The core atomic functions can be found in [`cuthbertlib.smc`](../../cuthbertlib/smc).

<!--feynman-kac-start-->
Sequential Monte Carlo methods provide particle approximations to general Feynman-Kac models:

$$
\mathbb{Q}_{t}(x_{0:t}) \propto \mathbb{M}_0(x_0) \, G_0(x_0) \prod_{s=1}^{t} M_s(x_s \mid x_{s-1}) \, G_s(x_{s-1}, x_s), \quad t \in \{ 0, \dots, T\},
$$

where $M_{s}$ are probability kernels and $G_{s}$ are positive and bounded functions.
<!--feynman-kac-end-->
