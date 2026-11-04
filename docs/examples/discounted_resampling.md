# Discounting particle-filter gradients

The `discounted_stop_gradient_decorator` adds the weight-derivative discount
from [DMOP-alpha (Tan et al.)](https://arxiv.org/abs/2407.03085v2) to an existing
resampler. With the same random key, the forward particle filter is unchanged;
`alpha` changes the derivative used for parameter estimation.

This example uses a reparameterizable bootstrap particle filter with systematic
resampling at every observation. Guided proposals and discrete transition
scores are outside the scope of this example.

## Model and filter

We use a scalar Gaussian state-space model. Only the autoregressive parameter
is differentiated. The original `particle_filter.build_filter` and
`cuthbert.filter` run the filter; no optimizer is needed.

```{.python #discounted_resampling-imports}
"""Compare bootstrap-filter gradients at several fixed discounts."""

import jax
import jax.numpy as jnp
import numpy as np

from cuthbert import filter as run_filter
from cuthbert.smc import particle_filter
from cuthbertlib import resampling

```

```{.python #discounted_resampling-model}
observations = jnp.array([0.3, -0.5, 0.7, 1.0, 0.2])
init_key, filter_key = jax.random.split(jax.random.key(13))


def log_likelihood(phi, resampler):
    """Run a bootstrap filter with the specified resampler."""

    def initial_sampler(key):
        return jax.random.normal(key)

    def transition(key, state, model_inputs):
        return phi * state + 0.5 * jax.random.normal(key)

    def log_potential(state_prev, state, model_inputs):
        return jax.scipy.stats.norm.logpdf(model_inputs, state, 0.7)

    pf = particle_filter.build_filter(
        initial_sampler, transition, log_potential, 32, resampler
    )
    states = run_filter(pf, observations, pf.init_prepare(key=init_key), key=filter_key)
    return states.log_normalizing_constant[-1]


```

## Compare derivatives

`alpha` is a required static float, fixed when the decorator is constructed.
There is no default. We use 0.97 here because it was selected for Dacca; other
models may need a different value. At zero, only the pathwise derivatives
through the particles remain. At one, the helper calls the existing
`stop_gradient_decorator`. An intermediate value discounts inherited
weight derivatives.

```{.python #discounted_resampling-compare}
phi = jnp.array(0.4)
baseline_value = log_likelihood(phi, resampling.systematic.resampling)
for alpha in (0.0, 0.97, 1.0):
    resampler = resampling.discounted_stop_gradient_decorator(
        resampling.systematic.resampling, alpha=alpha
    )
    value, gradient = jax.jit(
        jax.value_and_grad(lambda p: log_likelihood(p, resampler))
    )(phi)
    np.testing.assert_allclose(value, baseline_value, rtol=1e-6, atol=1e-6)
    assert np.isfinite(gradient)
    print(
        f"alpha={alpha:.2f}: log likelihood={float(value):.6f}, gradient={float(gradient):.6f}"
    )
```

The likelihood values agree while the gradient estimates can differ. The choice
of discount depends on the bias-variance tradeoff: alpha=0 can have high bias,
while alpha=1 uses the original stop-gradient estimator and can have high
variance. A single draw cannot establish which discount is best for optimization.

## Adaptive resampling

With adaptive resampling, the discount could be applied only when resampling
occurs. It would then count resampling events instead of observations. This
extension could be studied empirically, but the current DMOP-alpha theory does
not cover it and we have not tested it here. The example resamples at every
observation. The Cuthbert developers can decide whether to add support for
adaptive resampling.

## Further reading

See the [resampling API](../api_cuthbertlib/resampling.md) for the function
details and [differentiable particle filtering](diff_pf_resampling.md) for the
existing undiscounted approach, including adaptive resampling.

<!-- entangled code
```{.python file=examples_scripts/discounted_resampling.py}
<<discounted_resampling-imports>>
<<discounted_resampling-model>>
<<discounted_resampling-compare>>
```
-->
