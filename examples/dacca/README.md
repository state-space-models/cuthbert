# IF2 and the Dacca model

This branch provides an iterated filtering (IF2) implementation using Cuthbert
and an example for the Dacca cholera model.

It starts from Cuthbert's upstream `main` and is independent of the
[`alpha contribution`](https://github.com/kunyanghe21/cuthbert/tree/dmop-alpha).
IF2 uses parameter perturbations and ordinary particle filtering. This example
does not run IFAD or Adam.

## Install and run

From a terminal with Python 3.10 or newer:

```bash
git clone --branch if2-dacca-independent --single-branch https://github.com/kunyanghe21/cuthbert.git
cd cuthbert
python -m venv .venv
source .venv/bin/activate
python -m pip install ./pkg/cuthbertlib ./pkg/cuthbert
JAX_PLATFORMS=cpu JAX_ENABLE_X64=false python examples/dacca/run_if2.py
```

The default runs two IF2 iterations with 32 particles on the first 12 monthly
observations. It then evaluates the resulting parameters twice with the original
ordinary particle filter and new random keys. The output reports completion,
likelihood values and elapsed time, including compilation. This small run checks
that the example works; it does not establish convergence or scoring precision.

To save the parameter population, history, evaluation values and random keys:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=false python examples/dacca/run_if2.py --output dacca-output
```

The output directory must not already exist. `--months` accepts 1 to 600;
`--particles`, `--iterations` and `--seed` are also configurable. Increasing these
values can substantially increase runtime. The included data cover all 600
months, with 20 Gaussian Euler steps per month.

## Code

| File | Contents |
|---|---|
| [`iterated_filtering.py`](../../cuthbert/smc/iterated_filtering.py) | General IF2 implementation |
| [`_estimation.py`](../../cuthbert/smc/_estimation.py) | Parameter perturbations, validation and the loop over observations |
| [`model.py`](model.py) | Dacca initial state, transition, observation density and parameter transformations |
| [`run_if2.py`](run_if2.py) | A small IF2 run and independent ordinary-PF evaluation |
| [`test_iterated_filtering.py`](../../tests/cuthbert/smc/test_iterated_filtering.py) | IF2 reference calculations and failure tests |

For another bootstrap model, provide a function that maps parameters to the
three callbacks accepted by `particle_filter.build_filter`: an initial sampler,
a transition sampler and an observation log density. For example:

```python
import jax
from cuthbert.smc import iterated_filtering

result = iterated_filtering.fit(
    model.factory,
    model.PUBLISHED,
    model.inputs(12),
    key=jax.random.key(0),
    n_particles=32,
    n_iterations=2,
    perturbation_scale=model.PROCESS_SCALE,
    initial_perturbation_scale=model.INITIAL_SCALE,
    cooling_rate=0.9,
)
```

IF2 adds parameters to each state particle and resamples both together. It calls
Cuthbert's unchanged `particle_filter.build_filter`, `filter_prepare` and
`filter_combine`, with the original systematic or multinomial resampler. The new
loop retains only the final filter state instead of storing every state through
`cuthbert.filter`. A final resampling step gives an equally weighted parameter
population to carry into the next IF2 iteration.

Iteration `m`, starting at zero, multiplies perturbation scales by
`cooling_rate**m`. Initial-value coordinates can have a nonzero initial scale
and zero within-pass scale. If a pass produces nonfinite states, parameters or
likelihood, the code reports failure and retains the last valid population.
The returned `perturbed_log_likelihoods` are values for the perturbed models;
they are not evaluations of the static model at the returned mean parameters.

## Model and data

The model follows the Gaussian-noise Dacca example in
[PyPOMP at `17f8798`](https://github.com/pypomp/pypomp/blob/17f8798e7d944977335c81b8525353fa90e14be9/pypomp/models/dacca.py).
It preserves the nonnegative state clipping, permanent violation counter and
additive observation-density floor of `1e-18`. The model's transmission exponent
named `alpha` remains fixed at 1; it is unrelated to the resampling discount.
There are 23 estimation coordinates, including five normalized initial-state
coordinates with one redundant direction.

[`data.npz`](data.npz) is the saved input used in our Dacca comparison. It contains
observations, covariates on the Euler grid, step sizes, parameter names and
starting values. Its SHA256 is
`71446b8d04f4790bb636b17c15d51a9ce8984294a6ceceda500a3aef05bf9ba2`.
The example uses the published parameter vector; the other saved starting
vectors are unused. The source data are in
[PyPOMP's Dacca data directory](https://github.com/pypomp/pypomp/tree/17f8798e7d944977335c81b8525353fa90e14be9/pypomp/data/dacca).
The PyPOMP copyright and license are retained in [`LICENSE.PyPOMP`](LICENSE.PyPOMP).

## Tests and results

```bash
python -m pip install pytest
JAX_PLATFORMS=cpu python -m pytest tests/cuthbert/smc/test_iterated_filtering.py tests/examples/test_dacca.py
```

The tests compare parameter populations, histories and perturbed likelihoods
with a direct IF2 calculation. They also cover parameter PyTrees, initial-only
perturbations, JIT/vmap, invalid scales and failed-pass rollback.
See [the local check results](CHECKS.md).

The completed Dacca comparison is kept in
[`pypomp/comparisons`](https://github.com/pypomp/comparisons), including its
[numerical checks](https://github.com/pypomp/comparisons/blob/55a7f6f3407c8dc1d7a2f643ede6597682481292/reports/correctness.md)
and [optimization results](https://github.com/pypomp/comparisons/blob/55a7f6f3407c8dc1d7a2f643ede6597682481292/reports/optimization.md).
The example and tests here are self-contained.
The linked results use the source and settings recorded with those experiments.
This example runs only the small check described above.
