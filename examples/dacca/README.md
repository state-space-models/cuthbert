# IF2 and the Dacca model

[This example](run_if2.py) fits the Dacca cholera model with [iterated filtering (IF2)](../../cuthbert/smc/iterated_filtering.py). IF2 adds parameters to each state particle and resamples both together.

## Install and run

```bash
git clone --branch if2-dacca-independent --single-branch https://github.com/kunyanghe21/cuthbert.git
cd cuthbert
python -m venv .venv
source .venv/bin/activate
python -m pip install ./pkg/cuthbertlib ./pkg/cuthbert
JAX_PLATFORMS=cpu JAX_ENABLE_X64=false python examples/dacca/run_if2.py
```

By default, the example runs two IF2 iterations with 32 particles on the first 12 months of data. It then evaluates the unperturbed model at the fitted parameters with two independent particle filter runs.

Use `--output dacca-output` to save the parameter population, history, likelihoods and random keys in a new directory. Use `--help` for the other options.

## Model and data

The [model](model.py) follows the Gaussian-noise Dacca example in [PyPOMP at `17f8798`](https://github.com/pypomp/pypomp/blob/17f8798e7d944977335c81b8525353fa90e14be9/pypomp/models/dacca.py). It uses 20 Euler steps per month. The 600 monthly observations and the covariates come from [PyPOMP's Dacca data directory](https://github.com/pypomp/pypomp/tree/17f8798e7d944977335c81b8525353fa90e14be9/pypomp/data/dacca). See [the PyPOMP license](LICENSE.PyPOMP).

The full Dacca results are in [`pypomp/comparisons`](https://github.com/pypomp/comparisons).
