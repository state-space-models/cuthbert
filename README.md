<!--intro-start-->
<div align="center">
<img src="https://raw.githubusercontent.com/state-space-models/cuthbert/main/docs/assets/cuthbert.png" alt="cuthbert logo"></img>
</div>

A JAX library for state-space model inference
(filtering, smoothing, static parameter estimation).

> Disclaimer: The name `cuthbert` was chosen as a playful nod to the well-known
> caterpillar cake rivalry between Aldi and M&S in the UK, as the classic state-space
> model diagram looks vaguely like a caterpillar. However, this software project
> has no formal connection to Aldi, M&S, or any food products (notwithstanding the coffee drunk during its writeup).
> `cuthbert` is simply a fun name for this state-space model library and should not be interpreted as an
> endorsement, association, or affiliation with any brand or animal themed baked goods.

[![Discord](https://img.shields.io/badge/Discord-5865F2?logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/W9BA4Aj7Rx)
[![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white&style=for-the-badge)](https://github.com/state-space-models/cuthbert)
[![PyPI](https://img.shields.io/pypi/v/cuthbert?style=for-the-badge)](https://pypi.org/project/cuthbert/)
[![Docs](https://img.shields.io/badge/Docs-b6d7a8?logo=materialformkdocs&logoColor=black&style=for-the-badge)](https://state-space-models.github.io/cuthbert/)
<!--intro-end-->

<!--goals-start-->
### Goals
- Simple, flexible and performant interface for state-space model inference.
- Decoupling of model specification and inference. `cuthbert` is built to swap between
different **inference** methods without be tied to a specific model specification.
- Compose with the [JAX ecosystem](#ecosystem) for extensive external tools.
- Functional API: The only classes in `cuthbert` are `NamedTuple`s and `Protocol`s.
All functions are pure and work seamlessly with `jax.grad`, `jax.jit`, `jax.vmap` etc.
- Methods for filtering: $p(x_t \mid y_{1:t}, \theta)$.
- Methods for smoothing: $p(x_{0:T} \mid y_{1:T}, \theta)$ or $p(x_{t} \mid y_{1:T}, \theta)$.
- Methods for static parameter estimation: $p(\theta \mid y_{1:T})$
or $\text{argmax} p(y_{1:T} \mid \theta)$.
- This includes support for forward-backward/Baum-Welch, particle filtering/sequential Monte Carlo,
Kalman filtering (+ extended/unscented/ensemble), expectation-maximization and more!

### Non-goals
- Tools for defining models and distributions. `cuthbert` is not a probabilistic programming language (PPL).
But can easily compose with [`dynamax`](https://github.com/probml/dynamax), [`distrax`](https://github.com/google-deepmind/distrax), [`numpyro`](https://github.com/pyro-ppl/numpyro) and [`pymc`](https://github.com/pymc-devs/pymc) in a similar way to how [`blackjax` does](https://blackjax-devs.github.io/blackjax/).
- ["SMC Samplers"](https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf) which sample from a posterior
distribution which is not (necessarily) a state-space model - [`blackjax` is great for this](https://github.com/blackjax-devs/blackjax/tree/main/blackjax/smc).
<!--goals-end-->

<!--codebase-structure-start-->
### Codebase structure

The codebase is structured as follows:

- `cuthbert`: The main package with unified interface for filtering and smoothing.
- `cuthbertlib`: A collection of atomic, smaller-scoped tools useful for state-space model inference,
that represent the building blocks that power the main `cuthbert` package.
<!--codebase-structure-end-->
- `docs`: Source code for the documentation for `cuthbert` and `cuthbertlib`.
- `pkg`: Packaging configuration for publishing `cuthbert` and `cuthbertlib` to PyPI.
- `tests`: Tests for the `cuthbert` and `cuthbertlib` packages.


<!--installation-start-->
## Installation

`cuthbert` depends on JAX, so you'll need to [install JAX](https://docs.jax.dev/en/latest/installation.html) for the available hardware (CPU, GPU, or TPU).
For example, on computers with NVIDIA GPUs:

```bash
pip install -U "jax[cuda13]"
```

### From PyPI

```bash
pip install -U cuthbert
```

Installing `cuthbert` will also install `cuthbertlib`.
You can also install `cuthbertlib` on its own:

```bash
pip install -U cuthbertlib
```

### Local development (uv)

```bash
git clone https://github.com/state-space-models/cuthbert.git
cd cuthbert
uv sync --package cuthbert --extra tests
```

### Local development (pip)

```bash
git clone https://github.com/state-space-models/cuthbert.git
cd cuthbert
pip install -e ./pkg/cuthbertlib
pip install -e "./pkg/cuthbert[tests]"
```

<!--installation-end-->

<!--ecosystem-start-->
## Ecosystem
- `cuthbert` is built on top of [`jax`](https://github.com/google/jax) and composes
easily with other JAX packages, e.g. [`optax`](https://github.com/google-deepmind/optax)
for optimization, [`flax`](https://github.com/google/flax) for neural networks, and
[`blackjax`](https://github.com/blackjax-devs/blackjax) for (SG)MCMC as well as the PPLs
mentioned [above](#non-goals).
- What about [`dynamax`](https://github.com/probml/dynamax)?
    - `dynamax` is a great library for state-space model specification and inference with
    discrete or Gaussian state-space models. `cuthbert` is focused on inference
    with arbitrary state-space models via  e.g. SMC that is not supported in `dynamax`.
    However as they are both built on [`jax`](https://github.com/google/jax)
    they can be used together! A `dynamax`
    model can be passed to `cuthbert` for inference.
- And [`particles`](https://github.com/nchopin/particles)?
    - [`particles`](https://github.com/nchopin/particles) and the accompanying book
    [Sequential Monte Carlo Methods in Practice](https://link.springer.com/book/10.1007/978-3-030-47845-2)
    are wonderful learning materials for state-space models and SMC.
    `cuthbert` is more focused on performance and composability with the JAX ecosystem.
- Much of the code in `cuthbert` is built on work from [`sqrt-parallel-smoothers`](https://github.com/EEA-sensors/sqrt-parallel-smoothers), [`mocat`](https://github.com/SamDuffield/mocat) and [`abile`](https://github.com/SamDuffield/abile).
<!--ecosystem-end-->

## Contributing

We're always looking for contributions!
Check out the [contributing guide](CONTRIBUTING.md) for more information.

