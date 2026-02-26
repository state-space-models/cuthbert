# Resampling

This sub-repository provides a unified interface for a variety of resampling
methods, which convert a set of weighted samples into an unweighted one which
likely contains duplicates.

A typical call to the library would be:

```python
sampling_key, resampling_key = jax.random.split(jax.random.key(0))
particles = jax.random.normal(sampling_key, (100, 2))
logits = jax.vmap(lambda x: jnp.where(jnp.all(x > 0), 0, -jnp.inf))(particles)

resampled_indices, _, resampled_particles = resampling.multinomial.resampling(resampling_key, logits, particles, 100)
```

Or for conditional resampling:

```python
# Here we resample but keep particle at index 0 fixed
conditional_resampled_indices, _, conditional_resampled_particles = resampling.multinomial.conditional_resampling(
    resampling_key, logits, particles, 100, pivot_in=0, pivot_out=0
)
```

Adaptive resampling (i.e. resampling only when the effective sample size is below a
threshold) is also supported via a decorator:

```python
adaptive_resampling = resampling.adaptive.ess_decorator(
    resampling.multinomial.resampling,
    threshold=0.5,
)
adaptive_resampled_indices, _, adaptive_resampled_particles = adaptive_resampling(
    resampling_key, logits, particles, 100
)
```