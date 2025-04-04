# Resampling

This sub-repository provides a unified interface for a variety of resampling methods
(converting a weighted sample into an unweighted one which likely contains duplicates).

A typical call to the library would be:
```python
sampling_key, resampling_key = jax.random.split(jax.random.PRNGKey(0))
particles = jax.random.normal(sampling_key, (100, 2))
logits = jax.vmap(lambda x: jnp.where(jnp.all(x > 0), 0, -jnp.inf))(particles)

resampled_indices = resampling.multinomial.resampling(resampling_key, logits, 100)
resampled_particles = particles[resampled_indices]
```

Or for conditional resampling:
```python
# Here we resample but keep particle at index 0 fixed
conditional_resampled_indices = resampling.multinomial.conditional_resampling(
    resampling_key, logits, 100, pivot_in=0, pivot_out=0
)
conditional_resampled_particles = particles[conditional_resampled_indices]

```