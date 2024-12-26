# Resampling

This sub-repository provides a unified interface for a variety of resampling methods
(converting a weighted sample into an unweighted one which likely contains duplicates).

A typical call to the library would be:
```python
particles = jax.random.normal(key, (100, 2))
weights = jax.vmap(lambda x: jnp.all(x > 0))(particles)

resampled_indices = resampling.stratified_resample(particles, weights, 100)
resampled_particles = particles[resampled_indices]
```
