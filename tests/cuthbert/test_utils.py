import jax
import jax.numpy as jnp

from cuthbert.utils import dummy_tree_like


def test_dummy_tree_like():
    pytree = {
        "bool": True,
        "int": 3,
        "float": 1.3,
        "jax_array": jnp.ones(4),
        "nan": jnp.nan,
    }
    dummy_pytree = dummy_tree_like(pytree)

    # All leaves of the pytree should be JAX arrays
    assert all(isinstance(x, jax.Array) for x in jax.tree.leaves(dummy_pytree))

    # Check dtypes are preserved
    assert dummy_pytree["bool"].dtype == jnp.bool_
    assert dummy_pytree["int"].dtype == jnp.int32
    assert dummy_pytree["float"].dtype == jnp.float32
    assert dummy_pytree["jax_array"].dtype == jnp.float32
    assert dummy_pytree["nan"].dtype == jnp.float32

    # Check values are minimum values for their dtypes
    assert dummy_pytree["bool"].item() is False
    assert dummy_pytree["int"].item() == jnp.iinfo(jnp.int32).min
    assert dummy_pytree["float"].item() == jnp.finfo(jnp.float32).min
    assert jnp.all(dummy_pytree["jax_array"] == jnp.finfo(jnp.float32).min)
    assert dummy_pytree["nan"].item() == jnp.finfo(jnp.float32).min
