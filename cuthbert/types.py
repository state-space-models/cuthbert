from typing import Any
from jax import Array
from jax.typing import ArrayLike

KeyArray = Array  # No native JAX type annotation for keys https://jax.readthedocs.io/en/latest/changelog.html#jax-0-4-16-sept-18-2023
ArrayTree = Any  # No native JAX type annotation for PyTrees https://github.com/google/jax/issues/3340
ArrayTreeLike = Any  # Tree with all leaves castable to jax.Array https://jax.readthedocs.io/en/latest/jax.typing.html#module-jax.typing
ScalarArray = Array  # jax.Array with just a single float element, i.e. shape ()
ScalarArrayLike = ArrayLike  # Object that will be cast to a FloatArray
