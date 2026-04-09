"""Type aliases for common types used in the library."""

from typing import Any, Callable, TypeAlias

from jax import Array
from jax.typing import ArrayLike

KeyArray: TypeAlias = Array  # No native JAX type annotation for keys https://jax.readthedocs.io/en/latest/changelog.html#jax-0-4-16-sept-18-2023
ArrayTree: TypeAlias = Any  # No native JAX type annotation for PyTrees https://github.com/google/jax/issues/3340
ArrayTreeLike: TypeAlias = Any  # Tree with all leaves castable to jax.Array https://jax.readthedocs.io/en/latest/jax.typing.html#module-jax.typing
ScalarArray: TypeAlias = (
    Array  # jax.Array with just a single float element, i.e. shape ()
)
ScalarArrayLike: TypeAlias = ArrayLike  # Object that will be cast to a ScalarArray

LogDensity: TypeAlias = Callable[[ArrayTreeLike], ScalarArray]
LogConditionalDensity: TypeAlias = Callable[
    [ArrayTreeLike, ArrayTreeLike], ScalarArray
]  # p(x_1 | x_0), where x_0 is the first argument and x_1 the second.
LogConditionalDensityAux: TypeAlias = Callable[
    [ArrayTreeLike, ArrayTreeLike], tuple[ScalarArray, ArrayTree]
]
