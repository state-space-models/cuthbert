import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def arraylike_to_array(fun_name: str, *args: ArrayLike) -> tuple[Array, ...]:
    """Converts ArrayLike inputs to Arrays.

    Args:
        fun_name: The name of the function which needs its arguements converted.
            Used to make error messages more informative.
        args: The arguments to convert.

    Returns:
        A tuple of the converted arguments.

    Raises:
        AssertionError: If `fun_name` is not a string.
        TypeError: If any input is not an ArrayLike.

    This is based off of the private API `jax._src.numpy.util.ensure_arraylike`,
    but is slower to compile (see https://github.com/jax-ml/jax/pull/25936 and
    linked issues).
    """
    assert isinstance(fun_name, str), f"fun_name must be a string. Got {fun_name}"
    for pos, arg in enumerate(args):
        if not isinstance(arg, ArrayLike):
            msg = f"{fun_name} requires ndarray or scalar arguments, got {type(arg)} at position {pos}."
            raise TypeError(msg)
    return tuple(jnp.asarray(arg) for arg in args)


def tria(A: Array) -> Array:
    """A triangularization operator using QR decomposition.

    Args:
        A: The matrix to triangularize.

    Returns:
        A lower triangular matrix R such that R @ R.T = A @ A.T.
    """
    _, R = jax.scipy.linalg.qr(A.T, mode="economic")
    return R.T
