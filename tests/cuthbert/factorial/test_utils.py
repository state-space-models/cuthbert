import chex
import jax.numpy as jnp
from jax import tree

from cuthbert.factorial.utils import serial_to_factorial, serial_to_single_factor


def extract(factorial_state, factorial_inds):
    return tree.map(lambda x: x[factorial_inds], factorial_state)


def test_serial_to_factorial_groups_values_by_index_in_order():
    serial_tree = {
        "x": jnp.array(
            [
                [[10.0], [11.0]],
                [[20.0], [21.0]],
                [[30.0], [31.0]],
            ]
        ),
        "y": jnp.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
            ]
        ),
    }
    factorial_inds = jnp.array([[0, 2], [1, 0], [2, 1]])

    # Build expected trees from serial traversal order in one pass.
    expected_trees = [
        {
            "x": jnp.zeros((0,) + serial_tree["x"].shape[2:]),
            "y": jnp.zeros((0,) + serial_tree["y"].shape[2:]),
        }
        for _ in range(3)
    ]
    for t, inds in enumerate(factorial_inds):
        for j, ind in enumerate(inds):
            expected_trees[ind]["x"] = jnp.concatenate(
                [expected_trees[ind]["x"], serial_tree["x"][t, j][None]],
            )
            expected_trees[ind]["y"] = jnp.concatenate(
                [expected_trees[ind]["y"], serial_tree["y"][t, j][None]],
            )

    factorial_trees = serial_to_factorial(extract, serial_tree, factorial_inds)

    assert len(factorial_trees) == 3
    for factor, actual_tree in enumerate(factorial_trees):
        chex.assert_trees_all_close(actual_tree, expected_trees[factor])


def test_serial_to_single_factor_matches_corresponding_factorial_tree():
    serial_tree = jnp.array(
        [
            [[1.0], [2.0]],
            [[3.0], [4.0]],
        ]
    )
    factorial_inds = jnp.array([[1, 0], [0, 1]])

    all_factors = serial_to_factorial(extract, serial_tree, factorial_inds)
    factor_1 = serial_to_single_factor(
        extract, serial_tree, factorial_inds, factorial_index=1
    )

    chex.assert_trees_all_close(factor_1, all_factors[1])
