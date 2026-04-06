import chex
import jax.numpy as jnp
from jax import tree

from cuthbert.factorial.utils import serial_to_factorial, serial_to_single_factor


def extract(factorial_state, factorial_inds):
    return tree.map(lambda x: x[factorial_inds], factorial_state)


def make_serial_tree():
    return {
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


def build_expected_factorial_trees(
    serial_tree, factorial_inds, init_factorial_tree=None
):
    num_factors = int(jnp.max(factorial_inds)) + 1

    if init_factorial_tree is None:
        expected_trees = [
            {
                "x": jnp.zeros((0,) + serial_tree["x"].shape[2:]),
                "y": jnp.zeros((0,) + serial_tree["y"].shape[2:]),
            }
            for _ in range(num_factors)
        ]
    else:
        expected_trees = [
            tree.map(
                lambda x: x[None],
                extract(init_factorial_tree, factor),
            )
            for factor in range(num_factors)
        ]

    for t, inds in enumerate(factorial_inds):
        for j, ind in enumerate(inds):
            expected_trees[ind] = tree.map(
                lambda current, new: jnp.concatenate([current, new[None]]),
                expected_trees[ind],
                tree.map(lambda x: x[t, j], serial_tree),
            )

    return expected_trees


def test_serial_to_factorial_groups_values_by_index_in_order():
    serial_tree = make_serial_tree()
    factorial_inds = jnp.array([[0, 2], [1, 0], [2, 1]])
    expected_trees = build_expected_factorial_trees(serial_tree, factorial_inds)

    factorial_trees = serial_to_factorial(extract, serial_tree, factorial_inds)

    assert len(factorial_trees) == 3
    for factor, actual_tree in enumerate(factorial_trees):
        chex.assert_trees_all_close(actual_tree, expected_trees[factor])


def test_serial_to_factorial_prepends_init_factorial_tree():
    serial_tree = make_serial_tree()
    factorial_inds = jnp.array([[0, 2], [1, 0], [2, 1]])
    init_factorial_tree = {
        "x": jnp.array([[1.0], [2.0], [3.0]]),
        "y": jnp.array([[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]]),
    }
    expected_trees = build_expected_factorial_trees(
        serial_tree, factorial_inds, init_factorial_tree=init_factorial_tree
    )

    factorial_trees = serial_to_factorial(
        extract,
        serial_tree,
        factorial_inds,
        init_factorial_tree=init_factorial_tree,
    )

    assert len(factorial_trees) == 3
    for factor, actual_tree in enumerate(factorial_trees):
        chex.assert_trees_all_close(actual_tree, expected_trees[factor])


def test_serial_to_single_factor_matches_corresponding_factorial_tree():
    serial_tree = make_serial_tree()
    factorial_inds = jnp.array([[0, 2], [1, 0], [2, 1]])
    factorial_index = 1

    all_factors = serial_to_factorial(extract, serial_tree, factorial_inds)
    factor_1 = serial_to_single_factor(
        extract, serial_tree, factorial_inds, factorial_index=factorial_index
    )

    chex.assert_trees_all_close(factor_1, all_factors[factorial_index])


def test_serial_to_single_factor_prepends_init_factorial_tree():
    serial_tree = make_serial_tree()
    factorial_inds = jnp.array([[0, 2], [1, 0], [2, 1]])
    factorial_index = 1
    init_factorial_tree = {
        "x": jnp.array([[1.0], [2.0], [3.0]]),
        "y": jnp.array([[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]]),
    }

    all_factors = serial_to_factorial(
        extract,
        serial_tree,
        factorial_inds,
        init_factorial_tree=init_factorial_tree,
    )
    factor_1 = serial_to_single_factor(
        extract,
        serial_tree,
        factorial_inds,
        factorial_index=factorial_index,
        init_factorial_tree=init_factorial_tree,
    )

    chex.assert_trees_all_close(factor_1, all_factors[factorial_index])
