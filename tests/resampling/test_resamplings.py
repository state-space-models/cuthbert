import itertools

import chex
import jax
from absl.testing import parameterized

from resampling import systematic, killing, multinomial
from tests.resampling.utils import resampling_tester, conditional_resampling_tester

Ms = [5, 10]
Ns = [5, 10]
product_MN = list(itertools.product(Ms, Ns))
zip_MN = list(zip(Ms, Ns))

resampling_test_cases = [
    {
        "method": "multinomial",
        "MN": "product",
    },
    {
        "method": "systematic",
        "MN": "product",
    },
    {
        "method": "killing",
        "MN": "zip",
    },
]

conditional_resampling_test_cases = [
    {
        "method": "multinomial",
    },
    {
        "method": "systematic",
    },
    {
        "method": "killing",
    },
]


def get_resampling(name):
    if name == "multinomial":
        return multinomial.resampling
    elif name == "systematic":
        return systematic.resampling
    elif name == "killing":
        return killing.resampling
    else:
        raise ValueError(f"Unknown resampling method: {name}")


def get_conditional_resampling(name):
    if name == "multinomial":
        return multinomial.conditional_resampling
    elif name == "systematic":
        return systematic.conditional_resampling
    elif name == "killing":
        return killing.conditional_resampling
    else:
        raise ValueError(f"Unknown resampling method: {name}")


class TestResamplings(chex.TestCase):
    def setUp(self):
        super().setUp()
        self.K = 1_000

    @chex.all_variants(with_pmap=False, without_jit=False)
    @parameterized.parameters(itertools.product([0, 42], resampling_test_cases))
    def test_resampling(self, seed, test_case):
        key = jax.random.key(seed)
        key_weights, key_test = jax.random.split(key, 2)

        if test_case["MN"] == "product":
            MNs = product_MN
        else:
            MNs = zip_MN

        method = get_resampling(test_case["method"])

        for M, N in MNs:
            resampling = self.variant(lambda k_, lw_: method(k_, lw_, M))
            log_weights = jax.random.uniform(key_weights, (N,))
            resampling_tester(key_test, log_weights, resampling, M, self.K)

    @chex.all_variants(with_pmap=False, without_jit=False)
    @parameterized.parameters(
        itertools.product([0, 42], conditional_resampling_test_cases)
    )
    def test_conditional_resampling(self, seed, test_case):
        key = jax.random.key(seed)
        key_weights, key_test = jax.random.split(key, 2)

        conditional_method = get_conditional_resampling(test_case["method"])
        for M in Ms:
            conditional_resampling = self.variant(
                lambda k_, lw_, pivot_in, pivot_out: conditional_method(
                    k_, lw_, M, pivot_in, pivot_out
                )
            )

            log_weights = jax.random.uniform(key_weights, (M,))
            conditional_resampling_tester(
                key_test, log_weights, conditional_resampling, M, self.K
            )
