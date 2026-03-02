import itertools

import chex
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jax import random
from jax.scipy.special import logsumexp

from cuthbertlib.resampling import killing, multinomial, systematic
from cuthbertlib.resampling.stop_gradient import stop_gradient_decorator
from tests.cuthbertlib.resampling.utils import (
    conditional_resampling_tester,
    resampling_tester,
)

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
    @pytest.mark.xdist_group(name="resampling")  # Serialize to avoid OOM
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
            # create dummy positions in the wrapper; accept the positions arg from tester and ignore it
            resampling = self.variant(
                lambda k_, lw_, positions: method(
                    k_, lw_, jax.random.normal(jax.random.key(0), (N,)), M
                )
            )
            log_weights = jax.random.uniform(key_weights, (N,))
            resampling_tester(key_test, log_weights, resampling, M, self.K)

    @chex.all_variants(with_pmap=False, without_jit=False)
    @pytest.mark.xdist_group(name="resampling")  # Serialize to avoid OOM
    @parameterized.parameters(
        itertools.product([0, 42], conditional_resampling_test_cases)
    )
    def test_conditional_resampling(self, seed, test_case):
        key = jax.random.key(seed)
        key_weights, key_test = jax.random.split(key, 2)

        conditional_method = get_conditional_resampling(test_case["method"])
        for M in Ms:
            conditional_resampling = self.variant(
                # accept positions arg from tester and ignore it
                lambda k_, lw_, positions, pivot_in, pivot_out: conditional_method(
                    k_,
                    lw_,
                    jax.random.normal(jax.random.key(0), (M,)),
                    M,
                    pivot_in,
                    pivot_out,
                )
            )

            log_weights = jax.random.uniform(key_weights, (M,))
            conditional_resampling_tester(
                key_test, log_weights, conditional_resampling, M, self.K
            )

    @pytest.mark.xdist_group(name="resampling")  # Serialize to avoid OOM
    @parameterized.parameters(
        itertools.product([0, 42, 1337], ["systematic", "multinomial", "killing"])
    )
    def test_stop_gradient_resampling(self, seed, method):
        """Tests that gradient estimates are approximately correct under the stop-gradient decorator."""
        xs = jnp.linspace(-2.0, 2.0, 2_000)
        n = xs.shape[0]
        key = random.key(seed)
        true_sigma = 1.0
        resampling_fn = get_resampling(method)

        def _gaussian_log_weights(xs: jnp.ndarray, sigma: float) -> jnp.ndarray:
            return -0.5 * (xs / sigma) ** 2

        def base_lse(sigma):
            return logsumexp(_gaussian_log_weights(xs, sigma))

        def resampled_lse(sigma, resampling_fn):
            logws = _gaussian_log_weights(xs, sigma)
            _, logits_out, _ = resampling_fn(key, logws, xs, n)
            return logsumexp(logits_out)

        grad_base = jax.grad(base_lse)(true_sigma)
        grad_stop = jax.grad(
            lambda sigma: resampled_lse(sigma, stop_gradient_decorator(resampling_fn))
        )(true_sigma)
        grad_plain = jax.grad(lambda sigma: resampled_lse(sigma, resampling_fn))(
            true_sigma
        )

        print(grad_base, grad_plain, grad_stop, method)
        chex.assert_trees_all_close(grad_stop, grad_base, rtol=0.05, atol=0.05)
