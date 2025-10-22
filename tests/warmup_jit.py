import numpy as np

from cuthbertlib.resampling.utils import inverse_cdf_cpu, inverse_cdf_default


def main():
    # Dummy small inputs
    uniforms = np.linspace(0, 1, 64, dtype=np.float32)
    logits = np.random.randn(64).astype(np.float32)

    # Trigger JAX compilation (blocks until ready)
    inverse_cdf_default(uniforms, logits).block_until_ready()

    # Trigger Numba compilation (safe in serial mode)
    try:
        inverse_cdf_cpu(uniforms, logits)
    except Exception as e:
        print(f"Skipping Numba warmup: {e}")

    print("Warm-up complete.")


if __name__ == "__main__":
    main()
