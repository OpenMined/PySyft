"""Regenerate gemma_inference.obfuscated.py + its certificate.

Run from the package root:  uv run python examples/generate.py

The policy here lists the *exact* lower-level JAX/Flax symbols the hidden region uses,
rather than broad globs (`jax.*`). Each call leaf (`jax.numpy.einsum`, …) is enforced
individually; `jax.lax` / `jax.nn` are included only because the calls are written as
deep attribute paths (`jax.lax.rsqrt`), so the checker also evaluates those module
references. `jax.numpy` needs no such entry — it's aliased as `jnp` (a bare name).
"""

import json
from pathlib import Path

from syft_verifuscate import run

EX = Path(__file__).parent

# 1-based inclusive line ranges of the hidden model definition.
PRIVATE = [[22, 93], [99, 130], [156, 297]]

# Exactly the external symbols the hidden region calls / subclasses.
ALLOW_FUNCTIONS = [
    "jax.numpy.einsum",
    "jax.numpy.mean",
    "jax.numpy.square",
    "jax.numpy.arange",
    "jax.numpy.sin",
    "jax.numpy.cos",
    "jax.numpy.concatenate",
    "jax.numpy.tril",
    "jax.numpy.triu",
    "jax.numpy.ones",
    "jax.numpy.where",
    "jax.numpy.repeat",
    "jax.numpy.sqrt",
    "jax.numpy.array",
    "jax.numpy.float32",
    "jax.numpy.bool_",
    "jax.lax.rsqrt",
    "jax.nn.softmax",
    "jax.nn.gelu",
    "flax.linen.Module",
    # module references required by the deep-path call style (jax.lax.rsqrt, jax.nn.softmax):
    "jax.lax",
    "jax.nn",
]
ALLOW_METHODS = ["arithmetic", "indexing", "comparison"]


def main() -> None:
    result = run(
        EX / "gemma_inference.py",
        private=PRIVATE,
        allow_functions=ALLOW_FUNCTIONS,
        allow_methods=ALLOW_METHODS,
    )
    cert_path = EX / "gemma_inference.certificate.json"
    cert_path.write_text(json.dumps(result.certificate, indent=2) + "\n")
    print(f"wrote {result.obfuscated_path}")
    print(f"wrote {cert_path}")
    print(f"policy_id={result.certificate['policy_id']} "
          f"n_calls_checked={result.certificate['n_calls_checked']}")


if __name__ == "__main__":
    main()
