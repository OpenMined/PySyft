"""Regenerate gemma_inference.obfuscated.py + its certificate.

Run from the package root:  uv run python examples/generate.py

The model carves out its own private region with `# syft-restrict: ...` comment markers, so
run() resolves the region straight from the source — this is the supported UX. (The
generate_ranges.py / gemma_inference_ranges.py variant drives the same model through explicit
hand-counted line ranges instead.)
"""

import json
from pathlib import Path

from syft_restrict import run

EX = Path(__file__).parent

result = run(
    path=EX / "gemma_inference.py",
    allow_functions=[
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
        "jax.numpy.transpose",
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
    ],
    allow_operators=["arithmetic", "indexing", "comparison"],
)

cert_path = EX / "gemma_inference.certificate.json"
cert_path.write_text(json.dumps(result.certificate, indent=2) + "\n")

policy_id = result.certificate["policy_id"]
n_calls_checked = result.certificate["n_calls_checked"]
print(
    f"""wrote {result.obfuscated_path}
wrote {cert_path}
policy_id={policy_id} n_calls_checked={n_calls_checked}"""
)
