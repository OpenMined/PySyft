"""Regenerate gemma_inference.obfuscated.py + its certificate.

Run from the package root:  uv run python examples/generate.py

The policy here lists the *exact* lower-level JAX/Flax symbols the hidden region uses,
rather than broad globs (`jax.*`). Each call leaf (`jax.numpy.einsum`, …) is enforced
individually; `jax.lax` / `jax.nn` are included only because the calls are written as
deep attribute paths (`jax.lax.rsqrt`), so the checker also evaluates those module
references. `jax.numpy` needs no such entry — it's aliased as `jnp` (a bare name).

OBFUSCATE keeps structure legible (identifiers renamed, constants blanked); HIDE replaces
whole lines with a ■■■■■■■■ marker. The verified region is their union.
"""

import json
from pathlib import Path

from syft_verifuscate import run

EX = Path(__file__).parent

result = run(
    path=EX / "gemma_inference.py",
    obfuscate=[
        [24, 88],  # CONFIG dict (+ commented size variants) + shared constants
        [91, 91],
        [99, 99],
        [110, 110],
        [122, 122],  # the 4 standalone helper def lines
        [151, 152],
        [155, 155],  # Einsum:      class + setup def | __call__ def
        [159, 160],
        [163, 163],  # RMSNorm
        [168, 171],
        [178, 178],  # Attention    (incl. `cfg: dict`)
        [210, 211],
        [215, 215],  # FeedForward
        [221, 225],
        [233, 233],  # Block        (incl. `cfg`, `attn_type`)
        [244, 247],
        [250, 250],  # Embedder
        [255, 258],
        [267, 267],  # Transformer
    ],
    hide=[
        [92, 93],
        [100, 107],
        [111, 119],
        [123, 129],  # the 4 standalone helper bodies
        [153, 153],
        [156, 156],  # Einsum      setup / __call__ bodies
        [161, 161],
        [164, 165],  # RMSNorm
        [172, 176],
        [179, 207],  # Attention
        [212, 213],
        [216, 218],  # FeedForward
        [226, 231],
        [234, 241],  # Block
        [248, 248],
        [251, 252],  # Embedder
        [259, 265],
        [268, 292],  # Transformer
    ],
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
    allow_methods=["arithmetic", "indexing", "comparison"],
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
