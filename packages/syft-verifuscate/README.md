# syft-verifuscate

Verify-then-obfuscate for JAX/Flax model-inference code that runs in an enclave on a second party's
private data. `verifuscate.run(...)`:

1. **Verifies** (static, before running) that the _private_ model-definition lines only do trusted JAX/
   Flax math — no imports, no file/network, no dynamic-Python escape hatches, no named method calls on
   opaque values. This is the no-data-theft guarantee.
2. **Obfuscates** those lines (rename identifiers, blank constants/einsum strings, strip comments) so
   the model architecture stays secret. Every non-private line is copied byte-for-byte.

The design and threat model are documented under `research/verifuscate/` in the repo root
(`SYNTHESIS.md`, `approach-A-ast-obfuscation.md`, `approach-B-whitelist-sandbox.md`).

## Usage

```python
import syft_verifuscate as verifuscate

result = verifuscate.run(
    "gemma_inference.py",
    private=[[22, 93], [99, 280]],                                # 1-based inclusive line ranges to hide+verify
    allow_functions="jax.*, flax.linen.*",                        # things callable BY NAME (path-resolved)
    allow_methods="arithmetic, indexing, comparison, metadata",    # operators allowed ON A VALUE
)
# On success: writes gemma_inference.obfuscated.py and returns result.certificate.
# On a policy violation: raises PolicyViolation naming each offending line (strict=True, the default).
```

Use `verifuscate.verify(...)` for the check alone (it returns violations instead of raising), or pass
`strict=False` to `run` to get a `RunResult` with `.ok` / `.violations` and no exception.

## Scope and honest limits

The static checker enforces the documented rule set with single-pass alias resolution (no deep
data-flow analysis). It does **not** address the output channel (a model encoding private data into its
legitimate result), timing/side channels, or the general undecidability of Python — see
`research/verifuscate/approach-B-whitelist-sandbox.md` §6. The obfuscated artifact is **display-only**
(the real, unobfuscated code is what runs in the enclave).
