# syft-restrict

Static analysis for JAX/Flax inference code. You mark the parts of a file that contain the model
math as *private*; `syft-restrict` checks that those lines only do trusted computation — no sneaky
data exfiltration — and can emit an obfuscated copy that hides the architecture.

The tool never runs your code. It parses the source, walks the private lines, and either reports
violations or hands back an attestation plus the obfuscated artifact. Run that inside a
[trusted enclave](https://en.wikipedia.org/wiki/Trusted_execution_environment) and a clean result
is evidence that the file is a genuine inference pipeline, not a data-stealing wrapper.

The idea comes from [RestrictedPython](https://github.com/zopefoundation/RestrictedPython): start
from default-deny and only permit what you explicitly trust. Three differences here:

- Analysis, not execution. You get a report (or certificate) and an obfuscated file, not a sandboxed runtime.
- Aimed at ML inference (JAX / Flax), not general Python.
- A public/private split. Imports, I/O, the generation loop, and thin wrappers around library calls
  live in the *public* region and are read as-is. Only the *private* region is checked and obfuscated;
  the private code may call into the public wrappers.

## Install

```bash
pip install syft-restrict     # or: uv add syft-restrict
```

## Usage

```python
import syft_restrict as restrict

result = restrict.run(
    "gemma_inference.py",
    obfuscate=[[22, 93], [99, 280]],                        # 1-based ranges: identifiers renamed, constants blanked
    hide=[],                                                # 1-based ranges: whole line replaced with ■■■■■■■■
    allow_functions=["jax.*", "flax.linen.*"],              # paths callable BY NAME (resolved against imports)
    allow_methods=["arithmetic", "indexing", "comparison"], # operator bundles allowed ON A VALUE
)
# Verified region = obfuscate ∪ hide (both are private code; both are checked).
# On success: writes gemma_inference.obfuscated.py and returns result.certificate.
# On a violation: raises PolicyViolation naming each offending line (strict=True, the default).
```

`restrict.verify(...)` runs the check without writing output — it returns violations instead of
raising. Pass `strict=False` to `run` if you want a `RunResult` with `.ok` / `.violations` and no
files written.

See [examples/gemma_inference.py](examples/gemma_inference.py) for a full example and
[examples/gemma_inference.obfuscated.py](examples/gemma_inference.obfuscated.py) for what gets
generated (`python examples/generate.py` to regenerate).

## How it works

`restrict.run()` verifies first, then obfuscates:

1. Parse the whole file and build an import binding table (`import jax.numpy as jnp` → `jnp` maps to `jax.numpy`).
2. Walk the private lines with default-deny. Each AST node must match an explicit rule: allowed node
   type, allow-listed call target, local name, enabled operator bundle, or a `self.<name>` access.
   Anything else is a violation.
3. If the walk is clean, obfuscate the private lines and write the artifact plus certificate. Otherwise nothing is written.

## Documentation

- [docs/verify.md](docs/verify.md) — how the checker decides what's allowed, what order things run in, edge cases, and known limits.
- [docs/blacklist.md](docs/blacklist.md) — everything that gets rejected, with violation codes.
- [docs/disallowed-ast-examples.md](docs/disallowed-ast-examples.md) — example snippets that fail and why.
- [docs/code-layout.md](docs/code-layout.md) — what each source module does.