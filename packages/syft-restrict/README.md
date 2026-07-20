# syft-restrict

A **static analyzer** for Python source files that **default-denies** dynamic
constructs in a **private** region, typically the model logic that must be kept private, while allowing the **public** region to be reviewed by the data owner in an **obfuscated** copy.

## Overview

`syft-restrict` splits a Python file into two regions:

- **Private** — the lines the user marks as private, typically the model logic that
  must not be readable. Mark them with `# syft-restrict: obfuscate-start` /
  `# syft-restrict: obfuscate-end` or `# syft-restrict: hide-start` / `# syft-restrict: hide-end` comments in the source.
- **Public** — every other line outside the marked ranges: imports, data loading, wrappers. It is
  never checked or obfuscated. The recipient of the obfuscated file can read it directly.

Given that split, `syft-restrict` executes the following two steps:

- **The verifier** statically walks the private region and analyzes it against a default-deny
  policy. Every construct must be explicitly allowed, or verification fails
  and reports each offending line.
- **The obfuscator** runs after a clean verify. It creates an obfuscated copy by rewriting the private
  region — renaming identifiers, blanking constants, replacing lines with `■■■■■■■■` — so the logic can't be read
  back out, while the public region is copied through untouched.

The **verifier** is inspired by
[**RestrictedPython**](https://github.com/zopefoundation/RestrictedPython), but
differs in a few ways:

- **It analyzes — it doesn't run.** It never executes the source. Only the static checks decide pass/fail.
- The public code may **import allowlisted libraries** and call into them.
  Imports are not allowed in private code, and any library call must be
  **explicitly allow-listed**.
- **Dynamic Python lives in the public part.** Anything dynamic the author needs
  (file/tokenizer loading, the generation loop, wrappers around library methods)
  must be written in the **public** region — where it can be reviewed directly — and
  may be **called by** the private part.

## Usage

Mark the private region in the source with comments:

```python
# syft-restrict: obfuscate-start
def attention(x):
    ...
# syft-restrict: obfuscate-end

MODEL_ID = "gemma-2b"  # syft-restrict: hide
```

_See [docs/verify.md](docs/verify.md#public-vs-private) for the full marker syntax (single-line markers, nesting rules, and error cases)._

Then run:

```python
import syft_restrict as restrict

result = restrict.run(
    "gemma_inference_marked.py",
    allow_functions=["jax.*", "flax.linen.*"],  # functions callable BY NAME (path-resolved)
    allow_operators=["arithmetic", "indexing", "comparison"],  # operators allowed ON A VALUE
)
# On success: writes gemma_inference_marked.obfuscated.py and returns result.certificate.
# On a policy violation: raises PolicyViolation naming each offending line.
# If the file has no `# syft-restrict: ...` markers: raises MarkerError.
```

The private region is designated **only** by `# syft-restrict: ...` comment markers in the
source; `run()` resolves them and refuses a file that carries none. This is how
**[examples/gemma_inference_marked.py](examples/gemma_inference_marked.py)** generates its
obfuscated copy — see [examples/generate_marked.py](examples/generate_marked.py).

Two optional strictness flags, both defaulting to the permissive behavior:

```python
result = restrict.run(
    "gemma_inference_marked.py",
    allow_functions=["jax.*", "flax.linen.*"],
    allow_operators=["arithmetic", "indexing", "comparison"],
    allow_local_assignments=True,   # False: a local aliased to a callable can't be called by name
    allow_base_class_attributes=True,  # False: a never-assigned self.<attr> is not presumed inherited
)
```

If syft-restrict was successfully executed on the true original file, the obfuscated file can be safely shared without exposing the private section.

Pass `strict=False` to `run` to get a `RunResult` with `.ok` / `.violations`
instead of an exception.

## Documentation

- [docs/verify.md](docs/verify.md) — how verification works and what private code may do (allow side).
- [docs/blacklist.md](docs/blacklist.md) — default-deny catalog and violation codes (deny side).
- [docs/code-layout.md](docs/code-layout.md) — source modules and test layout.
