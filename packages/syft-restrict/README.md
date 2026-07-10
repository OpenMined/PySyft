# syft-restrict

A **static analyzer** for Python source files that **default-denies** dynamic
Python constructs. 

## Overview

`syft-restrict` is inspired by
[**RestrictedPython**](https://github.com/zopefoundation/RestrictedPython), but
differs in a few ways:

- **Syft-restrict analyzes, it doesn't run.** syft-restrict statically analyzes
  a source file and **fails if the analyzed part uses any dynamic Python**. Its
  output is a **report** (the violations, or — on success — an attestation) plus
  an **obfuscated copy** of the code.
- **Public / private split.** The user marks some lines of the file _private_
  and leaves the rest _public_. restrict analyzes only the **private regions**,
  and only those are **obfuscated** in the emitted artifact; the public lines
  are copied through and read directly.  
- The public code may **import allowlisted libraries** and call into them.
  Imports are not allowed in private code, and any library call must be
  **explicitly allow-listed**.
- **Dynamic Python lives in the public part.** Anything dynamic the author needs
  (file/tokenizer loading, the generation loop, wrappers around library methods)
  must be written in the **public** region -- where it is reviewed directly --and
  may be **called by** the private part.

## Usage

```python
import syft_restrict as restrict

result = restrict.run(
    "gemma_inference.py",
    obfuscate=[[22, 93], [99, 280]],  # 1-based ranges: identifiers renamed, constants blanked
    hide=[],                          # 1-based ranges: whole line replaced with ■■■■■■■■
    allow_functions=["jax.*", "flax.linen.*"],  # functions callable BY NAME (path-resolved)
    allow_operators=["arithmetic", "indexing", "comparison"],  # operators allowed ON A VALUE
)
# On success: writes gemma_inference.obfuscated.py and returns result.certificate.
# On a policy violation: raises PolicyViolation naming each offending line.
```

This command reads
**[examples/gemma_inference.py](examples/gemma_inference.py)** and generates
**[examples/gemma_inference.obfuscated.py](examples/gemma_inference.obfuscated.py)**.

If syft-restrict was succesfully executed on the true inference file and the
library was not modified, this proves the code does not exfiltrate the inputs, and
the obfuscated file can be safely shared with a third party.

Use `restrict.verify(...)` for the check alone (it returns violations instead of
raising), or pass `strict=False` to `run` to get a `RunResult` with `.ok` /
`.violations` and no exception.


## Documentation

- [docs/verify.md](docs/verify.md) — how verification works and what private code may do (allow side).
- [docs/blacklist.md](docs/blacklist.md) — default-deny catalog and violation codes (deny side).
- [docs/code-layout.md](docs/code-layout.md) — source modules and test layout.
