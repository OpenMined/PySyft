# Code layout

Brief map of `src/syft_restrict/`:

- `astutil.py` — AST and line-range helpers with no policy logic. Shared by the verifier, obfuscator, and runner.
- `policy.py` — allow-listed function paths, the JAX/Flax denylist, operator bundles, and the `Policy` model.
- `verifier.py` — the static checker. Walks private lines default-deny and collects violations. This is the core.
- `obfuscator.py` — turns verified private lines into a readable-but-secret artifact (renamed identifiers, blanked constants).
- `runner.py` — verify → obfuscate → certificate. `run()` lives here.
- `errors.py` — `RestrictError` and `PolicyViolation`.
- `__init__.py` — public re-exports.