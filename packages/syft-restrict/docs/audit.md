# Allow-list audit

`syft_restrict.auditor.audit_allow_functions` classifies each path in an `allow_functions` list
against a curated catalog. It is advisory — it shows what an allow-list grants before a run; the
verifier's default-deny is what controls calls. `run()` performs this audit and attaches the report
to `result.audit`; it never affects `result.ok`.

## Verdicts

- **`unsafe`** — reads or writes files, opens network connections, or runs host callbacks.
- **`safe`** — pure computation (math, reductions, indexing, activations, constants, RNG,
  initializers, and autodiff/compile transforms that wrap pure computation).
- **`dual_use`** — not pure computation and not a clear file/network/callback op, but with one other
  capability the verifier cannot analyze; each entry states it. Rare — the examples are host/device
  transfers (`jax.device_get`/`device_put`) and `flax.linen.Module`'s `allow_base_class_attributes`
  flag.
- **`review`** — anything uncatalogued. A warning for a human, never a silent pass; unknown paths are
  never assumed safe.

Matching is strictest-first: (`unsafe` → `dual_use` → `safe`); an unmatched path is `review`.

## Catalog layout

No catalog ships with the package. A catalog is a directory passed as `catalog_dir`, laid out as:

```
<catalog_dir>/
  _common/default/catalog.json     # cross-library patterns, merged into every path
  <library>/<version>/catalog.json # rules for <library> at a matching version
```

- `<version>` is a prefix matched on dot boundaries: `0.11` covers `0.11.x`, never `0.19`. There is
  no per-library fallback — an unmatched version contributes no rules, so its paths fall to `review`.
- `_common/default` holds cross-library patterns (`*.io_callback`, `*.tofile`, …) and rules for
  libraries with no version-keyable import root (e.g. `orbax`).

A worked example lives in `examples/catalog`. Each file is
`{"_about": …, "unsafe": {…}, "dual_use": {…}, "safe": {…}}` with single-line string values. Keep it
sorted and normalized with the linter (`PATH` is the file or directory to lint):

```bash
uv run syft-restrict-lint PATH --fix
```

## Generate a catalog for a library

syft-restrict does **not** maintain catalogs; `examples/catalog` (`jax/0.11`, `flax/0.12`) is a worked
example. Uncatalogued paths fall to `review`, so a partial catalog is safe as long as it covers the
actual calls.

Keep the catalog in its own directory and point the auditor at it with `run(..., catalog_dir=…)`.

Fill in `{LIBRARY}` and `{VERSION}` below and paste it into an AI coding agent with Python and shell
access where that version is installed. Review every `unsafe` and `dual_use` entry before trusting
it.

```text
You make a risk catalog for a static allow-list audit tool. It classifies each dotted import path a
user allow-lists. Produce the catalog JSON for the commonly-used public API of {LIBRARY} {VERSION}.

CONTEXT
- Advisory tool: it shows a reviewer what an allow-list grants before a run.
- Any path you do NOT list becomes "review" (a warning). A partial catalog is fine. Cover the
  functions and classes that real models and tutorials use, and leave out the rest.
- {LIBRARY} {VERSION} is installed. Verify every path resolves in that version (import the module and
  check the attribute); drop any that do not.

CATEGORIES
- "unsafe": reads or writes files, opens network connections, or runs host callbacks (arbitrary host
  code). Be complete. Use fnmatch globs for whole risky namespaces (e.g. "{LIBRARY}.io.*").
- "safe": pure computation — returns a value with no external effect. Arithmetic, linear algebra,
  reductions, comparisons, indexing, activations, constants, random numbers, initializers, and
  compile/autodiff transforms that wrap pure computation.
- "dual_use": not pure computation and not a clear file/network/callback op, but with one other
  capability a static checker cannot analyze (e.g. moving data across the host/device boundary, or a
  flag that widens what the checker accepts). State that capability. This category is often empty.

RULES
- If the only notable thing about a function is that it computes on data, it is "safe" — even when
  very flexible. "dual_use" needs a capability that is not computation.
- Never put an uncertain path in "safe"; leave it out (it becomes "review"). Do not use a wide glob
  like "{LIBRARY}.*" in "safe".
- Notes: one line, plain description. For "unsafe"/"dual_use", state the specific reason (for
  "dual_use", something other than "computes on data"). Never describe how to misuse anything.

OUTPUT
Produce one JSON object in this shape (keys in any order; the linter sorts them):

{
  "_about": "Risk rules for {LIBRARY} {VERSION}. 'unsafe' = disk/network/host-callback; 'safe' = pure computation; 'dual_use' = flagged for a specific capability beyond pure computation, reason stated per entry. This is advisory, not a proof.",
  "unsafe":   { "<dotted-path or fnmatch glob>": "why it is unsafe" },
  "dual_use": { "<dotted-path>": "the specific non-computation capability it has" },
  "safe":     { "<dotted-path>": "what the op is" }
}

Omit an empty category. Every value is a single-line string. Use the primary public import path
(e.g. "{LIBRARY}.nn.relu"), not a deep private module path.

WRITE AND VALIDATE
- Write it to "catalog/{LIBRARY}/{VERSION}/catalog.json" (create the directories).
- Run "uv run syft-restrict-lint catalog --fix"; fix whatever it reports.
- Re-import every catalogued path in {LIBRARY} {VERSION}, drop any that fail, and lint again.
```
