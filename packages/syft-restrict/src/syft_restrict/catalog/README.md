# Risk catalog

Curated, advisory risk rules for the allow-list audit (`syft_restrict.audit`). **Advisory, not a
proof** — the verifier's default-deny is what actually gates calls; this catalog only helps a human
see, before running, what capabilities an `allow_functions` list grants.

## Layout

```
catalog/
  _common/
    default/catalog.json     # library-agnostic patterns, merged into every path
  <library>/
    <version>/catalog.json   # rules for <library> when the installed version matches <version>
```

- `<library>` is the first dotted component of an allowed path (`jax`, `flax`, …).
- `<version>` is a version prefix (`0.11`, `0.19`). The audit picks the **longest** version dir whose
  name matches the installed version on a dot boundary — `0.11` covers `0.11.x`, never `0.11` vs
  `0.19`. **There is no version-agnostic fallback per library:** if no version dir matches the
  installed version, that library contributes no rules and its paths fall to `review`. Add a version
  dir to cover a release.
- `_common/default/catalog.json` is always merged in. It holds truly cross-library patterns (`*.io_callback`, `*.tofile`, …) and blanket rules
  for libraries whose import root cannot be version-keyed (e.g. `orbax`: the `orbax` import root is a
  namespace package with no `__version__`; the distribution is `orbax-checkpoint`).

## File shape

Each `catalog.json` is:

```json
{
  "_about": "free-text note",
  "unsafe":   { "<dotted-path glob>": "why it is unsafe" },
  "dual_use": { "<dotted-path glob>": "the concrete reason it is flagged" },
  "safe":     { "<dotted-path glob>": "what the op is (terse)" }
}
```

- `unsafe` = known disk/network/host-callback surface.
- `safe` = pure computation: ordinary math (`einsum`, `matmul`, activations, reductions, reshapes),
  constants, RNG, and initializers.
- `dual_use` = a path flagged for a specific capability beyond pure computation. Each entry must
  state its own concrete reason (e.g. crossing the host/device boundary).

A path is matched strictest-first (`unsafe` → `dual_use` → `safe`). Anything matched by none defaults
to `review` — never silently to `safe`.

> [!Note]
>
> `safe` means "no disk/network/host-callback capability", not "no information
> flow". The catalog lists capabilities, not guarantees.
