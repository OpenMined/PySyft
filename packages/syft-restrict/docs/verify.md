# How verification works

`syft-restrict` checks the **private** lines of a Python file. Those lines must only do trusted
inference math: no host I/O, no dynamic code, no reaching into interpreter internals.

The tool **never runs** your code. It reads the source, checks the private region, and returns a
list of violations (empty on success).

> [!NOTE]
> This page is the **allow side**: how checking works and what private code may do.
>
> For everything that is default-denied, see [blacklist.md](blacklist.md).

Tests mirror this split:

| Doc                          | Test module                                                 | Role                   |
| ---------------------------- | ----------------------------------------------------------- | ---------------------- |
| This page                    | `tests/verify/test_whitelist.py`, `test_whitelisted_lib.py` | Green path             |
| [blacklist.md](blacklist.md) | `tests/verify/test_disallowed.py`                           | Default-deny catalog   |
| Edge / attack shapes         | `tests/verify/test_bypasses.py`                             | Multi-step regressions |

---

## Public vs private

Every file has two regions:

| Region      | Typical contents                                        | Checked by syft-restrict |
| ----------- | ------------------------------------------------------- | ------------------------ |
| **Public**  | Imports, data loading, wrappers the data owner can read | No (human review)        |
| **Private** | Hidden model math (`setup` / `__call__`, layers)        | Yes                      |


You mark private code either with `# syft-restrict: ...` comments in the source, or with explicit
1-based line ranges passed as `obfuscate`/`hide` to `run()` (the private region is their union
either way).

```python
# public — data owner reads this
import jax.numpy as jnp
from flax import linen as nn

def transpose(x):
    if not isinstance(x, jax.Array):
        raise TypeError
    return x.T

# syft-restrict: obfuscate-start
class Net(nn.Module):
    def setup(self):
        self.dense = nn.Dense(8)

    def __call__(self, x):
        return self.dense(transpose(x))
# syft-restrict: obfuscate-end
```

### Marker syntax

| Form                     | Example                                             | Marks                        |
| ------------------------ | --------------------------------------------------- | ---------------------------- |
| `obfuscate-start`/`-end` | `# syft-restrict: obfuscate-start` / `-end`         | a block, identifiers renamed |
| `hide-start`/`-end`      | `# syft-restrict: hide-start` / `-end`              | a block, whole lines blanked |
| Single-line `obfuscate`  | `MODEL_ID = "gemma-2b"  # syft-restrict: obfuscate` | that one line only           |
| Single-line `hide`       | `SALT = 1  # syft-restrict: hide`                   | that one line only           |

The marker comment lines themselves are excluded from the resolved range. They pass through to
the obfuscated output untouched, so a reader can still see where the private region was even
though its contents were renamed or blanked.

A `hide` block (or single-line `hide` marker) may nest inside an open `obfuscate` block — hide is
strictly stronger, so carving a stricter sub-region out of a looser one is safe:

```python
# syft-restrict: obfuscate-start
class Net(nn.Module):
    def setup(self):
    # syft-restrict: hide-start
        self.dense = nn.Dense(8)
    # syft-restrict: hide-end
# syft-restrict: obfuscate-end
```

The reverse isn't allowed (`obfuscate` can't nest inside `hide`), and neither kind nests inside
itself. Any of these raise `MarkerError`:

| Situation                                                       | Result                                                           |
| --------------------------------------------------------------- | ---------------------------------------------------------------- |
| `start` with no matching `end` (or vice versa)                  | `MarkerError`, names the line                                    |
| `hide-end`/`obfuscate-end` closing the wrong kind               | `MarkerError` (mismatched kind)                                  |
| `obfuscate` nested inside `hide`                                | `MarkerError`                                                    |
| `obfuscate` nested inside `obfuscate` (or `hide` inside `hide`) | `MarkerError`                                                    |
| A block with nothing between its start/end                      | `MarkerError` (empty block)                                      |
| No `# syft-restrict: ...` marker anywhere in the file           | `MarkerError` (the private region is designated by markers) |

`run()` resolves the private region from these markers; a file with none raises `MarkerError`.

> [!WARNING]
> Private code may **call** public wrappers by name. Public code is trusted, not verified.
>
> A clean `verify()` means the private region cannot escape on its own, not that
> the whole file is safe to execute.

### Imports

Imports happen in the public region but govern what the private region may
reach:

- **Public imports are unchecked.** Any module may be imported in public code;
  syft-restrict never restricts or inspects the import itself.
- **Private imports are banned outright** — an `import` inside the private
  region is rejected as `banned-construct`.
- **The private region's *use* of an import is what's gated.** A private call
  like `jnp.einsum(...)` resolves through the public import table and must match
  `allow_functions` (see below); otherwise it fails. So the control point is the
  private-side call, not the public import.

Star imports (`from jax import *`) are disallowed everywhere, because they make
it impossible to review the imported names.

---

## What the verifier does

The verifier walks the AST of the private region and verifies each construct against the allow list.

1. **Parse** the whole file.
2. **Record imports** from the public region (`import jax.numpy as jnp` → `jnp` means `jax.numpy`).
3. **Walk every private construct.** For each piece of syntax, either an
   explicitly rule allows it or it fails.
4. **Collect violations** with a line number and a short code (e.g. `banned-name`).

The default-deny semantics means that if nothing explicitly allows a construct,
it is rejected. New Python syntax stays blocked until someone reviews it.

---

## Always allowed syntax

These constructs are accepted in private code **when** nested pieces also obey
the rules. An allowed outer construct does not grant permission to its inner
pieces. For example, `if` is allowed, but the condition and body must also obey
the rules. The verifier checks each piece independently.

Assignments are allowed, but the target and the value are still checked:
reassigning a reserved name (`self`, an import alias, a public wrapper) or
assigning to an attribute on an opaque value (`obj.x = ...`) is rejected, and
the value is checked like any other expression.

| Category          | Examples                                                   |
| ----------------- | ---------------------------------------------------------- |
| Definitions       | `def`, `class`, `lambda`, `return`                         |
| Names & constants | variables, numbers, strings (not f-strings)                |
| Assignment        | `=`, `+=`, annotated assigns                               |
| Containers        | `list`, `tuple`, `dict`, `set`, comprehensions             |
| Calls             | `f(...)` when the callee is allowed (below)                |
| Control flow      | `if`, `for`, `while`, `break`, `continue`, `pass`, ternary |
| Operators         | only if the matching **bundle** is enabled (below)         |

> [!NOTE]
>
> **Never** allowed: imports, `with`, `try`/`raise`, `async`, generators,
> `assert`, `del`, f-strings, walrus `:=`, `match`/`case`, decorators. Those are listed
> under [blacklist.md](blacklist.md).

---

## Per-file policy configuration

### `allow_functions` — library paths callable by name

Dotted paths are resolved through the import table. An optional
`disallow_functions` list has priority over the allow list (hard floor over a
broad glob).

```python
import jax.numpy as jnp

jnp.einsum("ij->i", x)   # → jax.numpy.einsum  — must match allow_functions
jnp.save(path, x)        # → jax.numpy.save    — fails if in disallow_functions
```

| Rule                         | Meaning                                                 |
| ---------------------------- | ------------------------------------------------------- |
| Use paths, not whole modules | `jax.numpy.einsum` is named; `x.method()` is not a path |
| Call libraries **inline**    | `jnp.sin(x)` yes; `f = jnp.sin; f(x)` no                |
| Do not rebind import aliases | `jnp = evil` is rejected                                |
| Prefer specific allows       | `jax.numpy.einsum` over bare `jax.*`                    |

### `allow_operators` — operator bundles on a value

Named methods on an unknown value (`x.reshape(...)`, `x.T`) are never allowed:
the verifier cannot prove what they do. Instead you enable **language
operators** as bundles:

| Bundle       | Operators                           | Example         |
| ------------ | ----------------------------------- | --------------- |
| `arithmetic` | `+ - * / // % ** @`, unary          | `x + 1e-6`      |
| `comparison` | `== != < <= > >=`, `and`/`or`/`not` | `t == "local"`  |
| `indexing`   | `[]` and slices                     | `x[..., :half]` |

Anything else should go through a **public wrapper**:

```python
# public wrapper — data owner can review this
def shape_of(x):
    if not isinstance(x, jax.Array):
        raise TypeError
    return x.shape

# private
h = shape_of(x)   # allowed if shape_of is a public def
```

Disabled groups report `operator-disabled`.

---

## What you may call

A call is allowed only if the verifier can **prove** the callee. Otherwise it reports
`call-unresolved` (even when nothing looks “banned”).

| Allowed callee                                             | Example                              |
| ---------------------------------------------------------- | ------------------------------------ |
| Allow-listed import path, called inline                    | `jnp.sin(x)`, `nn.Dense(8)`          |
| Function or class defined in this file (private or public) | `Attention(cfg)`, `transpose(w)`     |
| Safe builtin (fixed list below)                            | `len(xs)`, `list(range(n))`          |
| Local traced to a safe source                              | `block = Attn(); block(x)`           |
| Vetted `self.<name>` / `self.<name>[i]`                    | `self.dense(x)`, `self.layers[i](x)` |

| Not allowed              | Example             | Why                    |
| ------------------------ | ------------------- | ---------------------- |
| Named method on a value  | `x.reshape(8, -1)`  | Type unknown           |
| Stashed library function | `f = jnp.sin; f(x)` | Must call paths inline |
| Opaque subscript call    | `d["k"](x)`         | Callee not identified  |

### Safe builtins

These may be called by bare name. They may **not** be reassigned.

- `int`
- `float`
- `bool`
- `len`
- `range`
- `enumerate`
- `zip`
- `min`
- `max`
- `sum`
- `abs`
- `round`
- `all`
- `any`
- `tuple`
- `list`
- `dict`
- `set`
- `sorted`
- `reversed`
- `isinstance`
- `super`

### Names you may not reassign

Because call sites trust some names by spelling alone:

| Name kind            | Example                                      |
| -------------------- | -------------------------------------------- |
| Import aliases       | `jnp`, `nn`                                  |
| Public wrappers      | `transpose`                                  |
| Private defs/classes | `Attention`, `helper`                        |
| Safe builtins        | `list`, `range`                              |
| `self` / `cls`       | only as the real first parameter of a method |

Rebind is rejected even in **public** glue if the name is a private def, otherwise a public
`helper = evil` between private chunks could be used to evade the verifier.

Ordinary locals may be reassigned freely, but the verifier tracks their source
and rejects any call to a local that can't be resolved to a safe origin.

---

## `self` and Flax-style modules

Private models usually look like:

```python
class Net(nn.Module):              # base must be allow-listed (e.g. flax.linen.Module)
    def setup(self):                 # often public (data owner can read wiring)
        self.dense = nn.Dense(8)     # allow-listed constructor → safe to call later
        self.layers = [Block() for _ in range(3)]

    def __call__(self, x):           # often private
        x = self.dense(x)
        block = self.layers[0]
        return block(x)
```

| Rule                                              | Meaning                                                               |
| ------------------------------------------------- | --------------------------------------------------------------------- |
| Only first method param may be named `self`/`cls` | Nested `def helper(self):` is rejected                                |
| Do not rebind `self`/`cls`                        | `self = x` is rejected                                                |
| Single-level only                                 | `self.dense` yes; `self.sub.evil` no                                  |
| Call only if every assignment was a vetted source | allow-listed constructor, file-local class/def, or list/comp of those |
| Inherited attrs (never assigned)                  | Treated as safe (e.g. `self.param` on `nn.Module`)                    |
| `self.x += ...`                                   | Always unsafe for later calls                                         |

> [!TIP]
> Put dangerous or opaque wiring in **public** `setup` if the data owner should read it; keep
> private `__call__` as pure math.
>
> The verifier still uses public `setup` assignments when deciding
> whether `self.<name>(...)` is safe.

### Classes and hooks

| Allowed                                                       | Not allowed                                                                     |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Bases that resolve to an allow-listed path (e.g. `nn.Module`) | `object`, random private bases, non-allow-listed libs                           |
| —                                                             | Any decorator: `@property`, `@staticmethod`, `@nn.compact`, arbitrary functions |
| Defining `setup`, `__call__`, `__post_init__`                 | `__getattr__`, `__reduce__`, other magic methods      |
| —                                                             | `metaclass=` / other class keywords                                             |

---

## Whitelist examples

```python
# control flow + safe builtins
def f(xs):
    acc = [abs(v) for v in xs if v]
    for v in xs:
        if v:
            break
    return acc
```

```python
# public wrapper + private call
def transpose(x):
    return x  # data owner reviews this

def private_math(w):
    return transpose(w)
```

```python
# library path (imports public)
import jax.numpy as jnp
# private (allowlisted path):
r = jnp.einsum("ij,jk->ik", a, b)
```

```python
# annotations may mention banned type names; they are not executed
def f(x: list[str]) -> dict[str, bytes]:
    return {}
```

---

## Limits of static checking

Even a clean verify does **not** stop:

1. **Encoding secrets in model outputs** (logits/tokens)
2. **Timing / cache side channels** — enclave-level concern.
3. **New dangerous APIs under a broad allow** (`jax.*`) — prefer tight allows + `disallow_functions`.
4. **Compromised JAX/Flax builds** — attest library versions separately.
5. **Malicious public wrappers** — still depend on human review of the public region.

---

## See also

- [blacklist.md](blacklist.md) — default-deny catalog and violation codes
- [code-layout.md](code-layout.md) — source modules
- `tests/verify/` — executable examples of allow vs deny
