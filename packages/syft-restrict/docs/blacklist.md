# What is not allowed

This document lists everything the checker rejects in the **private** region.

- The model is **default-deny**. Tables below name common rejections with clear
codes.
- Anything not listed as allowed in [verify.md](verify.md) is also
rejected.
- Each entry includes the **violation code** returned by `verify()`.


---

## Index of violation codes

| Code               | Meaning                                                                  |
| ------------------ | ------------------------------------------------------------------------ |
| `node-type`        | Syntax not on the allow list (walrus, `match`, …)                        |
| `banned-name`      | Reference or call to a banned builtin (`open`, `eval`, …)                |
| `banned-construct` | Forbidden statement/expression form (import, `with`, f-string, …)        |
| `call-not-allowed` | Library path not allow-listed (or hit by `disallow_functions`)           |
| `call-unresolved`  | Call target not provably safe                                            |
| `method-on-value`  | Named method on an unknown value (`x.reshape`)                           |
| `attr-on-value`    | Attribute read/write on an unknown value, or bad `self` chain            |
| `attr-not-allowed` | Library attribute path not allow-listed                                  |
| `dunder-attr`      | Dunder attribute (`.__class__`, …)                                       |
| `dunder-name`      | Bare dunder name (`__class__`)                                           |
| `dunder-def`       | Defining a forbidden magic method                                        |
| `decorator`        | Decorator not on the allow list                                          |
| `class-keyword`    | e.g. `metaclass=`                                                        |
| `class-base`       | Base class not allow-listed                                              |
| `reserved-name`    | Rebinding a trusted name (`jnp`, public wrapper, private def, `self`, …) |
| `operator-disabled` | Operator used without enabling its group in `allow_operators`             |

---

## Forbidden constructs

These constructs are banned outright in private code.

Code: **`banned-construct`**.

| Construct             | Example                            | Why                                                    |
| --------------------- | ---------------------------------- | ------------------------------------------------------ |
| Import                | `import os`, `from os import path` | Imports belong in public code                          |
| `with`                | `with open(f) as g: ...`           | Runs enter/exit hooks; often I/O                       |
| `try` / `raise`       | `try: ... finally: ...`            | Exception tricks / host surfaces                       |
| `global` / `nonlocal` | `global x`                         | Escape local naming rules                              |
| `del`                 | `del x`                            | Can remove names the policy relies on                  |
| `assert`              | `assert cond`                      | Disappears under `python -O`                           |
| Async                 | `async def`, `await`, …            | Out of scope for pure inference                        |
| Generators            | `yield`, `yield from`              | Suspended execution                                    |
| F-strings             | `f"hi"`, `f"{x}"`                  | Interpolation runs formatting with no normal call site |

```python
# rejected
import os
y = f"value={x}"
with ctx() as g:
    pass
```

### Unknown / future syntax

Any syntax not on the allow list is rejected.

Code: **`node-type`**.

| Example                | Notes            |
| ---------------------- | ---------------- |
| `y = (z := 1)`         | Walrus           |
| `match x: case 1: ...` | Pattern matching |

New Python syntax stays blocked until reviewed and explicitly allowed.

---

## Banned builtins

These names may never be **used** (i.e, loaded) in private code, even if they are
**not called**.

Code: **`banned-name`**.

### Dynamic code / reflection / I/O

- `eval`
- `exec`
- `compile`
- `__import__`
- `getattr`
- `setattr`
- `delattr`
- `hasattr`
- `vars`
- `globals`
- `locals`
- `dir`
- `open`
- `input`
- `breakpoint`
- `memoryview`
- `type`
- `__build_class__`
- `print`

### Formatting / buffer builtins

Same idea as calling dunders on a value (`x.__repr__()`), spelled as a bare call:

- `repr`
- `str`
- `ascii`
- `format`
- `bytes`

> [!WARNING]
> `bytes(x)` can dump raw buffer contents. `print` is a stdout channel. F-strings are banned
> separately as constructs (above), including with no `{...}` at all.

```python
# all rejected (banned-name)
f = open
d = {"o": open}
def run(op=open): ...
open("/etc/passwd")          # call reports banned-name once
y = [v for v in open(path)]  # passive positions still checked
```

---

## Calls and attributes

| What                                         | Example                                    | Code               |
| -------------------------------------------- | ------------------------------------------ | ------------------ |
| Library path not allowed                     | `np.dot(a, b)`                             | `call-not-allowed` |
| Disallow list hits an otherwise-allowed path | `jnp.save(...)` under `disallow_functions` | `call-not-allowed` |
| Call target not proven safe                  | `fn(x)` where `fn` is a parameter          | `call-unresolved`  |
| Named method on a value                      | `x.reshape(8, -1)`, `items.append(1)`      | `method-on-value`  |
| Attribute on a value                         | `x.shape`, `x.T`, `obj.send = data`        | `attr-on-value`    |
| Deep `self` chain                            | `self.a.b`, `self.sub.evil(...)`           | `attr-on-value`    |
| Unsafe `self.<name>(...)`                    | stashed `open` on `self` in `setup`        | `attr-on-value`    |
| Dunder attribute                             | `obj.__class__`, `self.__dict__`           | `dunder-attr`      |
| Bare dunder name                             | `c = __class__`                            | `dunder-name`      |
| Library attr not allowed                     | `np.pi` when numpy is not allowed          | `attr-not-allowed` |

```python
# rejected
a = x.reshape(8, -1)     # method-on-value
b = x.shape              # attr-on-value
c = obj.__class__        # dunder-attr

def apply(fn, x):
    return fn(x)         # call-unresolved

```

> [!IMPORTANT]
> **`call-unresolved` is default-deny for callees.** It catches dangerous callables that never
> mention a banned builtin by name (parameters, opaque locals, `d["k"](x)`).

Only single-level `self.<name>` / `cls.<name>` is special-cased. Rules for when `self.x(...)` is
safe are in [verify.md](verify.md#self-and-flax-style-modules).

---

## Classes, decorators, definitions

| What                       | Example                               | Code            |
| -------------------------- | ------------------------------------- | --------------- |
| Non-allow-listed decorator | `@evil`, `@property`, `@staticmethod` | `decorator`     |
| Class keywords             | `class M(nn.Module, metaclass=Meta)`  | `class-keyword` |
| Bad base class             | `class M(SomeLib)`, `class M(object)` | `class-base`    |
| Forbidden magic method     | `def __getattr__`, `def __reduce__`   | `dunder-def`    |

Allowed hooks only: `setup`, `__call__`, `__post_init__`.  
Allowed decorators only: `nn.compact`, `jax.jit`, `jax.named_scope`, `flax.linen.compact`.

```python
# rejected
class M(object):    # class-base
    @property       # decorator
    def w(self):
        return 1

    def __getattr__(self, name):    # dunder-def
        return None
```

---

## Reserved names

Rebinding a name the checker trusts reports **`reserved-name`**.

| Reserved             | Example rebind                                | Why                                    |
| -------------------- | --------------------------------------------- | -------------------------------------- |
| Import aliases       | `jnp = make_evil()`                           | Would lie about resolved library paths |
| Public wrappers      | nested `def transpose` shadowing a public one | Defeats the reviewed wrapper           |
| Private defs/classes | `helper = evil` after `def helper`            | Bare calls trust the def by name       |
| Safe builtins        | `list = None`                                 | Bare calls trust `list(...)` by name   |
| `self` / `cls`       | `self = x`, nested `def f(self):`             | `self.*` is trusted by spelling        |

Applies to assignment, `for` / comprehension targets, parameters, and nested defs.  
Private def names are also protected against rebind in **public** glue between private ranges.

```python
# rejected
import jax.numpy as jnp
jnp = 1     # reserved-name

def helper(x):
    return x
def f(evil):
    helper = evil      # reserved-name
    return helper(1)
```

---

## Operator bundles

If a group of operators is not in `allow_operators`, using any of its operators
reports **`operator-disabled`**.

| Bundle       | If missing, these fail |
| ------------ | ---------------------- |
| `arithmetic` | `a + b`, `-a`, …       |
| `comparison` | `a < b`, `a and b`, …  |
| `indexing`   | `x[0]`, `x[1:3]`, …    |

---

## Optional `disallow_functions`

There is no built-in library denylist. Everything not explicitly allow-listed is
rejected. Safety comes from **`allow_functions`**.

When you use a broad allow (`jax.*`), pass `disallow_functions=[...]` for a hard floor. Hits report
**`call-not-allowed`**.

Useful patterns under broad JAX/Flax allows:

- Host / debug / experimental: `jax.experimental.*`, `jax.debug.*`, `jax.pure_callback`, `*.io_callback`, `*.host_callback*`
- FFI / interop: `jax.dlpack.*`, `jax.ffi*`
- Array ↔ disk: `jax.numpy.save`, `load`, `tofile`, `fromfile`, `memmap`, …
- Checkpointing: `flax.serialization.*`, `flax.training.checkpoints.*`, `orbax.*`

```python
# public import, private use — still resolved and checked
from jax.numpy import save as persist
persist(x, "out.npz")   # call-not-allowed if disallow includes jax.numpy.save
```

---

## See also

- [verify.md](verify.md) — how checking works and what is allowed
- `tests/verify/test_disallowed.py` — disallowed tests
- `tests/verify/test_bypasses.py` — multi-step attack regressions
