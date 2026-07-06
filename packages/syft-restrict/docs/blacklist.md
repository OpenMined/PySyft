# What is not allowed

Everything the checker rejects in the private region. For the reasoning behind the rules see [verify.md](verify.md). For example snippets see [disallowed-ast-examples.md](disallowed-ast-examples.md).

The model is default-deny: the tables below name what's rejected explicitly (for clear violation
messages), but anything missing from the whitelist in [verify.md](verify.md) is rejected too.

Each entry lists the error `code` reported by the checker.

---

## Banned statement / expression node types

Rejected immediately — they reach the host, filesystem, or interpreter, or reintroduce dynamic
control flow. (`_BANNED_NODES` in `verifier.py`.) Code: `banned-construct`.

| Node                                                 | Why                                                                                     |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `Import`, `ImportFrom`                               | Imports belong in the public region; inside private code they'd name arbitrary modules. |
| `With`                                               | Context managers run `__enter__`/`__exit__` code.                                       |
| `Try`, `Raise`                                       | Exception control flow isn't needed for pure inference math.                            |
| `Global`, `Nonlocal`                                 | Rebinding outer-scope names breaks the local-only name rules.                           |
| `Delete`                                             | `del` can remove names the checker relies on.                                           |
| `Assert`                                             | Asserts vanish under `python -O`; they must never carry safety guarantees.              |
| `AsyncFunctionDef`, `AsyncFor`, `AsyncWith`, `Await` | Async machinery is out of scope.                                                        |
| `Yield`, `YieldFrom`                                 | Generators suspend and resume execution.                                                |

## Unknown / future syntax

Any node type not on the allow-list in [verify.md](verify.md#always-on-allow-list) is rejected —
walrus (`NamedExpr`), `match`/`case`, etc. Code: `node-type`. New Python syntax stays denied until
reviewed.

---

## Banned builtins

These names may never be called or referenced — aliasing them, putting them in a container, or
returning them is caught at the reference site. (`BANNED_NAMES` in `policy.py`.)

**Dynamic-code / reflection / IO hatches** — code `banned-call`:
`eval`, `exec`, `compile`, `__import__`, `getattr`, `setattr`, `delattr`, `hasattr`, `vars`,
`globals`, `locals`, `dir`, `open`, `input`, `breakpoint`, `memoryview`, `type`, `__build_class__`,
`print`.

**Dunder-proxy builtins** — same escape as calling a dunder on a value (`x.__repr__()`), spelled as
a bare call. Code `banned-call`: `repr`, `str`, `ascii`, `format`, `bytes`. (`bytes(x)` losslessly
serializes an array's raw memory; `print` is a stdout exfil channel.)

The same escape via an f-string conversion flag (`f"{x!r}"`, `f"{x!s}"`, `f"{x!a}"`, `f"{x=}"`) has
no `Call` node and is rejected as `method-on-value`.

---

## The JAX / serialization denylist

Host callbacks, IO, FFI, and serialization paths that can run host code or touch disk. The denylist
beats the allowlist — rejected even under an otherwise-allowed module like `jax.numpy.*`.
(`JAX_DENYLIST` in `policy.py`.) Hitting one by dotted path or bare public import reports
`call-not-allowed`.

Covers: `jax.experimental.*`, `jax.debug.*`, `jax.pure_callback`, `*.io_callback`, `*.host_callback*`,
`jax.profiler.*`, `jax.monitoring.*`, `jax.distributed.*`, `jax.dlpack.*`, `jax.ffi*`, `jax.extend.*`;
array↔disk functions `jax.numpy.{save,savez,savez_compressed,load,tofile,fromfile,memmap,savetxt,
loadtxt,genfromtxt}`; and `flax.serialization.*`, `flax.training.checkpoints.*`, `orbax.*`.

---

## Calls, attributes, and names

| What                                     | Example                                 | Code               |
| ---------------------------------------- | --------------------------------------- | ------------------ |
| Non-allow-listed library path            | `np.dot(a, b)` (numpy not allow-listed) | `call-not-allowed` |
| Named method on an opaque value          | `x.reshape(8, -1)`, `items.append(1)`   | `method-on-value`  |
| Attribute read on an opaque value        | `x.shape`, `x.T`, `x.ndim`              | `attr-on-value`    |
| Dunder attribute read on any object      | `obj.__class__`, `obj.__dict__`         | `dunder-attr`      |
| Bare dunder name reference               | `c = __class__`                         | `dunder-name`      |
| Non-`self` attribute write               | `obj.send = data`                       | `attr-on-value`    |
| Self chain deeper than one level         | `self.sub.evil(...)`, `self.a.b`        | `attr-on-value`    |
| Non-allow-listed attribute off a library | `np.pi` (numpy not allow-listed)        | `attr-not-allowed` |

Only single-level `self.<name>` / `cls.<name>` reads and writes are allowed (see the
[self-attribute safety table](verify.md#edge-cases) in verify.md).

---

## Classes, decorators, and definitions

| What                                                            | Example                             | Code            |
| --------------------------------------------------------------- | ----------------------------------- | --------------- |
| Non-allow-listed decorator (incl. `@property`)                  | `@evil`, `@property`                | `decorator`     |
| Class keyword argument                                          | `class M(object, metaclass=Meta)`   | `class-keyword` |
| Non-allow-listed base class                                     | `class M(SomeLib)`                  | `class-base`    |
| Magic/hook method other than `setup`/`__call__`/`__post_init__` | `def __getattr__`, `def __reduce__` | `dunder-def`    |

`@property` is rejected because it runs code on bare attribute access (`block.w`) — same hook
class as a dunder def.

---

## Reserved-name rebinding

Rebinding a name the resolver trusts would poison verification. Rejected wherever the name is
bound — assignment, `for`/comprehension target, or parameter. Code: `reserved-name`.

- A trusted module alias (`jnp`, `nn`, `lax`, …) — rebinding makes the import table a lie, so every
  "allow-listed path" through that name becomes attacker-controlled.
- A visible wrapper name — rebinding `transpose = evil` defeats the wrapper's type guard.
- `self` / `cls` — the exemption is trusted by identifier alone; rebinding it (or reusing it as an
  unrelated parameter) would grant an attacker's object the same trust. See
  [verify.md#edge-cases](verify.md#edge-cases).

---

## Container / aliasing tricks

Storing a banned-builtin reference where it could be dispatched later is rejected at construction
time (we don't track which slot holds what). Code: `banned-construct`.

- Banned reference inside a `list`/`dict`/`set`/`tuple` literal — `con = [eval]`, `d = {"o": open}`.
- Storing a banned reference into a subscript slot — `d["k"] = open`.

The reference is also caught in every other position: `f = eval` (alias), `a = b = open` (chained),
`a, b = (1, open)` (unpack), `return open` then `leak()(...)`, `op=open` (default arg),
`open if c else eval` (IfExp branch), and via a homoglyph copy of a previously-stashed name. All
report `banned-call` at the reference.

---

## Operator bundles not enabled

If the policy's `allow_methods` doesn't enable a bundle, using its operators reports
`bundle-disabled`: `arithmetic` (`BinOp`/`UnaryOp`), `comparison` (`Compare`/`BoolOp`), `indexing`
(`Subscript`/`Slice`).