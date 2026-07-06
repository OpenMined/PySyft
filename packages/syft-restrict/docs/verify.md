# How verification works

`syft-restrict` statically analyzes the **private** lines of a source file. The goal is to show
those lines only perform trusted math: no file or network access, no dynamic Python, no reaching
into interpreter internals. The code is never executed. You get a violation list (empty on success)
and, from `run()`, an obfuscated copy plus a certificate.

This page covers what's allowed, the order checks run in, a few non-obvious corners, and what the
checker cannot catch. For a flat rejection list see [blacklist.md](blacklist.md).

Like [RestrictedPython](https://github.com/zopefoundation/RestrictedPython), the model is
default-deny: if a node type, call, attribute, or name isn't explicitly permitted below, it's
rejected. New Python syntax stays blocked until someone reviews it.

---

## The whitelist

Private inference code needs to wire up JAX operations: Flax modules with `setup`/`__call__`, JAX
function calls, arithmetic, shape/dtype plumbing (lists, dicts, tuples, slices, f-strings for
einsum strings), and control flow that doesn't depend on secret data. It should not be able to touch
the host, filesystem, network, interpreter internals, or build code from strings.

A node passes only if it's on the always-on allow-list or enabled by per-file configuration. Everything
else fails.

### Always-on allow-list

These structural node types are always OK in the private region: module structure (`Module`, `Expr`);
definitions (`FunctionDef`, `ClassDef`, `arguments`, `arg`, `Return`, `Lambda`); names (`Name` with
`Load`/`Store`/`Del`); constants; assignment (`Assign`, `AugAssign`, `AnnAssign`); containers
(`List`, `Tuple`, `Dict`, `Set`); comprehensions; calls (`Call`, `keyword`, `Starred`); attribute
access (only `self.<name>`, see below); control flow (`If`, `For`, `While`, `Break`, `Continue`,
`Pass`, `IfExp`); and f-strings (`JoinedStr`, `FormattedValue`). The exact set is `_ALLOWED_NODES`
in `verifier.py`.

Extra constraints on top of "the node type is allowed":

- Class bases must be allow-listed module attrs (e.g. `nn.Module`), `object`, or another class
  defined in the private region. Decorators must be on the allow-list (`@nn.compact`, `@jax.jit`, …).
  Class keyword arguments like `metaclass=` are rejected. Magic/hook methods are limited to
  `setup`, `__call__`, and `__post_init__`.
- Attribute access on a value is not on this list. Only single-level `self.<name>` / `cls.<name>`
  reads and writes are allowed — the receiver is the class being defined, not some opaque value.
  Everything else (`x.shape`, any dunder read) is rejected and should go through a wrapper (see
  [Operator bundles on a value](#operator-bundles-on-a-value)).

### Per-file configuration

Two knobs the author sets per file. Anything not enabled here is rejected like any other disallowed node.

#### `allow_functions` — paths callable by name

Dotted paths are resolved against the file's import bindings. The denylist wins over the allowlist.

```python
import jax.numpy as jnp        # binding recorded: jnp -> jax.numpy
jnp.einsum("...", x)           # resolves to jax.numpy.einsum -> allow-listed  ✓
jnp.save(x)                    # resolves to jax.numpy.save   -> deny-listed   ✗
```

We don't allow-list whole modules. We allow-list `(module, attribute-path)` pairs, which works
because dynamic attribute access is banned. Rules that enforce this rule:

1. Resolve every call/attribute to a dotted path; it must match an allow pattern and not match a
   deny pattern.
2. Trusted module aliases (`jnp`, `nn`, …) can't be rebound — that would lie to the binding table.
   Reserved aliases can't be reassigned, used as parameters, or used as loop/comprehension targets.
3. Allow-listed callables must be called inline. An allow-listed path may appear only as the function
   of a `Call` — never stored, returned, or passed as an argument. That closes the "stash it in a
   variable, call later" escape.
4. Decorators must be on the allow-list (they run at def/class creation time).
5. Base classes must be allow-listed (class creation runs `__init_subclass__`/metaclass code).
6. Only `setup`, `__call__`, and `__post_init__` may be defined as magic/hook methods.

#### Operator bundles on a value

For `value.method(...)` we don't know the receiver's type, so we can't tell what a named method
does. Two rules handle this:

1. On an opaque value, only generic language-level operator methods are allowed — never a
   library-specific named method. These dunder operators are safe on any type and can't hide a
   `.format`-style escape. They're grouped into toggleable bundles:

   | Bundle (`allow_methods`) | Covers                              | Example                          |
   | ------------------------ | ----------------------------------- | -------------------------------- |
   | `arithmetic`             | `+ - * / // % ** @`, unary, bitwise | `var + 1e-6`, `x @ transpose(w)` |
   | `comparison`             | `== != < <= > >=`, `and`/`or`/`not` | `attn_type == "local"`           |
   | `indexing`               | subscript + slice                   | `x[..., :half]`, `cache[i]`      |

   There's no metadata bundle on purpose. `.shape`, `.ndim`, `.dtype` are named attribute reads on
   a value we can't type-pin, so they're rejected like any other attribute-on-value.

2. Library-specific methods and attribute reads go in visible wrapper functions — `.T`, `reshape`,
   `astype`, `x.at[i].set(v)`, a `.shape` read. The author writes the wrapper in the **public**
   region, where the data owner can read it and an `isinstance` guard pins the type the checker
   couldn't:

   ```python
   # public region — the data owner reads this
   def transpose(x):
       if not isinstance(x, jax.Array):   # explicit raise, NOT assert (asserts vanish under -O)
           raise TypeError
       return x.T
   ```

   Private code calls `transpose(w)`, not `w.T`. Wrapper names are reserved and can't be rebound in
   the private region.

#### The full call-target rule

A call target is allowed if it's one of:

- (a) an external allow-listed dotted path called inline;
- (b) a name defined and checked in the private region (e.g. the model's own `Attention(...)`);
- (c) a wrapper function from the public region;
- (d) for an operation on a value, a generic operator method from an enabled bundle.

No library-specific named method on a value is ever allowed.

---

## Order of operations

Verification runs before obfuscation:

```
run("file.py", obfuscate=[[84, 280]], allow_functions=["jax.*"])
  1. PARSE the full source (restrict sees everything; the data owner gets only the public part)
  2. RESOLVE imports -> binding table  (import jax.numpy as jnp  =>  jnp -> jax.numpy)
  3. WALK the private lines default-deny: node type allowed? call target allowed? resolved path not
     deny-listed? reserved alias not rebound? decorators/bases/dunder-defs allowed? no banned
     construct/name?   -- if ANY private node fails, restrict aborts and emits nothing.
  4. Only now: OBFUSCATE the private lines (rename identifiers, blank constants).
  5. EMIT the artifact (public glue + obfuscated math) plus a certificate.
```

Inside a trusted enclave ([TEE](https://en.wikipedia.org/wiki/Trusted_execution_environment)), a
passing run is evidence that this is a JAX inference pipeline with a hidden architecture that
doesn't steal its inputs.

---

## Edge cases

Subtle corners, kept here so code comments can stay short.

- **Annotation subtrees skip name/container checks.** Type annotations (`x: str`, `def f() -> list[bytes]`)
  aren't executed, so a name inside one isn't treated as a banned reference. The checker marks the
  whole annotation subtree, not just the top node, because generics nest type names several levels
  deep (`list[str]` is `Subscript(Name('list'), Name('str'))`). A `Call`/`Attribute` inside an
  annotation still gets checked — `x: evil()` is caught.

- **Comprehension targets are checked from the enclosing expression.** `ast.comprehension` has no
  line number, so a range check on that node never fires. The reserved-name check for
  `[cls for cls in ...]` runs from the enclosing `ListComp`/`SetComp`/`DictComp`/`GeneratorExp`.

- **`self`/`cls` trust is a lexical string match, not a verified binding.** `self.<name>` is
  trusted by matching the literal identifier — the checker never proves the name is bound to the
  real instance. That's sound only for the genuine first parameter of a method defined directly in a
  class body. So `self`/`cls` can't be rebound, used as a non-first parameter, or used as a
  nested-function/lambda parameter.

- **Self-attribute safety table.** `self.<name>(...)` and `self.layer[i](x)` are allowed only when
  `<name>` is inherited from the (already-vetted, allow-listed) base class or was assigned from a
  vetted-safe source everywhere in the class — an allow-listed constructor, a locally-defined
  class/function, or a list/tuple/comprehension of those. Compound assignment (`self.x += ...`)
  always disqualifies the attribute.

- **f-string conversion flags call dunders with no `Call` node.** `f"{x!r}"`, `f"{x!s}"`, `f"{x!a}"`,
  and `f"{x=}"` invoke `__repr__`/`__str__`/`__format__` on the value, but there's no `Call` node
  for the call checks to see — so they're rejected directly. Plain `f"{x}"` is fine.

- **Aliasing is caught at the reference, not the call.** A banned builtin (`open`, `eval`, …) is
  flagged wherever its name appears — aliased locally (`f = open`), stashed in a container
  (`{"o": open}`), returned, passed as an argument, or picked in an `IfExp` branch. Storing a
  banned reference in a container literal or subscript slot is rejected at construction time.

---

## What verification does not stop

The whitelist above closes the obvious vectors: no file/socket/host-callback/dynamic-code node can
pass. What's left are problems a static AST checker structurally can't solve — each needs a
separate control.

1. **The output is a leak channel.** Inference has to return logits or tokens; a malicious model can
   encode private data in them. Mitigate with output rate limits, quantization, DP noise, or a
   reviewed output schema.
2. **Timing and cache side channels.** Execution time and memory access patterns leak regardless of
   which ops ran. Handle at the enclave level.
3. **New dangerous symbols under an already-allowed prefix.** Default-deny blocks unknown new paths,
   but a bad addition inside `jax.numpy.*` is a gap until the denylist is updated. Keep the policy
   versioned and reviewed against each pinned JAX release.
4. **Bugs or supply-chain compromise in trusted libraries.** The checker only constrains caller
   code; a malicious jax/flax build defeats it. Attest exact library versions and hashes.

## Examples

### Build-and-run code from a string

- AST: `exec`, `eval`, `compile`, `__import__` as call targets
- Why: arbitrary code / import escape

### `Exec` / `eval`

- AST: py2 `Exec`; `Call` to `eval`
- Why: same as above

### Non-allowlisted imports

- AST: `Import`, `ImportFrom` to any module not in the allowlist
- Why: imports are how code names things outside itself — the crux of
  [per-file configuration](verify.md#per-file-configuration)

### Reflection / dunder

- AST: any `Name`/`Attribute`/`arg`/alias starting with `_`
- Why: the `__class__` → `__globals__` → `__subclasses__` → `__builtins__` ladder

### `getattr`/`setattr`/`delattr`/`hasattr`/`vars`/`globals`/`locals`/`dir`

- AST: `Call` to these names
- Why: dynamic attribute access defeats static path allowlisting

### `getattr`/`setattr` with a computed name

- AST: —
- Why: even a guarded `getattr` breaks static provability

### `.format` / any named method on a value

- AST: `Call` whose func is an `Attribute` on a non-allowlisted value, e.g.
  `"{0.__class__.__init__.__globals__}".format(x)`
- Why: format strings walk attributes at runtime without a visible `Call`; covered by
  [operator bundles on a value](verify.md#operator-bundles-on-a-value) — only generic operator
  bundles are allowed on a value

### Host I/O

- AST: `open`, real `print`, `input`, `file`
- Why: filesystem / stdout exfiltration

### OS / process

- AST: any `os`, `sys`, `subprocess`, `shutil`, `pathlib`, `socket`, `ssl`, `http`, `urllib`,
  `requests`, `ctypes`, `cffi`, `mmap`, `multiprocessing`, `threading`, `asyncio`, `signal` import
  or attr
- Why: direct exfiltration / native code / escape

### Pickle / marshal

- AST: `pickle`, `marshal`, `dill`, `shelve`, `joblib`
- Why: code execution on load + serialization exfiltration

### Global mutation

- AST: `Global`, `Nonlocal`; assignment to module-level names from inside functions
- Why: stashing data in module state for later read-out

### Attribute write to a foreign object

- AST: `Store` on `obj.<name>` where `obj` isn't `self` (e.g. `some_obj.send = data`)
- Why: only `self.<name>` writes are allowed; writing onto another object is an exfil channel

### Async

- AST: `AsyncFunctionDef`, `Await`, `AsyncFor`, `AsyncWith`, `Yield`, `YieldFrom`
- Why: concurrency/escape, not needed for inference

### Context managers

- AST: `With`
- Why: `open(...) as f`, `socket(...)` — none needed in pure inference

### Exceptions reaching the host

- AST: `Try`, `Raise`
- Why: `except` can swallow a guard or walk traceback frames — default deny

### Arbitrary decorators

- AST: `decorator_list` entries not in allowlist
- Why: a decorator is an arbitrary call wrapping the function — must be on the allowlist

### `@property` / descriptors

- AST: `@property` (and any descriptor) not on the allowlist
- Why: runs code on bare attribute access (`obj.w`) with no explicit call; pure inference needs
  only `setup`/`__call__`

### Loop/comprehension over I/O

- AST: `For`/`*Comp` whose iterable is a denied call
- Why: `[x for x in open(f)]` reintroduces I/O

### Denied call in a "passive" position

- AST: `Call` to a denied target in a default arg (`def f(x=evil())`), annotation
  (`def g(x: evil())`), or bare class-body statement (`class C: evil()`)
- Why: these run at def/class creation; the checker walks every node regardless of position

### `del` of guard names

- AST: `Delete`
- Why: could un-define a guard

### Unlisted modern syntax

- AST: walrus `NamedExpr` `(x := …)`; `Match`/`case`; `Assert`; `except*`/`TryStar`; PEP 695 type
  params (`def f[T]`, `type X = …`)
- Why: not on the allowlist, so default-deny rejects them; listed so reviewers know they were
  considered (asserts also vanish under `python -O`)
