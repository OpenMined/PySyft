# restrict

## 1. Overview

restrict is inspired by **RestrictedPython**: it **default-denies** every dynamic part of Python. It differs in a few ways:

- **Syft-restrict analyzes, it doesn't run.** syft-restrict statically analyzes a source file and **raises if the analyzed part uses any dynamic Python**. Its output is a **report** (the violations, or — on success — an attestation) plus an **obfuscated copy** of the code.
- **syft-restrict is currently tailored for analysis of machine learning inference code** 
- **Public / private split.** The user marks some lines of the file _private_ and leaves the rest _public_. restrict analyzes only the **AST of the private part**, and only those private lines are **hidden (obfuscated)** in the emitted artifact; the public lines are copied through and read directly.
- **Imports are allowed (and therefore trusted).** Unlike RestrictedPython, the analyzed code may **import allowlisted libraries** and call into them — which means the reader of the output **needs to trust those imported classes/functions** (only their allow-listed paths; see §3).
- **Dynamic Python lives in the public part.** Anything dynamic the author needs (file/tokenizer loading, the generation loop, wrappers around library methods) must be written in the **public** region — where it is read directly — and may be **called by** the private part.

## 2. Usage

```python
import syft_restrict as restrict

result = restrict.run(
    "gemma_inference.py",
    obfuscate=[[22, 93], [99, 280]],                              # 1-based ranges: identifiers renamed, constants blanked
    hide=[],                                                      # 1-based ranges: whole line replaced with ■■■■■■■■
    allow_functions=["jax.*", "flax.linen.*"],                    # things callable BY NAME (path-resolved)
    allow_methods=["arithmetic", "indexing", "comparison"],       # operators allowed ON A VALUE
)
# Verified region = obfuscate ∪ hide (both are private code; both are checked).
# On success: writes gemma_inference.obfuscated.py and returns result.certificate.
# On a policy violation: raises PolicyViolation naming each offending line (strict=True, the default).
```

This command reads **[examples/gemma_inference.py](examples/gemma_inference.py)** and generates **[examples/gemma_inference.obfuscated.py](examples/gemma_inference.obfuscated.py)**. If we can assume that syft-restrict was executed on the true inference file and the library was not modified, for instance because it was executed in a [TEE](https://en.wikipedia.org/wiki/Trusted_execution_environment), this proves:
1. this code is an inference pipelines for a jax model, where the model architecture is hidden in the report
2. this code does not steal the inputs

Use `restrict.verify(...)` for the check alone (it returns violations instead of raising), or pass `strict=False` to `run` to get a `RunResult` with `.ok` / `.violations` and no exception.

## 3. The whitelist (for the private region)

What the private code legitimately needs to do: **tie together JAX operations** — define Flax modules (classes with `setup`/`__call__`), call JAX functions, do arithmetic, build the shape/dtype plumbing (lists, dicts, tuples, slices, f-strings for einsum equations), and run data-independent control flow. Nothing that can reach the host machine, the filesystem, the network, the Python interpreter's internals, or build-and-run code from a string.

Default rule, exactly like RestrictedPython: **any node type not in the ALLOW column is REJECTED.** Future new Python syntax is denied until a human reviews it.

For **examples of disallowed constructs** (and why each is rejected), see **[disallowed-ast-examples.md](disallowed-ast-examples.md)**. Those examples are illustrative only — safety comes from the default-deny whitelist below, not from any deny list.

### 3.1 ALLOW — always-on whitelist

These structural nodes are always permitted in the private region. Beyond them, the author enables a **configurable set per-file** — allowlisted call/attribute _paths_ (`allow_functions`) and operator _bundles_ on values (`allow_methods`) — listed in **§3.2**. A node is permitted only if it appears here or in an _enabled_ §3.2 entry; everything else is rejected (default-deny).

| Category                          | Allowed `ast` nodes                                                | Example (from the private region of `gemma_inference.py`)                                                                                                                                            | Notes / constraints                                                                                                                                                                                                                                     |
| --------------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Module structure                  | `Module`, `Expr`                                                   | the file itself (`Module`); the module docstring `"""Gemma 3 IT — Flax Inference Module ..."""` is a bare-expression statement (`Expr`)                                                              | top level                                                                                                                                                                                                                                               |
| Definitions                       | `FunctionDef`, `ClassDef`, `arguments`, `arg`, `Return`, `Lambda`  | `class RMSNorm(nn.Module):` (`ClassDef`); `def __call__(self, x):` (`FunctionDef` + `arguments`/`arg`); `return x * jax.lax.rsqrt(...)` (`Return`); `lambda: None` in `_get(...)` (`Lambda`)         | class bases restricted to allowlisted module attrs (e.g. `nn.Module`); decorators restricted to allowlist (`@nn.compact`, `@jax.jit` — see §3.2.1)                                                                                                      |
| Names — `Name`                    | `Name`                                                             | any bare identifier: `x`, `half`, `cfg`, `Attention` (a variable, parameter, function, or class reference)                                                                                           | **underscore ban applies to the identifier text** — see the note below this row                                                                                                                                                                         |
| Names — read context `Load`       | `Load`                                                             | the `x` and `half` in `x * half` (every value _read_)                                                                                                                                                | not a node — it's the `ctx` flag on a `Name`/`Attribute`/`Subscript` marking a _read_; always allowed                                                                                                                                                   |
| Names — write context `Store`     | `Store`                                                            | the `half` in `half = dim // 2` (the name being _bound_)                                                                                                                                             | same `ctx` flag marking a _write_ (vs `Load`'s read); allowed only for a local `Name` or `self.<name>` — no global or foreign-attribute writes                                                                                                          |
| Constants                         | `Constant`                                                         | `2` and `1e-6` (numbers); `"bsd,ndh->bsnh"` (str literal); `None`, `True`, `False`                                                                                                                   | any literal scalar — pure, no constraint                                                                                                                                                                                                                |
| Assignment                        | `Assign`, `AugAssign`, `AnnAssign`                                 | `x = x + h` (`Assign`); `attn_type: str = "local"` (`AnnAssign`). _(`AugAssign` e.g. `h += ...` — not used in the private region, illustrative.)_                                                    | targets: local `Name` or `self.<name>`; **no global writes**, no attribute writes to foreign objects                                                                                                                                                    |
| f-strings                         | `JoinedStr`, `FormattedValue`                                      | none in the private region — the einsum equations here are plain `Constant` strings (`"bsd,ndh->bsnh"`). _(A real `JoinedStr` is the public `format_chat`'s `f"<start_of_turn>user\n{prompt}..."`.)_ | needed if einsum equations are built as f-strings; the format-string `__format__` trick must be neutralised — only allow formatting of `Constant`/`Name`/allowlisted exprs, and ban `_`-prefixed format specs (a historical RestrictedPython bug class) |
| Type hints                        | `Subscript`/`Name` in annotations                                  | `cfg: dict` (`Transformer`); `attn_type: str = "local"` (`Block`)                                                                                                                                    | `cfg: dict` etc.                                                                                                                                                                                                                                        |
| Containers                        | `List`, `Tuple`, `Dict`, `Set`                                     | `dict(num_layers=18, embed_dim=640, ...)` (`Dict`); `{"local": ..., "global": ...}` in `make_masks` (`Dict`); `("local",) * 5 + ("global",)` in `_attn_types` (`Tuple`)                              | literal data plumbing; elements/keys must themselves be allowed nodes — a denied call or attr inside (e.g. `[open(f)]`) still fails                                                                                                                     |
| Comprehensions                    | `ListComp`, `DictComp`, `SetComp`, `GeneratorExp`, `comprehension` | `self.layer = [Block(cfg=self.cfg, attn_type=attn_types[i]) for i in range(num_layers)]` (`ListComp` over `range(...)`)                                                                              | **only if** the iterable is a pure expression (`range(...)`, a literal, a local). Banned if the iterable touches I/O. No `async` comprehensions.                                                                                                        |
| Calls                             | `Call`, `keyword`, `Starred`                                       | `Attention(cfg=self.cfg)` (`Call` + `keyword` `cfg=`); `jnp.einsum("bsd,ndh->bsnh", x)` (`Call`). *(`Starred` e.g. `jnp.stack([*xs])` — not used here, illustrative.)\*                              | the function being called **must resolve to an allowlisted symbol** (§3.2.1); `Starred` allowed only into allowlisted calls                                                                                                                             |
| Attribute — `self.<name>` (read)  | `Attribute` (`Load`)                                               | `self.scale`, `self.cfg`, `self.layer`                                                                                                                                                               | reading the module's own attributes — receiver is the class being defined, not an opaque value, so always safe. (Reads _on a value_, even `x.shape`, are **denied** — route them through a visible wrapper, §3.2.)                                      |
| Attribute — `self.<name>` (write) | `Attribute` (`Store`)                                              | `self.w = ...` inside `setup`                                                                                                                                                                        | the **only** attribute write allowed; Flax needs it. Writes to any other object (`obj.x = ...`) are denied                                                                                                                                              |
| Control flow                      | `If`, `For`, `While`, `Break`, `Continue`, `Pass`, `IfExp`         | `for i in range(num_layers):` (`For`); `if cache is None:` (`If`); `cache[i] if cache is not None else None` (`IfExp`)                                                                               | data-independent loops fine; `For` iterable restricted like comprehensions                                                                                                                                                                              |

> **The underscore rule (carried over from RestrictedPython) — the most delicate part of the policy.** Any identifier starting with `_` — the `id` of a `Name`, the `attr` of an `Attribute`, an `arg` name, or an import `alias` — is in principle rejected; this single rule kills the introspection escape ladder (`obj.__class__.__bases__[0].__subclasses__()` …). But the private model uses leading-underscore names heavily (`_get`, `_attn_types`, `_query_norm`, `__call__`), so a **curated relaxation** is required: a leading-underscore identifier is **rejected** when it is (a) **read off a foreign object** (`obj._secret`, `obj.__class__`) or (b) used as a **call target** (`_f(...)`), but **allowed** when it is (c) a _local definition_ (`def _get(...)`, `_tmp = ...`) or (d) a `self.`-attribute (`self._query_norm`, `self._key_norm`). (Defining `_`/dunder **hook methods** on a model class is separately restricted to `setup` / `__call__` / `__post_init__` — §3.2.1 #6.) This carve-out is where review effort concentrates.

> **Attribute — what is NOT allowed (the two `self.<name>` rows above, plus the allowlisted-path row in §3.2, are the whole allowed set).** If a read doesn't match one of those, it's denied. Denied **reads** (`Load`), with examples:
>
> - **A dotted path that resolves to a non-allowlisted symbol** — `jnp.save`, `jax.experimental.io_callback`, `jax.numpy.load` (denylist beats allow, §3.2.1).
> - **Any attribute read on an opaque value** — `x.shape`, `x.dtype`, `embed_table.T`, `x.real`, `x.device` (we can't pin the receiver's type, so _every_ attribute name on a value is denied — including `.shape`/`.ndim`/`.dtype`; route each through a visible wrapper function, §3.2).
> - **Any underscore/dunder attribute read off a foreign object** — `obj._secret`, `x.__class__`, `module.__dict__` (the reflection ladder; §3.1).
>
> And the one denied **write** (`Store`): an attribute write to any non-`self` object — `obj.send = data`.

> **The one allowed-syntax exception — class keyword arguments (`metaclass=`).** A `keyword` node is on the allow-list (Calls row) for ordinary call kwargs like `Attention(cfg=...)`. But a `keyword` attached to a **`ClassDef`** — `class C(metaclass=evil)`, or _any_ class keyword — is **denied**, because building a class with a `metaclass=` (or other class kwarg) runs attacker-chosen code at class-creation time. So the rule is _position-dependent_: the same `keyword` node is allowed inside a `Call` but rejected inside a `ClassDef`. (Base classes are separately restricted to allow-listed symbols — see the Definitions row and §3.2.1 #5.)

### 3.2 Configurable to allow

Two channels the author sets per-file. Anything not enabled in one of them is rejected like any other non-allowlisted node.

**Paths called/read by name — `allow_functions` (§3.2.1).** Dotted paths resolved exactly against the import bindings; the denylist beats the allow.

| Form                           | AST node             | Example (from the private region)                                    | Constraint                                                                                                                 |
| ------------------------------ | -------------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Allowlisted module path (read) | `Attribute` (`Load`) | `jnp.einsum`, `jax.lax.rsqrt` — the dotted path in front of a `Call` | allowed only when the **whole path resolves to an allowlisted symbol** (§3.2.1); the underscore ban applies to each `attr` |

**Operator bundles on a value — `allow_methods` (§3.2.2).** Type-agnostic-safe generic operators, never library-specific named methods — so each can be toggled on without re-opening an escape.

| Bundle (`allow_methods`) | AST nodes / operators                                                                                                           | Example (from the private region)                                                                                             |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `arithmetic`             | `BinOp`, `UnaryOp`; operators `Add Sub Mult Div FloorDiv Mod Pow MatMult LShift RShift BitOr BitAnd BitXor`, `USub UAdd Invert` | `var + 1e-6` (`Add`); `base_freq ** freq_exp` (`Pow`); `x @ transpose(embed_table)` (`MatMult`); `causal & window` (`BitAnd`) |
| `comparison`             | `Compare`, `BoolOp`; operators `Eq NotEq Lt LtE Gt GtE Is IsNot In NotIn`, `And Or`, `Not`                                      | `attn_type == "local"` (`Eq`); `cache is None` (`Is`); `causal and window` (`And`)                                            |
| `indexing`               | `Subscript`, `Slice`                                                                                                            | `x[..., :half]` (`Subscript` + `Slice`); `kv[0]`, `cache[i]`, `shape_of(x)[-1]`                                               |

#### 3.2.1 Constraining how allow-listed functions and classes are used

The private region still imports and calls JAX, and **an import is the only way the code can name anything outside its own AST** — while _most JAX is safe pure math, a handful of functions can reach the host_. So once §3.2 has said _which_ functions and methods may be named, these rules fix **how those allow-listed names may then be used** — so nothing defeats the path resolver or smuggles execution into a `def`/`class`. Rule 1 is the resolution mechanism itself; rules 2–6 keep it sound. (**Rebinding**, below, means pointing a name at a different object — `jnp = something_else` — innocuous normally, but a weapon here because the resolver assumes `jnp` still means the allow-listed module.)

**1. We resolve every call/attribute to a dotted path and require it to be allow-listed (and not deny-listed).**

```python
import jax.numpy as jnp        # binding recorded: jnp → jax.numpy
jnp.einsum("...", x)           # resolves to jax.numpy.einsum → allow-listed ✓
jnp.save(x)                    # resolves to jax.numpy.save  → deny-listed ✗
```

We don't allow-list a module wholesale; we allow-list **(module, attribute-path)** pairs, enforced by _reading_ the code — which works precisely because dynamic attribute access is banned. Mechanism: (a) `Import`/`ImportFrom` may only name allow-listed modules, and each binding is recorded (`jnp → jax.numpy`); (b) every `Attribute`/`Call` is traced to a fully-qualified dotted path via those bindings; (c) the resolved path must match an allow pattern (`jax.numpy.*`, `jax.lax.*`, …) **and not** match a deny pattern — _the denylist beats the allow_, even inside an otherwise-allowed module (so `jax.numpy.einsum` passes while `jax.numpy.save` is rejected). Because there's no `getattr`/`exec`/underscore-reflection, this static resolution is **sound**: there is no other runtime path from `jnp` to a denied symbol. The dangerous-API denylist (host callbacks, `numpy.save/load`, FFI/dlpack, profiler/distributed, serialization, …) is catalogued in **[../../research/verifuscate/approach-B-jax-denylist-reference.md](../../research/verifuscate/approach-B-jax-denylist-reference.md)**.

**2. We deny reassigning a trusted module name (`jnp`, `nn`, `lax`, …).**

```python
jnp = SomethingEvil()
jnp.save(x)        # checker sees allow-listed path "jnp.save" → approves
                   # at runtime jnp is the evil object, .save() runs attacker code
```

The static path resolver (rule 1) trusts the binding table built from the imports. If the code later _rebinds_ `jnp`, the table would be a lie and every "allow-listed path" through that name attacker-controlled — so this vector would defeat the path allow-list if left unaddressed.
_Mitigation:_ treat the imported-module alias names as **RESERVED**. The reserved set is exactly the aliases the import resolver recorded — `jax`, `jnp` (`jax.numpy`), `lax` (`jax.lax`), `nn` (`flax.linen`), `jrandom`/`random` as imported, `functools`, etc. A reserved name may **not** be reassigned, used as a function/lambda parameter, used as a `for`/comprehension/`with` target, or otherwise rebound _anywhere_ in the private region. Any node that would store to a reserved name ⇒ REJECT. (This is cheap to enforce: it's just "is this `Store`/`arg`/loop-target one of the reserved names?")

**3. We deny referencing an allow-listed callable without calling it inline.**

```python
f = jnp.einsum        # path resolver sees a *reference*, not a Call
...
f(equation, x)        # later call goes through plain Name `f` — unchecked
d = {"s": jnp.save}; d["s"](x)   # same trick via a container
```

The rule-1 resolver only follows _literal dotted paths that are the function of a Call_. Pull an allow-listed callable out into a variable, list, or dict and the eventual `f(...)` is just a call on an ordinary local name — outside the path allow-list entirely.
_Mitigation:_ **require allow-listed callables to be called inline.** An allow-listed attribute path (`jnp.einsum`, `jax.lax.rsqrt`, …) may appear _only_ as the function position of a `Call` — never referenced and stored, returned, or passed as an argument. `jnp.einsum(...)` is fine; `g = jnp.einsum`, `return jnp.einsum`, `vmap(jnp.einsum)` (passing it as an arg) are all REJECT. This keeps every allow-listed call statically resolvable at its call site.

**4. We deny any decorator that isn't on the allow-list.**

```python
@evil            # `evil` is called when the def/class is created, before any "math" runs
def block(...): ...
@a.b(...)        # a.b(...) is called, then its result is called on the function
class Block(...): ...
```

A decorator is an ordinary call that executes the instant the `def`/`class` is reached.
_Mitigation:_ **allow-list the decorators** — `@nn.compact`, the `@jax.jit` family (incl. `@functools.partial(jax.jit, …)` when `functools.partial` is allow-listed and its first arg resolves to an allow-listed transform), `@jax.named_scope` (review). Any decorator expression not on that list ⇒ REJECT. The decorated body is still fully walked, so an allow-listed decorator can't smuggle denied ops.

**5. We deny non-allow-listed base classes.**

```python
class Block(EvilBase): ...          # EvilBase.__init_subclass__ / metaclass runs at class creation
```

Creating a class consults its bases — `EvilBase.__init_subclass__` (and any metaclass it carries) runs attacker code.
_Mitigation:_ **base classes must be allow-listed** (`nn.Module`, `object`); a `ClassDef` with a non-allow-listed base ⇒ REJECT. (The related `metaclass=` / class-keyword route is handled as a syntax exception in §3.1 — a `keyword` on a `ClassDef` is denied.)

**6. We deny defining magic/hook methods on a model class (only `setup`/`__call__` allowed).**

```python
class Block(nn.Module):
    def __getattr__(self, name): ...        # runs on every attribute miss
    def __init_subclass__(cls, **kw): ...   # runs at class creation
    def __reduce__(self): ...               # runs at pickle time
    def __getitem__(self, k): ...           # runs on subscripting
```

These are hooks: Python calls them automatically at attribute-access, class-creation, pickling, or subscript time. The body is attacker code that fires _without an explicit call in the math_ — so even a clean-looking `block[0]` could detonate.
_Mitigation:_ inside a class body, **only allow defining the handful of method names Flax actually needs** — `setup`, `__call__`, and `__post_init__` if Flax requires it. Defining _any other_ underscore/dunder method ⇒ REJECT. (Note this is stricter than the §3.1 underscore relaxation, and it should be: §3.1 lets you _read_ `self._foo` and _define_ `_foo`-style helpers; this rule additionally forbids _defining magic hook methods_ on a model class.)

#### 3.2.2 The method-call gap: constraining `value.method(...)` calls on returned objects

For a call of the form `value.method(...)` we do **not** know the receiver's type — so, unlike a resolved `module.func(...)` path (§3.2.1), we cannot tell what invoking a method (or a `getattr`) on it actually does:

```python
x.reshape(8, -1)   # x's type is unknown — is this jax.Array.reshape, or a .format-style escape?
```

**The fix: only generic operator _methods_ on values; library ops become _functions_.**

Two rules:

1. **On an opaque value, allow only generic, language-level operator methods — never a library-specific named method.** These are the dunder operators (`__add__`, `__getitem__`, comparisons, …): they're safe on _any_ type, and — crucially — they aren't named-method calls, so the `.format`-style escape has nowhere to hide. They are enabled as toggleable **bundles**:

   | Bundle (pass in `allow_methods`) | Covers                                 | Example from the private region      |
   | -------------------------------- | -------------------------------------- | ------------------------------------ | ------------------------------------------ |
   | `arithmetic`                     | `+ - * / // % ** @`, unary, bitwise `& | ^ ~`                                 | `var + 1e-6`, `x @ transpose(embed_table)` |
   | `indexing`                       | subscript + slice                      | `x[..., :half]`, `kv[0]`, `cache[i]` |
   | `comparison`                     | `== != < <= > >=`, `and`/`or`/`not`    | `attn_type == "local"`               |

   There is deliberately **no metadata bundle**: a `.shape`/`.ndim`/`.dtype` read is a _named attribute access_ on a value whose type we can't pin (it could be the `.format`-style escape just as easily as a real array), so it is treated like any other attribute-on-value and rejected — it must be wrapped (rule 2).

2. **Every library-specific method or attribute read is extracted into a visible wrapper function** — `.T`, `reshape`, `astype`, `sum`, `x.at[i].set(v)`, a Flax `Variable.value`, a `.shape`/`.ndim`/`.dtype` read. Instead of `x.transpose()` in the private code, the author writes the wrapper in the **public** region and the private code calls it _by name_; the data owner reads it, where an `isinstance` guard pins the receiver type the checker couldn't:

   ```python
   # in the public (non-private) region — the data owner reads this
   def transpose(x):
       if not isinstance(x, jax.Array):     # explicit raise, NOT assert (asserts vanish under python -O)
           raise TypeError
       return x.T
   ```

   So the private code writes `transpose(embed_table)`, not `embed_table.T`. The wrapper names are **reserved** — the private region can't rebind them (§3.2.1 #2), so `transpose = evil` is rejected. (Optionally, restrict template-checks each wrapper — a type guard plus a single delegated call — so they're machine-checked, not only read.)

Net effect: **no named method is ever called on an opaque value.** The only thing done _to a value_ is a generic operator; everything library-specific is a named, type-guarded function.

#### 3.2.3 The full call-target rule

Combining the §3.2.1 rules with the method rules above, what may sit in a `Call`'s function position (or be done to a value) is a **four-way OR**:

A call target is allowed iff it is one of:

- **(a)** an external allow-listed dotted path called inline (§3.2.1 — a _function_);
- **(b)** a name defined-and-checked in the private region (transitively safe — e.g. `Attention(...)`, `apply_rope(...)`);
- **(c)** a **wrapper function defined in the public region** (read by the DO, its name reserved against rebinding);
- **(d)** — for an operation _on a value_ — a generic operator **method** from an enabled bundle (§3.2.2).

**No library-specific named method on a value is ever allowed.**

Anything else ⇒ REJECT. Cases (b) keep the rule from being too strict: the model's own classes/functions are calls on names defined _inside_ the private lines, so calling them can't introduce anything the checker hasn't already walked.

## 4. Order of operations — verify the private lines BEFORE obfuscating them

```
restrict.run("file.py", obfuscate=[[84, 280]], hide=[], allow_functions=["jax.*"])
   │
   ├─ 1. PARSE the full source  ──► ast.parse(source)
   │       (restrict sees everything; the DO will only get the public part)
   │
   ├─ 2. RESOLVE imports → binding table  (import jax.numpy as jnp ⇒ jnp→jax.numpy)
   │
   ├─ 3. WALK the PRIVATE lines (84–280) with a default-deny visitor:
   │        • node type ∈ ALLOW set (§3.1)?           else REJECT
   │        • every Name/Attribute/arg/alias: underscore rule (§3.1)
   │        • Call target ∈ {external allowlisted path called INLINE,
   │             locally-defined-and-checked name, allowlisted method
   │             name on a value} (§3.2, §3.2.3)?         else REJECT
   │        • resolved external path ∉ denylist (§3.2)?   else REJECT
   │        • reserved module-alias names never rebound (§3.2.1#2)? else REJECT
   │        • decorators/bases/class-keywords/own-dunder-methods
   │             ∈ allowlist (§3.2.1#4-6)?         else REJECT
   │        • no Global/Nonlocal/foreign-attr-Store    else REJECT
   │        • no exec/eval/getattr/open/try/with/...    else REJECT
   │        (call-checking is POSITION-INDEPENDENT: defaults,
   │         annotations, class-body stmts are walked too)
   │   ── if ANY private node fails ⇒ restrict ABORTS, emits nothing.
   │
   ├─ 4. Only now: MANGLE allowlisted jax calls per allow_functions,
   │        and OBFUSCATE lines 84–280 → ■■■■■■.
   │
   └─ 5. EMIT (artifact = public glue + obfuscated math,
              attestation{source_hash, policy_hash, PASS}).
```

"Attestation" (step 5) is a cryptographically signed statement the secure enclave produces: a machine-verifiable receipt saying _"I ran exactly this check over exactly these bytes and it passed."_ The DO checks the signature instead of trusting anyone's word.

The enclave runs steps 1–5 and attests: _"I, the enclave, parsed bytes with sha256 = H, ran whitelist policy version P over the PRIVATE lines, it PASSED, and the obfuscated artifact you're reading was derived from exactly those bytes."_

## 5. What this whitelist does NOT stop

Everything in §3 is a _closed_ vector: no file/socket/host-callback/dynamic-code node can pass, including all the subtle dynamic-Python escapes of §3.2.1. The problems below are the ones the static whitelist structurally cannot solve. Each needs an orthogonal control (output discipline, a runtime sandbox, resource bounds, attestation) — none is fixed by tightening the AST checker.

1. **The output is itself a leak channel.** Inference must return logits/tokens, and a malicious model can encode private data in them (low-order bits, token choices); "only pure math ran" doesn't imply "the output carries no private bits." → output rate-limiting / quantization / DP-noise / reviewed schema.
2. **Timing & cache side channels.** Execution time, memory-access and CPU-cache patterns leak information regardless of which ops ran; the whitelist can't enforce constant-time execution. → enclave-level side-channel mitigations.
3. **A future JAX host-callback API under an already-allowed prefix.** Default-deny closes unknown _new_ paths automatically, but a newly added dangerous symbol inside an allowed prefix (e.g. a hypothetical `jax.numpy.<new_io>`) is a gap until the denylist is updated. → keep the policy versioned and reviewed against each pinned JAX release.
4. **Bugs or supply-chain compromise in the trusted libraries.** The model only constrains the _caller's_ code; a malicious jax/flax/orbax build defeats it. → attest the exact library versions/hashes; the DO must trust those releases.
