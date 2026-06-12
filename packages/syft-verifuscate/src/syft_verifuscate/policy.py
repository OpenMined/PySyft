"""Policy: what the hidden region is allowed to call and do.

Two channels, mirroring the two verification mechanisms (see research approach-B §3.6.5):

- ``functions`` — dotted paths callable BY NAME (resolved exactly against the import bindings),
  e.g. ``jax.*``, ``flax.linen.*``. Checked by glob match, with ``JAX_DENYLIST`` beating the allow.
- ``methods``  — operator *bundles* allowed ON A VALUE, e.g. ``arithmetic``, ``indexing``. These are
  language-level operators (``__add__``, ``__getitem__``, …), never named library methods. No named
  method may be called on an opaque value at all.
"""

from __future__ import annotations

import ast
import fnmatch
from dataclasses import dataclass, field

# ── Operator bundles: bundle name -> the AST node types it enables on a value ──────────────
# These are generic, type-agnostic-safe operators (not named-method calls), so the format-string
# escape cannot hide among them (research approach-B §3.6.2).
OPERATOR_BUNDLES: dict[str, tuple[type[ast.AST], ...]] = {
    "arithmetic": (ast.BinOp, ast.UnaryOp),
    "comparison": (ast.Compare, ast.BoolOp),
    "indexing": (ast.Subscript, ast.Slice),
}
# The metadata bundle is special: it allows a few pure *metadata reads* on a value (ints/dtype,
# no side effects). Transforms like `.T` are NOT here — they're library-specific and must be wrapped
# (research approach-B §3.6.2/§3.6.3).
METADATA_ATTRS: frozenset[str] = frozenset({"shape", "ndim", "dtype", "size"})

ALL_BUNDLES: frozenset[str] = frozenset(OPERATOR_BUNDLES) | {"metadata"}

# ── Dangerous JAX / serialization surface — denylist BEATS the allow (approach-B §3.2/§3.3) ──
# Host-callback / IO / FFI / serialization escape hatches that can run host code or touch disk.
JAX_DENYLIST: tuple[str, ...] = (
    "jax.experimental.*",
    "jax.debug.*",
    "jax.pure_callback",
    "*.io_callback",
    "*.host_callback",
    "*.host_callback.*",
    "jax.profiler.*",
    "jax.monitoring.*",
    "jax.distributed.*",
    "jax.dlpack.*",
    "jax.ffi",
    "jax.ffi.*",
    "jax.extend.*",
    # array <-> file on disk, even though jax.numpy.* is otherwise allowed
    "jax.numpy.save",
    "jax.numpy.savez",
    "jax.numpy.savez_compressed",
    "jax.numpy.load",
    "jax.numpy.tofile",
    "jax.numpy.fromfile",
    "jax.numpy.memmap",
    "jax.numpy.savetxt",
    "jax.numpy.loadtxt",
    "jax.numpy.genfromtxt",
    "flax.serialization.*",
    "flax.training.checkpoints.*",
    "orbax.*",
)

# ── Builtins that are dynamic-escape / IO hatches and may never be called (approach-B §2.2) ──
BANNED_NAMES: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "vars",
        "globals",
        "locals",
        "dir",
        "open",
        "input",
        "breakpoint",
        "memoryview",
    }
)

# Decorators allowed above a def/class in the hidden region (approach-B §3.4 / §3.5.1 #4).
ALLOWED_DECORATORS: frozenset[str] = frozenset(
    {"nn.compact", "jax.jit", "jax.named_scope", "flax.linen.compact"}
)

# The only dunder/hook methods a model class may *define* (approach-B §3.5.1 #6).
ALLOWED_DUNDER_DEFS: frozenset[str] = frozenset({"__call__", "setup", "__post_init__"})

# Names always preserved verbatim by the obfuscator and never treated as opaque values.
DEFAULT_KEEP: frozenset[str] = frozenset(
    {"self", "cls", "nn", "Module", "setup", "__call__", "__post_init__"}
)


@dataclass
class Policy:
    """Parsed allow-lists. ``reserved`` is filled in by the verifier from the file's imports."""

    functions: list[str] = field(default_factory=list)
    methods: set[str] = field(default_factory=set)
    reserved: set[str] = field(default_factory=set)

    @classmethod
    def parse(cls, allow_functions: str = "", allow_methods: str = "") -> "Policy":
        functions = _split(allow_functions)
        methods = set(_split(allow_methods))
        unknown = methods - ALL_BUNDLES
        if unknown:
            raise ValueError(
                f"unknown method bundle(s): {sorted(unknown)}; allowed: {sorted(ALL_BUNDLES)}"
            )
        return cls(functions=functions, methods=methods)

    # ── path matching ──────────────────────────────────────────────────────────────────
    def function_allowed(self, dotted: str) -> bool:
        """True iff a fully-qualified dotted path is allowed (and not denylisted)."""
        if any(fnmatch.fnmatchcase(dotted, pat) for pat in JAX_DENYLIST):
            return False
        return any(_path_matches(dotted, pat) for pat in self.functions)

    def bundle_enabled(self, name: str) -> bool:
        return name in self.methods

    def policy_id(self) -> str:
        """A short, stable identifier for the policy (for the certificate)."""
        import hashlib

        blob = "|".join(sorted(self.functions)) + "##" + "|".join(sorted(self.methods))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _split(spec: str) -> list[str]:
    return [part.strip() for part in spec.split(",") if part.strip()]


def _path_matches(dotted: str, pattern: str) -> bool:
    """Match a dotted path against an allow pattern.

    ``jax.*`` matches ``jax`` and anything beneath it (``jax.numpy.einsum``); an exact pattern
    matches only itself.
    """
    if pattern.endswith(".*"):
        prefix = pattern[:-2]
        return dotted == prefix or dotted.startswith(prefix + ".")
    return fnmatch.fnmatchcase(dotted, pattern)
