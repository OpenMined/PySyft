"""Policy: what the private region is allowed to call and do (see docs/verify.md).

Two channels the author configures per file:

- ``functions`` — dotted paths callable BY NAME (resolved against import bindings),
  e.g. ``jax.*``, ``flax.linen.*``. An optional *disallow* list beats the allow.
- ``operators`` — operator *bundles* allowed ON A VALUE, e.g. ``arithmetic``, ``indexing``.
  These are language operators (``+``, ``[]``, …), never named library methods on a value.
"""

from __future__ import annotations

import ast
import fnmatch
import hashlib

from pydantic import BaseModel, Field

# ── Operator bundles: bundle name -> the AST node types it enables on a value ──────────────
# These are generic, type-agnostic-safe operators (not named-method calls), so the format-string
# escape (`"{0.__class__}".format(x)`) cannot hide among them (docs/verify.md#operator-bundles-on-a-value).
OPERATOR_BUNDLES: dict[str, tuple[type[ast.AST], ...]] = {
    "arithmetic": (ast.BinOp, ast.UnaryOp),
    "comparison": (ast.Compare, ast.BoolOp),
    "indexing": (ast.Subscript, ast.Slice),
}
# NOTE: there is deliberately no metadata bundle. Reads like `.shape`/`.ndim`/`.dtype` on an opaque
# value are named attribute accesses we can't pin to a type, so they're rejected like any other
# attr-on-value and must be routed through a public wrapper function (docs/verify.md#operator-bundles-on-a-value).
ALL_BUNDLES: frozenset[str] = frozenset(OPERATOR_BUNDLES)

# ── Builtins that are dynamic-escape / IO hatches and may never be called (docs/blacklist.md) ──
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
        "type",
        "__build_class__",
        "print",
        "repr",
        "str",
        "ascii",
        "format",
        "bytes",
        # site-injected builtins: stdout channels, interpreter shutdown, interactive help
        "copyright",
        "credits",
        "license",
        "exit",
        "quit",
        "help",
    }
)

# Builtins safe to call directly by bare name: pure, deterministic, no reflection/IO/dynamic-code
# surface. A bare-name call is default-deny (docs/verify.md#the-full-call-target-rule) -- everything
# NOT on this list must instead resolve to an import binding, a def/class defined in this file, or a
# local traced to one of those (see verifier._check_call / _is_safe_local_source).
SAFE_BUILTIN_CALLS: frozenset[str] = frozenset(
    {
        "int",
        "float",
        "bool",
        "len",
        "range",
        "enumerate",
        "zip",
        "min",
        "max",
        "sum",
        "abs",
        "round",
        "all",
        "any",
        "tuple",
        "list",
        "dict",
        "set",
        "sorted",
        "reversed",
        "isinstance",
        "super",
    }
)


# The only dunder/hook methods a model class may *define* (docs/verify.md#allow_functions--paths-callable-by-name).
ALLOWED_DUNDER_DEFS: frozenset[str] = frozenset({"__call__", "setup", "__post_init__"})

# Names always preserved verbatim by the obfuscator and never treated as opaque values.
DEFAULT_KEEP: frozenset[str] = frozenset(
    {"self", "cls", "nn", "Module", "setup", "__call__", "__post_init__"}
)


class Policy(BaseModel):
    """Parsed allow-lists.

    ``reserved_names`` is filled by ``verify()`` from the file's import bindings (on a copy of
    the policy — the caller's instance is not mutated).
    """

    # allowed functions passed by the user, example: ["jax.*", "flax.linen.*"]
    allowed_functions: list[str] = Field(default_factory=list)

    # operator bundles passed by the user, example: ["arithmetic", "indexing", "comparison"]
    allowed_operators: set[str] = Field(default_factory=set)

    # optional disallow globs supplied by the user; these beat the allow, example: ["jax.numpy.save"]
    disallowed_functions: list[str] = Field(default_factory=list)

    # when False, a local assigned a safe callable is no longer trusted as a bare-name call target
    # (disables _track_safe_local); the callee must instead be called directly (docs/verify.md).
    allow_local_assignments: bool = True

    # when False, a `self.<attr>` never assigned in the class body is NOT presumed inherited-safe;
    # only attributes the class assigns a vetted source may be called (docs/verify.md).
    allow_base_class_attributes: bool = True

    # import aliases reserved against rebinding (set per-file by verify), e.g. {"jnp", "nn"}
    reserved_names: set[str] = Field(default_factory=set)

    @classmethod
    def parse(
        cls,
        allow_functions: list[str] | None = None,
        allow_operators: list[str] | None = None,
        disallow_functions: list[str] | None = None,
        allow_local_assignments: bool = True,
        allow_base_class_attributes: bool = True,
    ) -> "Policy":
        allowed_functions = _clean(allow_functions)
        allowed_operators = set(_clean(allow_operators))
        disallowed_functions = _clean(disallow_functions)
        unknown = allowed_operators - ALL_BUNDLES
        if unknown:
            raise ValueError(
                f"unknown operator bundle(s): {sorted(unknown)}; allowed: {sorted(ALL_BUNDLES)}"
            )
        return cls(
            allowed_functions=allowed_functions,
            allowed_operators=allowed_operators,
            disallowed_functions=disallowed_functions,
            allow_local_assignments=allow_local_assignments,
            allow_base_class_attributes=allow_base_class_attributes,
        )

    # ── path matching ──────────────────────────────────────────────────────────────────
    def function_allowed(self, dotted: str) -> bool:
        """True if a fully-qualified dotted path is allowed (and not disallowed).

        An optional user-supplied ``disallowed_functions`` glob beats the allow, so an author can
        keep a hard floor over a broad allow like ``jax.*``. Empty (the default) => pure allow-list.
        """
        if any(fnmatch.fnmatchcase(dotted, pat) for pat in self.disallowed_functions):
            return False
        return any(_path_matches(dotted, pat) for pat in self.allowed_functions)

    def bundle_enabled(self, name: str) -> bool:
        return name in self.allowed_operators

    def policy_id(self) -> str:
        """A short, stable identifier for the policy (for the certificate)."""
        blob = (
            "|".join(sorted(self.allowed_functions))
            + "##"
            + "|".join(sorted(self.allowed_operators))
            + "##"
            + "|".join(sorted(self.disallowed_functions))
            + "##"
            + f"local={int(self.allow_local_assignments)}|baseattr={int(self.allow_base_class_attributes)}"
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _clean(items: list[str] | None) -> list[str]:
    return [s.strip() for s in (items or []) if s.strip()]


def _path_matches(dotted: str, pattern: str) -> bool:
    """Match a dotted path against an allow pattern.

    ``jax.*`` matches ``jax`` and anything beneath it (``jax.numpy.einsum``); an exact pattern
    matches only itself.
    """
    if pattern.endswith(".*"):
        prefix = pattern[:-2]
        return dotted == prefix or dotted.startswith(prefix + ".")
    return fnmatch.fnmatchcase(dotted, pattern)
