"""The static checker (research approach B).

``verify(source, private, policy)`` parses the file, restricts attention to the *private* line ranges
(the hidden model definition), and walks those nodes default-deny: only explicitly-allowed node types,
calls, operators, and attribute reads pass. It never raises on a policy issue — it returns a
``VerifyResult`` with the violations so callers can inspect them.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from .policy import (
    ALLOWED_DECORATORS,
    ALLOWED_DUNDER_DEFS,
    BANNED_NAMES,
    METADATA_ATTRS,
    OPERATOR_BUNDLES,
    Policy,
)

# ── allowed AST node types in the hidden region (approach-B §2.1) ────────────────────────────
_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Module,
    ast.Expr,
    ast.FunctionDef,
    ast.ClassDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.Lambda,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Del,
    ast.Constant,
    ast.Call,
    ast.keyword,
    ast.Starred,
    ast.Attribute,
    ast.Subscript,
    ast.Slice,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.comprehension,
    ast.If,
    ast.For,
    ast.While,
    ast.Break,
    ast.Continue,
    ast.Pass,
    ast.IfExp,
    ast.Assign,
    ast.AugAssign,
    ast.AnnAssign,
    ast.JoinedStr,
    ast.FormattedValue,
    # operator/cmpop/boolop/unaryop singletons are leaf nodes under the above; always fine.
    ast.operator,
    ast.cmpop,
    ast.boolop,
    ast.unaryop,
    ast.expr_context,
)

# Banned statement/expr node types (approach-B §2.2): present => violation.
_BANNED_NODES: tuple[type[ast.AST], ...] = (
    ast.Import,
    ast.ImportFrom,
    ast.With,
    ast.Try,
    ast.Raise,
    ast.Global,
    ast.Nonlocal,
    ast.Delete,
    ast.Assert,
    ast.AsyncFunctionDef,
    ast.AsyncFor,
    ast.AsyncWith,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
)


@dataclass(frozen=True)
class Violation:
    line: int
    code: str
    message: str


@dataclass
class VerifyResult:
    ok: bool
    violations: list[Violation] = field(default_factory=list)
    n_calls_checked: int = 0


@dataclass
class FileScan:
    """Names harvested from the whole file, used to classify calls in the hidden region."""

    bindings: dict[str, str]  # alias -> fully-qualified module path (jnp -> jax.numpy)
    hidden_defs: set[str]  # class/func names defined inside the private region
    visible_defs: set[
        str
    ]  # function names defined in the visible region (the wrappers)


def verify(source: str, private, policy: Policy) -> VerifyResult:
    ranges = _normalize_ranges(private)
    tree = ast.parse(source)
    scan = _scan_file(tree, ranges)
    policy.reserved = set(scan.bindings)
    checker = _Checker(policy, scan, ranges)
    checker.visit(tree)
    return VerifyResult(
        ok=not checker.violations,
        violations=checker.violations,
        n_calls_checked=checker.n_calls,
    )


# ── file scan ────────────────────────────────────────────────────────────────────────────
def _scan_file(tree: ast.Module, ranges) -> FileScan:
    bindings: dict[str, str] = {}
    hidden_defs: set[str] = set()
    visible_defs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bindings[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                bindings[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if _in_ranges(node, ranges):
                hidden_defs.add(node.name)
            elif isinstance(node, ast.FunctionDef):
                visible_defs.add(node.name)
    return FileScan(
        bindings=bindings, hidden_defs=hidden_defs, visible_defs=visible_defs
    )


# ── the checker ──────────────────────────────────────────────────────────────────────────
class _Checker:
    def __init__(self, policy: Policy, scan: FileScan, ranges):
        self.policy = policy
        self.scan = scan
        self.ranges = ranges
        self.violations: list[Violation] = []
        self.n_calls = 0
        self._call_funcs: set[int] = (
            set()
        )  # Attribute nodes already judged as a call func

    def add(self, node: ast.AST, code: str, message: str) -> None:
        self.violations.append(Violation(getattr(node, "lineno", 0), code, message))

    def visit(self, node: ast.AST) -> None:
        """Walk the tree; enforce only on nodes inside the private ranges, recurse everywhere."""
        if _in_ranges(node, self.ranges):
            self._enforce(node)
        for child in ast.iter_child_nodes(node):
            self.visit(child)

    # — per-node enforcement (recursion is handled by visit) —
    def _enforce(self, node: ast.AST) -> None:
        if isinstance(node, _BANNED_NODES):
            self.add(
                node,
                "banned-construct",
                f"{type(node).__name__} is not allowed in the hidden region",
            )
            return
        if not isinstance(node, _ALLOWED_NODES):
            self.add(
                node,
                "node-type",
                f"{type(node).__name__} is not on the node-type allow-list",
            )
            return

        if isinstance(node, ast.FunctionDef):
            self._check_def(node)
        elif isinstance(node, ast.ClassDef):
            self._check_class(node)
        elif isinstance(node, ast.Call):
            self._check_call(node)
        elif isinstance(node, ast.Attribute):
            self._check_attribute(node)
        elif isinstance(node, (ast.BinOp, ast.UnaryOp)):
            self._require_bundle(node, "arithmetic")
        elif isinstance(node, (ast.Compare, ast.BoolOp)):
            self._require_bundle(node, "comparison")
        elif isinstance(node, (ast.Subscript, ast.Slice)):
            self._require_bundle(node, "indexing")
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            self._check_assign_targets(node)
        elif isinstance(node, ast.For):
            self._check_reserved_target(node.target)
        elif isinstance(node, ast.comprehension):
            self._check_reserved_target(node.target)
        elif isinstance(node, ast.arg):
            self._check_reserved_name(node, node.arg)

    # — defs / classes —
    def _check_def(self, node: ast.FunctionDef) -> None:
        self._check_decorators(node)
        if _is_dunder(node.name) and node.name not in ALLOWED_DUNDER_DEFS:
            self.add(
                node,
                "dunder-def",
                f"defining magic method {node.name!r} is not allowed",
            )

    def _check_class(self, node: ast.ClassDef) -> None:
        self._check_decorators(node)
        if node.keywords:
            self.add(
                node,
                "class-keyword",
                "class keyword arguments (e.g. metaclass=) are not allowed",
            )
        for base in node.bases:
            dotted = _dotted(base)
            ok = (dotted and self._resolved_allowed(dotted)) or (
                isinstance(base, ast.Name)
                and base.id in (self.scan.hidden_defs | {"object"})
            )
            if not ok:
                self.add(
                    base,
                    "class-base",
                    f"base class {_describe(base)!r} is not allow-listed",
                )

    def _check_decorators(self, node) -> None:
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            dotted = _dotted(target)
            resolved = self._resolve(dotted) if dotted else None
            if not (resolved in ALLOWED_DECORATORS or dotted in ALLOWED_DECORATORS):
                self.add(
                    dec,
                    "decorator",
                    f"decorator {_describe(target)!r} is not allow-listed",
                )

    # — calls —
    def _check_call(self, node: ast.Call) -> None:
        self.n_calls += 1
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in BANNED_NAMES:
                self.add(node, "banned-call", f"call to {func.id!r} is not allowed")
            # Otherwise a bare-name call (local var / hidden or visible def / safe builtin) is allowed;
            # nothing dangerous can reach a local name given the other rules.
            return
        if isinstance(func, ast.Attribute):
            self._check_call_attribute(node, func)
            return
        # func is a Call / Subscript / etc.: calling a *value* (e.g. self.layer[i](...), Block(...)(x)).
        # The value's provenance is checked elsewhere; calling it (its __call__) is allowed.

    def _check_call_attribute(self, call: ast.Call, func: ast.Attribute) -> None:
        self._call_funcs.add(
            id(func)
        )  # so _check_attribute doesn't re-flag the same node
        dotted = _dotted(func)
        if dotted is not None:
            root = dotted.split(".")[0]
            if root in ("self", "cls"):
                return  # self.method(...) — receiver type is the module class, not opaque
            if root in self.scan.bindings:
                if not self._resolved_allowed(dotted):
                    self.add(
                        call,
                        "call-not-allowed",
                        f"call to {self._resolve(dotted)!r} is not allow-listed",
                    )
                return
        # Attribute on an opaque value: this is a NAMED METHOD ON A VALUE — never allowed (§3.6).
        self.add(
            call,
            "method-on-value",
            f"named method {func.attr!r} called on a value whose type is unknown; "
            f"route it through a visible wrapper function instead",
        )

    # — attribute reads (not the func of a call) —
    def _check_attribute(self, node: ast.Attribute) -> None:
        if id(node) in self._call_funcs:
            return  # already judged as a call's function position by _check_call_attribute
        if _is_dunder(node.attr):
            self.add(
                node,
                "dunder-attr",
                f"access to dunder attribute {node.attr!r} is not allowed",
            )
            return
        dotted = _dotted(node)
        if dotted is not None:
            root = dotted.split(".")[0]
            if root in ("self", "cls"):
                return
            if root in self.scan.bindings:
                if not self._resolved_allowed(dotted):
                    self.add(
                        node,
                        "attr-not-allowed",
                        f"reference to {self._resolve(dotted)!r} is not allow-listed",
                    )
                return
        # Attribute read on an opaque value: allowed only for the metadata bundle (.shape/.ndim/.dtype).
        if node.attr in METADATA_ATTRS:
            if not self.policy.bundle_enabled("metadata"):
                self.add(
                    node,
                    "bundle-disabled",
                    f"attribute read {node.attr!r} needs the 'metadata' bundle",
                )
        else:
            self.add(
                node,
                "attr-on-value",
                f"attribute {node.attr!r} on a value is not a metadata read; "
                f"route it through a visible wrapper function instead",
            )

    # — operators —
    def _require_bundle(self, node: ast.AST, bundle: str) -> None:
        if not self.policy.bundle_enabled(bundle):
            ops = "/".join(t.__name__ for t in OPERATOR_BUNDLES[bundle])
            self.add(
                node, "bundle-disabled", f"{ops} needs the {bundle!r} method bundle"
            )

    # — assignment / reserved names —
    def _check_assign_targets(self, node) -> None:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for t in targets:
            self._check_reserved_target(t)

    def _check_reserved_target(self, target: ast.AST) -> None:
        for name_node in _iter_names(target):
            if isinstance(name_node.ctx, ast.Store):
                self._check_reserved_name(name_node, name_node.id)

    def _check_reserved_name(self, node: ast.AST, name: str) -> None:
        if name in self.policy.reserved:
            self.add(
                node,
                "reserved-name",
                f"{name!r} is a reserved module alias and may not be rebound",
            )
        elif name in self.scan.visible_defs:
            self.add(
                node,
                "reserved-name",
                f"{name!r} is a visible wrapper name and may not be rebound",
            )

    # — path resolution —
    def _resolve(self, dotted: str) -> str:
        root, _, rest = dotted.partition(".")
        base = self.scan.bindings.get(root, root)
        return f"{base}.{rest}" if rest else base

    def _resolved_allowed(self, dotted: str) -> bool:
        return self.policy.function_allowed(self._resolve(dotted))


# ── helpers ──────────────────────────────────────────────────────────────────────────────
def _normalize_ranges(private) -> list[tuple[int, int]]:
    out = []
    for item in private:
        lo, hi = item
        out.append((int(lo), int(hi)))
    return out


def _in_ranges(node: ast.AST, ranges) -> bool:
    line = getattr(node, "lineno", None)
    if line is None:
        return False
    return any(lo <= line <= hi for lo, hi in ranges)


def _dotted(node: ast.AST) -> str | None:
    """Return the dotted path for a pure Name/Attribute chain, else None."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def _describe(node: ast.AST) -> str:
    return _dotted(node) or type(node).__name__


def _is_dunder(name: str) -> bool:
    return name.startswith("__")


def _iter_names(node: ast.AST):
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            yield n
