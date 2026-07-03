"""The static checker (research approach B).

``verify(source, private, policy)`` parses the file, restricts attention to the *private* line ranges
(the hidden model definition), and walks those nodes default-deny: only explicitly-allowed node types,
calls, operators, and attribute reads pass. It never raises on a policy issue — it returns a
``VerifyResult`` with the violations so callers can inspect them.
"""

from __future__ import annotations

import ast

from pydantic import BaseModel, ConfigDict, Field

from .policy import (
    ALLOWED_DECORATORS,
    ALLOWED_DUNDER_DEFS,
    BANNED_NAMES,
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

# FormattedValue.conversion codes for f-string `!r`/`!s`/`!a` (the ord() of the letter); -1 means
# no conversion flag was given. `{x=}` implicitly uses 114 ('r') when no explicit conversion is set.
_FSTRING_DUNDER_CONVERSIONS: frozenset[int] = frozenset({ord("r"), ord("s"), ord("a")})


class Violation(BaseModel):
    model_config = ConfigDict(frozen=True)

    line: int
    code: str
    message: str


class VerifyResult(BaseModel):
    ok: bool
    violations: list[Violation] = Field(default_factory=list)
    n_calls_checked: int = 0


class FileScan(BaseModel):
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
        )  # Attribute/Name nodes already judged as part of a call's func (avoid double-reporting)
        self._class_stack: list[ast.ClassDef] = []  # enclosing class, for self.<name> vetting
        self._self_attr_cache: dict[int, dict[str, bool]] = {}  # id(ClassDef) -> attr safety
        self._scope_stack: list[ast.AST] = (
            []
        )  # ClassDef/FunctionDef/Lambda in true nesting order, for self/cls parameter vetting
        self._annotation_nodes: set[int] = (
            set()
        )  # every node inside a type-annotation subtree (e.g. `x: str`, `x: dict[str, bytes]`)

    def add(self, node: ast.AST, code: str, message: str) -> None:
        self.violations.append(
            Violation(line=getattr(node, "lineno", 0), code=code, message=message)
        )

    def visit(self, node: ast.AST) -> None:
        """Walk the tree; enforce only on nodes inside the private ranges, recurse everywhere."""
        if _in_ranges(node, self.ranges):
            self._enforce(node)
        is_class = isinstance(node, ast.ClassDef)
        is_scope = is_class or isinstance(node, (ast.FunctionDef, ast.Lambda))
        if is_class:
            self._class_stack.append(node)
        # A type annotation (`x: str`, `x: list[str]`, `def f() -> dict[str, bytes]`) is never
        # invoked as a callable -- its only runtime effect is populating __annotations__ (itself
        # a dunder, already banned to read back) -- so nothing in it is a BANNED_NAMES reference
        # or a banned container literal. Mark every node in the whole annotation subtree, not
        # just its top node: subscripted/generic annotations nest the actual type names one or
        # more levels down (`list[str]` is `Subscript(value=Name('list'), slice=Name('str'))`,
        # and a multi-arg subscript like `dict[str, bytes]` puts them inside a Tuple slice).
        # This does NOT exempt Call/Attribute/Subscript nodes from their own checks -- a call in
        # a passive position (`x: evil()`) must still be caught, and is, since _check_call never
        # consults this set.
        for ann in (getattr(node, "annotation", None), getattr(node, "returns", None)):
            if ann is not None:
                for descendant in ast.walk(ann):
                    self._annotation_nodes.add(id(descendant))
        if is_scope:
            self._scope_stack.append(node)
        for child in ast.iter_child_nodes(node):
            self.visit(child)
        if is_scope:
            self._scope_stack.pop()
        if is_class:
            self._class_stack.pop()

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
        elif isinstance(node, ast.Lambda):
            self._check_self_cls_params(node.args)
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
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            # ast.comprehension itself carries no lineno/col_offset (verified: it's the one node
            # type in the grammar with no position info), so gating on ITS range membership would
            # never fire -- dispatch from the enclosing comprehension expression instead, which
            # does have a position, and check every one of its generators' targets from here.
            for generator in node.generators:
                self._check_reserved_target(generator.target)
        elif isinstance(node, ast.arg):
            self._check_reserved_name(node, node.arg)
        elif isinstance(node, (ast.List, ast.Dict, ast.Set, ast.Tuple)):
            self._check_container_literal(node)
        elif isinstance(node, ast.Name):
            self._check_name(node)
        elif isinstance(node, ast.FormattedValue):
            self._check_formatted_value(node)

    # — defs / classes —
    def _check_def(self, node: ast.FunctionDef) -> None:
        # Checks a function def: its decorators, that it defines no non-allow-listed dunder, and
        # that self/cls only appears where it's genuinely trustworthy (see _check_self_cls_params).
        self._check_decorators(node)
        if _is_dunder(node.name) and node.name not in ALLOWED_DUNDER_DEFS:
            self.add(
                node,
                "dunder-def",
                f"defining magic method {node.name!r} is not allowed",
            )
        self._check_self_cls_params(node.args)

    def _check_self_cls_params(self, args: ast.arguments) -> None:
        # self.<name> / cls.<name> is trusted everywhere else in this checker purely by matching
        # the literal identifier "self"/"cls" -- it never verifies the name is actually bound to
        # the real instance. That trust is only sound for the genuine first parameter of a method
        # defined directly in a class body (where Python itself guarantees the binding). Anywhere
        # else -- a nested function/lambda parameter of the same name, a non-first parameter, a
        # *args/**kwargs catch-all -- an attacker's own local object would receive the identical
        # blanket trust, with the self-attribute safety table keyed to the wrong (enclosing) class
        # instead of whatever object the identifier is actually bound to at runtime.
        is_direct_method = bool(self._scope_stack) and isinstance(
            self._scope_stack[-1], ast.ClassDef
        )
        first = args.posonlyargs[0] if args.posonlyargs else (args.args[0] if args.args else None)
        all_params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
        if args.vararg is not None:
            all_params.append(args.vararg)
        if args.kwarg is not None:
            all_params.append(args.kwarg)
        for a in all_params:
            if a.arg in ("self", "cls") and not (is_direct_method and a is first):
                self.add(
                    a,
                    "reserved-name",
                    f"{a.arg!r} may only be the first parameter of a method defined directly "
                    f"in a class body; self/cls attribute access is trusted by identifier alone",
                )

    def _check_class(self, node: ast.ClassDef) -> None:
        # Checks a class def: its decorators, no class keywords (e.g. metaclass=), and allow-listed bases.
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
        # Checks that every decorator on a def/class resolves to one on the decorator allow-list.
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
        # Checks a call: bans dynamic-escape builtins by name and routes attribute-calls for vetting.
        self.n_calls += 1
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in BANNED_NAMES:
                self.add(node, "banned-call", f"call to {func.id!r} is not allowed")
                self._call_funcs.add(id(func))  # _check_name would otherwise re-flag this Name
                return
            if (
                func.id in self.scan.bindings
                and func.id not in self.scan.hidden_defs
                and func.id not in self.scan.visible_defs
            ):
                # A bare name imported via `from X import name [as alias]` in the public region
                # (recorded in scan.bindings) resolves through the same binding table dotted
                # paths use — apply the same allowlist + denylist, not just the BANNED_NAMES check.
                if not self._resolved_allowed(func.id):
                    self.add(
                        node,
                        "call-not-allowed",
                        f"call to {self._resolve(func.id)!r} is not allow-listed",
                    )
                return
            # Otherwise a bare-name call (local var / hidden or visible def / safe builtin) is allowed;
            # nothing dangerous can reach a local name given the other rules.
            return
        if isinstance(func, ast.Attribute):
            self._check_call_attribute(node, func)
            return
        if isinstance(func, ast.Subscript) and _rooted_in_self(func):
            self._check_self_subscript_call(node, func)
            return
        # func is a Call / Subscript(non-self) / etc.: calling a *value* (e.g. Block(...)(x),
        # d["o"](...)). The value's provenance is checked elsewhere; calling it is allowed.

    def _check_self_subscript_call(self, call: ast.Call, func: ast.Subscript) -> None:
        # Checks self.<name>[i](...) — the Flax "self.layer[i](x)" idiom: only allowed when
        # <name> is inherited/vetted-safe the same way a direct self.<name>(...) call is.
        attr = _self_attr_name(func.value) if isinstance(func.value, ast.Attribute) else None
        if attr is not None and self._self_attr_is_safe(attr):
            return
        message = (
            f"self.{attr!r}[...] was assigned a value that isn't a list/tuple of allow-listed "
            f"constructors; calling an element of it is not allowed"
            if attr is not None
            else "only self.<name>[...] may be called this way, not a deeper self-rooted chain"
        )
        self.add(call, "attr-on-value", message)

    def _self_attr_is_safe(self, attr: str) -> bool:
        # An attribute the class never assigns is presumed inherited from the class's own
        # (already vetted, since bases are allow-listed) base class -- e.g. nn.Module's
        # self.param/self.variable. Only attributes the class itself assigns get vetted
        # against what was actually stored there.
        if not self._class_stack:
            return False
        cls_node = self._class_stack[-1]
        table = self._self_attr_cache.get(id(cls_node))
        if table is None:
            table = self._build_self_attr_table(cls_node)
            self._self_attr_cache[id(cls_node)] = table
        return table.get(attr, True)

    def _build_self_attr_table(self, cls_node: ast.ClassDef) -> dict[str, bool]:
        # Checks, for every self.<name> assigned anywhere in the class, whether every value
        # ever stored there is a vetted-safe source; AugAssign always disqualifies (compound-
        # mutating a submodule reference isn't a real Flax pattern and is inherently suspect).
        values: dict[str, list[ast.AST]] = {}
        disqualified: set[str] = set()
        for node in ast.walk(cls_node):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    attr = _self_attr_name(t)
                    if attr is not None:
                        values.setdefault(attr, []).append(node.value)
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                attr = _self_attr_name(node.target)
                if attr is not None:
                    values.setdefault(attr, []).append(node.value)
            elif isinstance(node, ast.AugAssign):
                attr = _self_attr_name(node.target)
                if attr is not None:
                    disqualified.add(attr)
        return {
            attr: attr not in disqualified
            and all(self._is_safe_self_value(v) for v in vs)
            for attr, vs in values.items()
        }

    def _is_safe_self_value(self, value: ast.AST) -> bool:
        # A value is a vetted "submodule" source if it's a call to an allow-listed library
        # constructor or a name defined in this file (hidden or visible), or a list/tuple/set/
        # comprehension of such values (the self.layer[i](x) idiom).
        if isinstance(value, ast.Call):
            func = value.func
            dotted = _dotted(func)
            if dotted is not None and dotted.split(".")[0] in self.scan.bindings:
                return self._resolved_allowed(dotted)
            return isinstance(func, ast.Name) and (
                func.id in self.scan.hidden_defs or func.id in self.scan.visible_defs
            )
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            return all(self._is_safe_self_value(e) for e in value.elts)
        if isinstance(value, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            return self._is_safe_self_value(value.elt)
        if isinstance(value, ast.DictComp):
            return self._is_safe_self_value(value.value)
        return False

    def _check_call_attribute(self, call: ast.Call, func: ast.Attribute) -> None:
        # Checks a dotted call: vets self/cls attribute calls, vets library calls, bans methods on opaque values.
        self._call_funcs.add(
            id(func)
        )  # so _check_attribute doesn't re-flag the same node
        dotted = _dotted(func)
        if dotted is not None:
            root = dotted.split(".")[0]
            if root in ("self", "cls"):
                attr = _self_attr_name(func)
                if attr is not None and self._self_attr_is_safe(attr):
                    return  # self.<name>(...) — <name> is inherited or was assigned a vetted source
                message = (
                    f"self.{attr!r} was assigned a value that isn't an allow-listed constructor "
                    f"or a locally-defined class; calling it is not allowed"
                    if attr is not None
                    else f"{dotted!r}: only a single self.<name> attribute may be called, "
                    f"not a deeper attribute chain"
                )
                self.add(call, "attr-on-value", message)
                return
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
        # Checks an attribute read: bans dunders, vets library refs, bans reads on opaque values.
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
                if _self_attr_name(node) is not None:
                    return  # self.<name> read/store — single level is always fine (Flax setup/param)
                self.add(
                    node,
                    "attr-on-value",
                    f"{dotted!r}: only a single self.<name> attribute may be accessed, "
                    f"not a deeper attribute chain",
                )
                return
            if root in self.scan.bindings:
                if not self._resolved_allowed(dotted):
                    self.add(
                        node,
                        "attr-not-allowed",
                        f"reference to {self._resolve(dotted)!r} is not allow-listed",
                    )
                return
        # Attribute read on an opaque value (including .shape/.ndim/.dtype): we can't pin the
        # receiver's type, so it must be routed through a visible wrapper function.
        self.add(
            node,
            "attr-on-value",
            f"attribute {node.attr!r} on a value is not allowed; "
            f"route it through a visible wrapper function instead",
        )

    # — bare name reads (not the func of a call) —
    def _check_name(self, node: ast.Name) -> None:
        # Checks a Load-context reference to a banned builtin: this is what closes aliasing
        # (`f = open`), and every other position a reference can occupy (container element,
        # return value, call argument, IfExp/BoolOp branch, ...) — the reference itself is the
        # violation, so nothing downstream needs to be traced. Also bans bare dunder names
        # (e.g. the implicit `__class__` cell every method body has) the same way an
        # Attribute-shaped dunder access already is — the dunder ban shouldn't depend on
        # whether the reference happens to have a dot in front of it.
        if not isinstance(node.ctx, ast.Load):
            return
        if id(node) in self._call_funcs:
            return  # already reported as banned-call by _check_call
        if node.id in BANNED_NAMES:
            if id(node) in self._annotation_nodes:
                return  # `x: str` / `def f() -> str` — a type annotation, not a call/reference
            self.add(node, "banned-call", f"reference to {node.id!r} is not allowed")
        elif _is_dunder(node.id):
            self.add(
                node,
                "dunder-name",
                f"reference to dunder name {node.id!r} is not allowed",
            )

    # — f-strings —
    def _check_formatted_value(self, node: ast.FormattedValue) -> None:
        # Checks an f-string interpolation's conversion flag: `!r`/`!s`/`!a` (and the `{x=}`
        # debug specifier, which defaults to the same repr conversion) invoke a dunder method
        # on the interpolated value with no Call node for _check_call to ever see. Plain
        # interpolation (`f"{x}"`, conversion == -1) is unaffected and stays allowed.
        if node.conversion in _FSTRING_DUNDER_CONVERSIONS:
            self.add(
                node,
                "method-on-value",
                "f-string conversion flags (!r/!s/!a) call a dunder method on a value whose "
                "type is unknown; route it through a visible wrapper function instead",
            )

    # — container literals —
    def _check_container_literal(self, node) -> None:
        # Checks a list/dict/set/tuple literal: storing a banned-builtin reference in a
        # persistent, index-addressable container is itself the violation — we don't attempt
        # to track which slot holds what through later subscript access, so the sound and
        # simple move is to reject the container at construction time instead. Exempt when this
        # literal is itself part of a type annotation (e.g. a multi-arg subscript like
        # `dict[str, bytes]` puts its arguments in a Tuple slice) — never invoked as a container.
        if id(node) in self._annotation_nodes:
            return
        if _contains_banned_reference(node):
            self.add(
                node,
                "banned-construct",
                "a list/dict/set/tuple literal may not hold a reference to a banned builtin",
            )

    # — operators —
    def _require_bundle(self, node: ast.AST, bundle: str) -> None:
        # Checks that the operator's bundle (arithmetic/comparison/indexing) is enabled by the policy.
        if not self.policy.bundle_enabled(bundle):
            ops = "/".join(t.__name__ for t in OPERATOR_BUNDLES[bundle])
            self.add(
                node, "bundle-disabled", f"{ops} needs the {bundle!r} method bundle"
            )

    # — assignment / reserved names —
    def _check_assign_targets(self, node) -> None:
        # Checks every target of an assignment for a forbidden rebind of a reserved name, and
        # flags storing a banned-builtin reference into a container slot (d["k"] = open) — the
        # container name itself is Load context here, so the reserved-name walk above never
        # looks at it, and the value being stored is otherwise never inspected at all.
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for t in targets:
            self._check_reserved_target(t)
            if (
                isinstance(t, ast.Subscript)
                and node.value is not None
                and _contains_banned_reference(node.value)
            ):
                self.add(
                    t,
                    "banned-construct",
                    "storing a reference to a banned builtin into a container slot is not allowed",
                )

    def _check_reserved_target(self, target: ast.AST) -> None:
        # Checks each name being bound (Store context) in an assign/for/comprehension target.
        # self/cls may never be rebound this way -- see _check_self_cls_params for why the
        # identifier alone must stay tied to the genuine instance/class parameter.
        for name_node in _iter_names(target):
            if isinstance(name_node.ctx, ast.Store):
                if name_node.id in ("self", "cls"):
                    self.add(
                        name_node,
                        "reserved-name",
                        f"{name_node.id!r} may not be rebound; self/cls attribute access is "
                        f"trusted by identifier alone, not a verified reference to the real "
                        f"instance",
                    )
                    continue
                self._check_reserved_name(name_node, name_node.id)

    def _check_reserved_name(self, node: ast.AST, name: str) -> None:
        # Checks that a bound name doesn't shadow a reserved module alias or a visible wrapper name.
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
        # Checks a dotted path against the policy allow-list after resolving its import alias.
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


def _self_attr_name(node: ast.AST) -> str | None:
    """Returns the attr name for a single-level self.<name>/cls.<name> access, else None."""
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in ("self", "cls")
    ):
        return node.attr
    return None


def _rooted_in_self(node: ast.AST) -> bool:
    """True iff an Attribute/Subscript chain's ultimate base is self/cls."""
    cur = node
    while isinstance(cur, (ast.Attribute, ast.Subscript)):
        cur = cur.value
    return isinstance(cur, ast.Name) and cur.id in ("self", "cls")


def _describe(node: ast.AST) -> str:
    return _dotted(node) or type(node).__name__


def _is_dunder(name: str) -> bool:
    return name.startswith("__")


def _iter_names(node: ast.AST):
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            yield n


def _contains_banned_reference(node: ast.AST) -> bool:
    """True iff a Load-context reference to a BANNED_NAMES identifier appears anywhere in node."""
    return any(
        isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in BANNED_NAMES
        for n in ast.walk(node)
    )
