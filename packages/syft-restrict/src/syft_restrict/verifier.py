"""The static checker — verifies that the private region only does trusted math.

``verify(source, private, policy)`` parses the file, restricts attention to the *private* line ranges
(the hidden model definition), and walks those nodes **default-deny**: a node is allowed only if a
check below explicitly permits it. It never raises on a policy issue — it returns a ``VerifyResult``
listing the violations, so callers can inspect them.

The full whitelist and the reasoning behind each rule live in ``docs/verify.md``; the full deny lists
live in ``docs/blacklist.md``. Comments here stay short and cite those docs where the "why" is subtle.
"""

from __future__ import annotations

import ast
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .astutil import (
    FileScan,
    describe,
    dotted_name,
    is_dunder,
    iter_names,
    node_in_ranges,
    normalize_ranges,
    rooted_in_self,
    scan_file,
    self_attr_name,
)
from .policy import (
    ALLOWED_DECORATORS,
    ALLOWED_DUNDER_DEFS,
    BANNED_NAMES,
    OPERATOR_BUNDLES,
    Policy,
)

# ── node-type allow-list: the structural syntax the private region may use (docs/verify.md#always-on-allow-list) ──
# Anything not listed here is rejected by default, so new/unknown syntax (walrus, match, …) is denied
# until a human reviews it. See docs/blacklist.md for what this default-deny catches.
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

# ── node-type deny-list: statements that reach the host, filesystem, or interpreter (docs/blacklist.md) ──
# Listed explicitly (rather than just left off the allow-list) so their violation names them clearly.
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

# ── violation-code registry: every code a check can raise, one line each (docs/blacklist.md) ──
# This is the single place to see the full set — grep a raising method name to find its check(s).
ViolationCode = Literal[
    "banned-construct",  # _enforce, _check_container_literal, _check_assign_targets
    "node-type",  # _enforce (node type outside the always-on allow-list)
    "dunder-def",  # _check_def (defining a magic/hook method)
    "class-keyword",  # _check_class (metaclass= or other class keyword arg)
    "class-base",  # _check_class (non-allow-listed base class)
    "decorator",  # _check_decorators
    "reserved-name",  # _check_self_cls_params, _check_reserved_target, _check_reserved_name
    "banned-call",  # _check_call, _check_name
    "call-not-allowed",  # _check_call, _check_call_attribute
    "dunder-attr",  # _check_call_attribute, _check_attribute
    "attr-on-value",  # _check_call_attribute, _check_self_subscript_call, _check_attribute
    "method-on-value",  # _check_call_attribute, _check_formatted_value
    "attr-not-allowed",  # _check_attribute
    "dunder-name",  # _check_name
    "bundle-disabled",  # _require_bundle
]


class Violation(BaseModel):
    model_config = ConfigDict(frozen=True)

    line: int
    code: ViolationCode
    message: str


class VerifyResult(BaseModel):
    ok: bool
    violations: list[Violation] = Field(default_factory=list)
    n_calls_checked: int = 0


def verify(source: str, private, policy: Policy) -> VerifyResult:
    ranges = normalize_ranges(private)
    tree = ast.parse(source)
    scan = scan_file(tree, ranges)
    policy.reserved_names = set(
        scan.import_bindings
    )  # trusted module aliases may not be rebound
    checker = _Checker(policy, scan, ranges)
    checker.visit(tree)
    return VerifyResult(
        ok=not checker.violations,
        violations=checker.violations,
        n_calls_checked=checker.n_calls,
    )


# ──────────────────────────────────────────────────────────────────────────────────────────────
# The checker walks every private-region node and runs exactly one check per node. Default-deny:
# a node passes only if its check permits it. Each check guards one escape category — full rationale
# in docs/verify.md, full deny lists in docs/blacklist.md:
#
#   dynamic-code / reflection builtins   -> _check_call, _check_name
#   IO / host-escape statements          -> _BANNED_NODES (in _enforce)
#   unknown / future syntax              -> _ALLOWED_NODES default-deny (in _enforce)
#   library call/attr by name            -> _check_call_attribute, _check_attribute (resolver + allow/disallow)
#   named method / attr on opaque value  -> _check_call_attribute, _check_attribute
#   f-string repr/str/ascii escape       -> _check_formatted_value
#   forged self/cls trust                -> _check_self_cls_params, _check_reserved_target
#   aliasing a banned callable           -> _check_name, _check_container_literal, _check_assign_targets
#   class-creation hooks                 -> _check_class, _check_decorators, _check_def
# ──────────────────────────────────────────────────────────────────────────────────────────────
class _Checker:
    def __init__(self, policy: Policy, scan: FileScan, ranges):
        self.policy = policy
        self.scan = scan
        self.ranges = ranges
        self.violations: list[Violation] = []
        self.n_calls = 0

        # Attribute/Name nodes already judged as a call's func by _check_call_attribute/_check_call,
        # so _check_attribute/_check_name don't re-flag the same node a second time.
        self._checked_call_targets: set[int] = set()

        # Enclosing classes/functions/lambdas, in true nesting order; pushed/popped by visit(). Read by
        # _check_self_cls_params (is the "self"/"cls" param a direct method's first param?) and by
        # _enclosing_class() (the nearest ClassDef, for self.<name> vetting).
        self._scope_stack: list[ast.AST] = []

        # Answers "is self.<attr> safe to call/subscript?" for _check_call_attribute and
        # _check_self_subscript_call; see _SelfAttrTrust below.
        self._self_attr = _SelfAttrTrust(scan, self._resolved_allowed)

        #  We store every node inside a type-annotation subtree (`x: str`, `x: dict[str, bytes]`), populated by
        # _mark_annotation_subtrees; exempt from the name/container checks in _check_name and
        # _check_container_literal because annotations are never invoked (see visit).
        self._annotation_nodes: set[int] = set()

        # One {local_name: self_attr_name} dict per enclosing class/function/lambda scope (mirrors
        # _scope_stack's push/pop), tracking which local variables currently alias a self.<attr> --
        # so `tmp = self.fn; tmp(x)` is checked the same way `self.fn(x)` directly is; see
        # _track_self_attr_alias and its use in _check_call.
        self._alias_stack: list[dict[str, str]] = []

    def report(self, node: ast.AST, code: ViolationCode, message: str) -> None:
        self.violations.append(
            Violation(line=getattr(node, "lineno", 0), code=code, message=message)
        )

    # ── tree walk ───────────────────────────────────────────────────────────────────────────
    def visit(self, node: ast.AST) -> None:
        """Walk the whole tree; enforce only on nodes inside the private ranges, recurse everywhere."""
        if node_in_ranges(node, self.ranges):
            self._enforce(node)

        is_scope = isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.Lambda))
        if is_scope:
            self._scope_stack.append(node)
            self._alias_stack.append({})
        self._mark_annotation_subtrees(node)

        for child in ast.iter_child_nodes(node):
            self.visit(child)

        if is_scope:
            self._scope_stack.pop()
            self._alias_stack.pop()

    def _enclosing_class(self) -> ast.ClassDef | None:
        """The nearest enclosing ClassDef, skipping any FunctionDef/Lambda frames above it."""
        for frame in reversed(self._scope_stack):
            if isinstance(frame, ast.ClassDef):
                return frame
        return None

    def _mark_annotation_subtrees(self, node: ast.AST) -> None:
        """A type annotation (`x: str`, `x: list[str]`, `def f() -> dict[str, bytes]`) is never invoked,
        so it can hold a name that is banned in a real reference (see docs/verify.md#edge-cases). Mark
        the WHOLE subtree, not just its top node: generics nest the type names one or more levels down
        (`list[str]` is `Subscript(Name('list'), Name('str'))`). Call/Attribute nodes inside still run
        their own checks — a call in a passive position (`x: evil()`) is caught, since no check reads
        this set to skip itself; only the name/container checks consult it."""
        for ann in (getattr(node, "annotation", None), getattr(node, "returns", None)):
            if ann is not None:
                for descendant in ast.walk(ann):
                    self._annotation_nodes.add(id(descendant))

    def _enforce(self, node: ast.AST) -> None:
        """Run the one check that applies to this node type (recursion is handled by visit)."""
        if isinstance(node, _BANNED_NODES):
            self.report(
                node,
                "banned-construct",
                f"{type(node).__name__} is not allowed in the hidden region",
            )
            return
        if not isinstance(node, _ALLOWED_NODES):
            self.report(
                node,
                "node-type",
                f"{type(node).__name__} is not on the node-type allow-list",
            )
            return

        # --- definitions & classes ---
        if isinstance(node, ast.FunctionDef):
            self._check_def(node)
        elif isinstance(node, ast.Lambda):
            self._check_arguments_dont_abuse_self_or_cls(node.args)
        elif isinstance(node, ast.ClassDef):
            self._check_class(node)
        # --- calls & attribute access ---
        elif isinstance(node, ast.Call):
            self._check_call(node)
        elif isinstance(node, ast.Attribute):
            self._check_attribute(node)
        # --- operators (gated by policy bundles) ---
        elif isinstance(node, (ast.BinOp, ast.UnaryOp)):
            self._require_bundle(node, "arithmetic")
        elif isinstance(node, (ast.Compare, ast.BoolOp)):
            self._require_bundle(node, "comparison")
        elif isinstance(node, (ast.Subscript, ast.Slice)):
            self._require_bundle(node, "indexing")
        # --- name binding (assignments, loops, comprehensions, params) ---
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            self._check_assign_targets(node)
        elif isinstance(node, ast.For):
            self._check_reserved_target(node.target)
        elif isinstance(
            node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        ):
            # ast.comprehension carries no lineno, so gating on ITS range membership never fires --
            # dispatch each generator's target from the enclosing comprehension expression, which does
            # have a position (see docs/verify.md#edge-cases).
            for generator in node.generators:
                self._check_reserved_target(generator.target)
        elif isinstance(node, ast.arg):
            self._check_reserved_name(node, node.arg)
        # --- literals, names, f-strings ---
        elif isinstance(node, (ast.List, ast.Dict, ast.Set, ast.Tuple)):
            self._check_container_literal(node)
        elif isinstance(node, ast.Name):
            self._check_name(node)
        elif isinstance(node, ast.FormattedValue):
            self._check_formatted_value(node)

    # ── definitions & classes ────────────────────────────────────────────────────────────────
    def _check_def(self, node: ast.FunctionDef) -> None:
        """Guards against: defining magic/hook methods (__getattr__, __reduce__, …) that Python runs
        automatically without an explicit call in the math, and shadowing a trusted module alias
        or visible wrapper name with a local def (the same forging _check_reserved_name blocks for
        a plain assignment target)."""
        # are the function decorator in the list of allowed decorators?
        self._check_decorators(node)

        # is the function name not a reserved name (shadowing an import)
        self._check_reserved_name(node, node.name)

        # not a dunder unless allowed_dunder
        if is_dunder(node.name) and node.name not in ALLOWED_DUNDER_DEFS:
            self.report(
                node,
                "dunder-def",
                f"defining magic method {node.name!r} is not allowed",
            )
        # if the function has cls or self only allow if it is the first argument for a method
        self._check_arguments_dont_abuse_self_or_cls(node.args)

    def _check_class(self, node: ast.ClassDef) -> None:
        # check only allowed decorators
        self._check_decorators(node)

        # is the class name not a reserved name (shadowing an import)
        self._check_reserved_name(node, node.name)

        # does not use keywords (like metaclass=)
        if node.keywords:
            self.report(
                node,
                "class-keyword",
                "class keyword arguments (e.g. metaclass=) are not allowed",
            )

        # if it uses base classes, are they allow-listed? (they would be called in the __init_subclass__ method)
        for base in node.bases:
            if not self._base_class_allowed(base):
                self.report(
                    base,
                    "class-base",
                    f"base class {describe(base)!r} is not allow-listed",
                )

    def _base_class_allowed(self, base: ast.AST) -> bool:
        """Only an allow-listed import (resolved through any aliases, e.g. nn.Module) may be a base.
        object and hidden-region classes are rejected: a base's metaclass / __init_subclass__ runs
        at class-creation time, so we require it to resolve to something the policy vetted."""
        path = dotted_name(base)
        return bool(path) and self._resolved_allowed(path)

    def _check_decorators(self, node) -> None:
        """Guards against: a decorator running attacker code the instant the def/class is reached."""
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            path = dotted_name(target)
            resolved = self._resolve(path) if path else None
            if not (resolved in ALLOWED_DECORATORS or path in ALLOWED_DECORATORS):
                self.report(
                    dec,
                    "decorator",
                    f"decorator {describe(target)!r} is not allow-listed",
                )

    def _check_arguments_dont_abuse_self_or_cls(self, args: ast.arguments) -> None:
        """Normally, we are not allowed to call x.y in hidden code,
        but we have an exception for calling self.<name> in a class. However, this is only allowed if
        self is bound to the real instance. This functino protects against forging the self/cls exemption.
        That is only sound for the genuine first parameter of a method
        defined directly in a class body (where Python guarantees the binding). Anywhere else -- a
        non-first param, a nested function/lambda param, a *args/**kwargs of the same name -- an
        attacker's own object would receive the identical blanket trust. See docs/verify.md#edge-cases."""
        is_direct_method = bool(self._scope_stack) and isinstance(
            self._scope_stack[-1], ast.ClassDef
        )
        first = (
            args.posonlyargs[0]
            if args.posonlyargs
            else (args.args[0] if args.args else None)
        )
        all_params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
        if args.vararg is not None:
            all_params.append(args.vararg)
        if args.kwarg is not None:
            all_params.append(args.kwarg)
        for a in all_params:
            if a.arg in ("self", "cls") and not (is_direct_method and a is first):
                self.report(
                    a,
                    "reserved-name",
                    f"{a.arg!r} may only be the first parameter of a method defined directly "
                    f"in a class body; self/cls attribute access is trusted by identifier alone",
                )

    # ── calls ────────────────────────────────────────────────────────────────────────────────
    def _check_call(self, node: ast.Call) -> None:
        """Guards against: calling a dynamic-code/IO builtin (eval, open, …) or a non-allow-listed public
        import, whether named directly or aliased. Attribute-position calls route to a stricter check."""

        self.n_calls += 1
        func = node.func
        if isinstance(func, ast.Name):
            # Banned builtins (eval, open, …) are not special-cased here: the func node is a Load Name,
            # so _check_name is the single backstop that flags it (and every aliasing position too).
            if func.id in self.scan.import_bindings:
                # A bare name imported via `from X import name [as alias]` in the public region resolves
                # through the same binding table dotted paths use — apply the same allow+deny list.
                if not self._resolved_allowed(func.id):
                    self.report(
                        node,
                        "call-not-allowed",
                        f"call to {self._resolve(func.id)!r} is not allow-listed",
                    )
                return
            aliased_attr = self._current_aliases().get(func.id)
            if aliased_attr is not None and not self._self_attr.is_safe(
                aliased_attr, self._enclosing_class()
            ):
                self.report(
                    node,
                    "attr-on-value",
                    f"{func.id!r} was assigned from self.{aliased_attr!r}, which isn't an "
                    f"allow-listed constructor or a locally-defined class; calling it is not "
                    f"allowed",
                )
                return
            # Otherwise a bare-name call (local var / hidden or visible def / safe builtin) is allowed.
            return
        # this checks the part before the call (e.g. for x.y.z() it checks x.y)
        if isinstance(func, ast.Attribute):
            self._check_call_attribute(node, func)
            return
        # this checks the part before the call (e.g. for x[i].z() it checks x[i])
        if isinstance(func, ast.Subscript) and rooted_in_self(func):
            self._check_self_subscript_call(node, func)
            return
        # func is a Call / non-self Subscript / etc.: calling a *value* (Block(...)(x), d["o"](...)).
        # The value's provenance was checked where it was produced; calling it is allowed.

    # NOTE: _check_call_attribute, _check_self_subscript_call, and _check_attribute (below) are
    # intentionally parallel — each splits a dotted path, checks self/cls-rootedness, checks
    # dunder-ness, then resolves against import bindings, but with different downstream codes and
    # messages for the call/subscript-call/read shape. Don't unify them into one shared function:
    # keep them hand-in-sync instead, since a merge risks changing which code fires for an edge case.
    def _check_call_attribute(self, call: ast.Call, func: ast.Attribute) -> None:
        """Guards against: calling a non-allow-listed library path, and calling a named method on an
        opaque value (x.reshape(...)) whose type — and thus what the call does — we can't pin."""
        self._checked_call_targets.add(
            id(func)
        )  # so _check_attribute won't re-flag this node
        path = dotted_name(func)
        if path is not None:
            root = path.split(".")[0]
            if root in ("self", "cls"):
                attr = self_attr_name(func)
                if attr is not None and is_dunder(attr):
                    self.report(
                        call,
                        "dunder-attr",
                        f"access to dunder attribute {attr!r} is not allowed",
                    )
                    return
                if attr is not None and self._self_attr.is_safe(
                    attr, self._enclosing_class()
                ):
                    return  # self.<name>(...) — <name> is inherited or was assigned a vetted source
                message = (
                    f"self.{attr!r} was assigned a value that isn't an allow-listed constructor "
                    f"or a locally-defined class; calling it is not allowed"
                    if attr is not None
                    else f"{path!r}: only a single self.<name> attribute may be called, "
                    f"not a deeper attribute chain"
                )
                self.report(call, "attr-on-value", message)
                return
            if root in self.scan.import_bindings:
                if not self._resolved_allowed(path):
                    self.report(
                        call,
                        "call-not-allowed",
                        f"call to {self._resolve(path)!r} is not allow-listed",
                    )
                return
        # A named method on an opaque value — never allowed; route it through a visible wrapper.
        self.report(
            call,
            "method-on-value",
            f"named method {func.attr!r} called on a value whose type is unknown; "
            f"route it through a visible wrapper function instead",
        )

    def _check_self_subscript_call(self, call: ast.Call, func: ast.Subscript) -> None:
        """Guards against: the Flax self.layer[i](x) idiom smuggling a tainted callable — only allowed
        when <name> is inherited or was assigned a vetted-safe source (see _SelfAttrTrust)."""
        attr = (
            self_attr_name(func.value)
            if isinstance(func.value, ast.Attribute)
            else None
        )
        if attr is not None and self._self_attr.is_safe(attr, self._enclosing_class()):
            return
        message = (
            f"self.{attr!r}[...] was assigned a value that isn't a list/tuple of allow-listed "
            f"constructors; calling an element of it is not allowed"
            if attr is not None
            else "only self.<name>[...] may be called this way, not a deeper self-rooted chain"
        )
        self.report(call, "attr-on-value", message)

    # ── attribute reads (not the func of a call) ─────────────────────────────────────────────
    def _check_attribute(self, node: ast.Attribute) -> None:
        """Guards against: reflection/dunder reads (x.__class__), non-allow-listed library paths, and any
        attribute read on an opaque value (x.shape) whose receiver type we can't pin."""
        if id(node) in self._checked_call_targets:
            return  # already judged as a call's function position by _check_call_attribute
        if is_dunder(node.attr):
            self.report(
                node,
                "dunder-attr",
                f"access to dunder attribute {node.attr!r} is not allowed",
            )
            return
        path = dotted_name(node)
        if path is not None:
            root = path.split(".")[0]
            if root in ("self", "cls"):
                if self_attr_name(node) is not None:
                    return  # self.<name> — a single level is always fine (Flax setup/param)
                self.report(
                    node,
                    "attr-on-value",
                    f"{path!r}: only a single self.<name> attribute may be accessed, "
                    f"not a deeper attribute chain",
                )
                return
            if root in self.scan.import_bindings:
                if not self._resolved_allowed(path):
                    self.report(
                        node,
                        "attr-not-allowed",
                        f"reference to {self._resolve(path)!r} is not allow-listed",
                    )
                return
        # Attribute read on an opaque value (including .shape/.ndim/.dtype): route it through a wrapper.
        self.report(
            node,
            "attr-on-value",
            f"attribute {node.attr!r} on a value is not allowed; "
            f"route it through a visible wrapper function instead",
        )

    # ── bare name reads (not the func of a call) ─────────────────────────────────────────────
    def _check_name(self, node: ast.Name) -> None:
        """Guards against: aliasing a banned builtin (`f = open; f(...)`) — the reference itself is the
        violation, so every position it can occupy (container, return, arg, IfExp branch) is covered
        without tracing downstream. Also bans bare dunder names (`__class__`), the same reflection
        surface as an Attribute dunder, regardless of whether a dot precedes it."""
        if not isinstance(node.ctx, ast.Load):
            return
        if id(node) in self._checked_call_targets:
            return  # already reported as banned-call by _check_call
        if node.id in BANNED_NAMES:
            if id(node) in self._annotation_nodes:
                return  # `x: str` / `def f() -> str` — a type annotation, not a reference
            self.report(node, "banned-call", f"reference to {node.id!r} is not allowed")
        elif is_dunder(node.id):
            self.report(
                node,
                "dunder-name",
                f"reference to dunder name {node.id!r} is not allowed",
            )

    # ── f-strings ────────────────────────────────────────────────────────────────────────────
    def _check_formatted_value(self, node: ast.FormattedValue) -> None:
        """Guards against: every f-string interpolation — plain `f"{x}"` included — invoking
        __format__ (and, via default object.__format__, __str__) on the value with no Call node
        for _check_call to see. Python calls type(x).__format__(x, spec) for every FormattedValue
        regardless of conversion flag, so there is no conversion-less case that "stays allowed"."""
        self.report(
            node,
            "method-on-value",
            "f-string interpolation calls __format__ on a value whose type is unknown; "
            "route it through a visible wrapper function instead",
        )

    # ── container literals ───────────────────────────────────────────────────────────────────
    def _check_container_literal(self, node) -> None:
        """Guards against: stashing a banned-builtin reference in a list/dict/set/tuple for later
        dispatch (`d = {"o": open}; d["o"](...)`). We don't track which slot holds what, so we reject
        the container at construction time. Exempt inside an annotation (`dict[str, bytes]` nests its
        args in a Tuple slice) — never invoked as a container."""
        if id(node) in self._annotation_nodes:
            return
        if _contains_banned_reference(node):
            self.report(
                node,
                "banned-construct",
                "a list/dict/set/tuple literal may not hold a reference to a banned builtin",
            )

    # ── operators ────────────────────────────────────────────────────────────────────────────
    def _require_bundle(self, node: ast.AST, bundle: str) -> None:
        """Guards against: using an operator bundle (arithmetic/comparison/indexing) the policy didn't
        enable for this file."""
        if not self.policy.bundle_enabled(bundle):
            ops = "/".join(t.__name__ for t in OPERATOR_BUNDLES[bundle])
            self.report(
                node, "bundle-disabled", f"{ops} needs the {bundle!r} method bundle"
            )

    # ── assignment / reserved names ──────────────────────────────────────────────────────────
    def _check_assign_targets(self, node) -> None:
        """Guards against: rebinding a reserved name, and stashing a banned-builtin into a container slot
        (`d["k"] = open`) — here the container name is Load context, so the reserved-name walk skips
        it and the stored value is otherwise never inspected."""
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for t in targets:
            self._check_reserved_target(t)
            if (
                isinstance(t, ast.Subscript)
                and node.value is not None
                and _contains_banned_reference(node.value)
            ):
                self.report(
                    t,
                    "banned-construct",
                    "storing a reference to a banned builtin into a container slot is not allowed",
                )
        self._track_self_attr_alias(node)

    def _track_self_attr_alias(self, node) -> None:
        """Detect `tmp = self.<attr>` / `tmp = self.<attr>[i]` / `tmp = <already-tracked alias>` so a
        later `tmp(...)` call in _check_call is checked the same way calling self.<attr> directly
        would be -- otherwise reading an unsafe self.<attr> into a local variable trivially
        defeats _SelfAttrTrust (self.<attr> reads are always allowed; calling a local variable is
        always allowed). Any other assignment to a previously-tracked name clears its alias -- it
        no longer refers to that self.<attr>. AugAssign always clears (no value to trace here)."""
        if not self._alias_stack:
            return
        aliases = self._alias_stack[-1]
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = (
                ([node.target] if node.value is not None else []),
                node.value,
            )
        else:  # AugAssign
            targets, value = [node.target], None
        for t in targets:
            if not isinstance(t, ast.Name):
                continue
            attr = self._self_attr_source(value) if value is not None else None
            if attr is not None:
                aliases[t.id] = attr
            else:
                aliases.pop(t.id, None)

    def _self_attr_source(self, value: ast.AST) -> str | None:
        """The self.<attr> a plain expression traces back to: a direct self.<attr> read, a
        single-level self.<attr>[i] subscript, or a copy of an already-tracked local alias."""
        attr = self_attr_name(value)
        if attr is not None:
            return attr
        if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Attribute):
            attr = self_attr_name(value.value)
            if attr is not None:
                return attr
        if isinstance(value, ast.Name):
            return self._current_aliases().get(value.id)
        return None

    def _current_aliases(self) -> dict[str, str]:
        return self._alias_stack[-1] if self._alias_stack else {}

    def _check_reserved_target(self, target: ast.AST) -> None:
        """Guards against: rebinding self/cls (which would forge the identifier-based trust — see
        _check_self_cls_params) or shadowing a reserved alias / wrapper name (below)."""
        for name_node in iter_names(target):
            if isinstance(name_node.ctx, ast.Store):
                if name_node.id in ("self", "cls"):
                    self.report(
                        name_node,
                        "reserved-name",
                        f"{name_node.id!r} may not be rebound; self/cls attribute access is "
                        f"trusted by identifier alone, not a verified reference to the real "
                        f"instance",
                    )
                    continue
                self._check_reserved_name(name_node, name_node.id)

    def _check_reserved_name(self, node: ast.AST, name: str) -> None:
        """Guards against: rebinding a trusted module alias (`jnp = evil`, which would poison the path
        resolver) or a visible wrapper name (`transpose = evil`)."""
        if name in self.policy.reserved_names:
            self.report(
                node,
                "reserved-name",
                f"{name!r} is a reserved module alias and may not be rebound",
            )
        elif name in self.scan.visible_defs:
            self.report(
                node,
                "reserved-name",
                f"{name!r} is a visible wrapper name and may not be rebound",
            )

    # ── path resolution ──────────────────────────────────────────────────────────────────────
    def _resolve(self, path: str) -> str:
        """Rewrite a dotted path's import alias to its fully-qualified form (`jnp.einsum` -> `jax.numpy.einsum`)."""
        root, _, rest = path.partition(".")
        base = self.scan.import_bindings.get(root, root)
        return f"{base}.{rest}" if rest else base

    def _resolved_allowed(self, path: str) -> bool:
        """Resolve an import alias, then apply the policy allow-list (disallow beats allow)."""
        return self.policy.function_allowed(self._resolve(path))


# ── self-attribute trust ────────────────────────────────────────────────────────────────────────
def _iter_self_attrs_in_target(target: ast.AST):
    """Yield every self.<attr>/cls.<attr> name found anywhere in a Store-context target tree,
    including nested inside tuple/list-unpacking (``self.a, self.b = ...``) or a starred target."""
    attr = self_attr_name(target)
    if attr is not None:
        yield attr
        return
    if isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            yield from _iter_self_attrs_in_target(elt)
    elif isinstance(target, ast.Starred):
        yield from _iter_self_attrs_in_target(target.value)


class _SelfAttrTrust:
    """Answers one question for ``_check_call_attribute``/``_check_self_subscript_call``: is
    ``self.<attr>`` (in the given enclosing class) safe to call or subscript?

    An attribute the class never assigns is presumed inherited from its (already vetted, since
    bases are allow-listed) base class -- e.g. nn.Module's ``self.param``/``self.variable``. Only
    attributes the class itself assigns are vetted against what was actually stored there. Results
    are memoized per class for the lifetime of one ``_Checker`` (one ``verify()`` call).
    """

    def __init__(self, scan: FileScan, resolved_allowed) -> None:
        self._scan = scan
        self._resolved_allowed = resolved_allowed
        self._cache: dict[int, dict[str, bool]] = {}  # id(ClassDef) -> {attr: is-safe}

    def is_safe(self, attr: str, cls_node: ast.ClassDef | None) -> bool:
        if cls_node is None:
            return False
        table = self._cache.get(id(cls_node))
        if table is None:
            table = self._build_table(cls_node)
            self._cache[id(cls_node)] = table
        return table.get(attr, True)

    def _build_table(self, cls_node: ast.ClassDef) -> dict[str, bool]:
        """For every self.<name> assigned anywhere in the class, record whether EVERY value ever stored
        there is a vetted-safe source. AugAssign always disqualifies (compound-mutating a submodule
        reference isn't a real Flax pattern and is inherently suspect) -- and so does a self.<attr>
        found nested in a tuple/list-unpack target or bound via a for-loop/comprehension target:
        neither is a real Flax pattern either, and unlike a plain `self.x = value` there's no single
        value expression to vet, so we can't tell what ends up in the attribute."""
        values: dict[str, list[ast.AST]] = {}
        disqualified: set[str] = set()
        for node in ast.walk(cls_node):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    attr = self_attr_name(t)
                    if attr is not None:
                        values.setdefault(attr, []).append(node.value)
                    elif isinstance(t, (ast.Tuple, ast.List)):
                        disqualified.update(_iter_self_attrs_in_target(t))
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                attr = self_attr_name(node.target)
                if attr is not None:
                    values.setdefault(attr, []).append(node.value)
            elif isinstance(node, ast.AugAssign):
                attr = self_attr_name(node.target)
                if attr is not None:
                    disqualified.add(attr)
            elif isinstance(node, (ast.For, ast.comprehension)):
                disqualified.update(_iter_self_attrs_in_target(node.target))
        table = {
            attr: attr not in disqualified and all(self._is_safe_value(v) for v in vs)
            for attr, vs in values.items()
        }
        for attr in disqualified:
            table.setdefault(attr, False)
        return table

    def _is_safe_value(self, value: ast.AST) -> bool:
        """A vetted "submodule" source: a call to an allow-listed library constructor or a name defined
        in this file (hidden or visible), or a list/tuple/set/comprehension of such (the layer idiom)."""
        if isinstance(value, ast.Call):
            func = value.func
            path = dotted_name(func)
            if path is not None and path.split(".")[0] in self._scan.import_bindings:
                return self._resolved_allowed(path)
            return isinstance(func, ast.Name) and (
                func.id in self._scan.hidden_defs or func.id in self._scan.visible_defs
            )
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            return all(self._is_safe_value(e) for e in value.elts)
        if isinstance(value, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            return self._is_safe_value(value.elt)
        if isinstance(value, ast.DictComp):
            return self._is_safe_value(value.value)
        return False


# ── helpers ──────────────────────────────────────────────────────────────────────────────────
def _contains_banned_reference(node: ast.AST) -> bool:
    """True if a Load-context reference to a BANNED_NAMES identifier appears anywhere in node."""
    return any(
        isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in BANNED_NAMES
        for n in ast.walk(node)
    )
