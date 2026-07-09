"""The static checker — verifies that the private region only does trusted math.

``verify(source, private, policy)`` parses the file, restricts attention to the *private* line ranges
(the private model definition), and walks those nodes **default-deny**: a node is allowed only if a
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
    SAFE_BUILTIN_CALLS,
    Policy,
)

# ── node-type allow-list: syntax the private region may use (docs/verify.md#always-on-allow-list) ──
# Anything else is denied by default (docs/blacklist.md).
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
# Listed explicitly so their violation names them clearly.
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
# Grep a method name below to find which check raises which code.
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
    "call-unresolved",  # _check_call (bare-name/value call not traceable to a safe source)
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
# One check per node, default-deny. Rationale: docs/verify.md; deny lists: docs/blacklist.md.
#
#   dynamic-code / reflection builtins   -> _check_call, _check_name
#   IO / host-escape statements          -> _BANNED_NODES (in _enforce)
#   unknown / future syntax              -> _ALLOWED_NODES default-deny (in _enforce)
#   library call/attr by name            -> _check_call_attribute, _check_attribute (resolver + allow/disallow)
#   named method / attr on opaque value  -> _check_call_attribute, _check_attribute
#   f-string repr/str/ascii escape       -> _check_formatted_value
#   forged self/cls trust                -> _check_self_cls_params, _check_reserved_target
#   aliasing a banned callable           -> _check_name, _check_container_literal, _check_assign_targets
#   unresolved call target (default-deny) -> _check_call, _is_safe_local_source
#   class-creation hooks                 -> _check_class, _check_decorators, _check_def
# ──────────────────────────────────────────────────────────────────────────────────────────────
class _Checker:
    def __init__(self, policy: Policy, scan: FileScan, ranges):
        self.policy = policy
        self.scan = scan
        self.ranges = ranges
        self.violations: list[Violation] = []
        self.n_calls = 0

        # nodes already judged as a call's func, so _check_attribute/_check_name skip them
        self._checked_call_targets: set[int] = set()

        # enclosing class/def/lambda stack, for _enclosing_class() and self/cls-position checks
        self._scope_stack: list[ast.AST] = []

        # answers "is self.<attr> safe to call?" -- see _SelfAttrTrust below
        self._self_attr = _SelfAttrTrust(scan, self._resolved_allowed)

        # nodes inside a type annotation (never invoked), exempt from name/container checks
        self._annotation_nodes: set[int] = set()

        # per-scope {local: self_attr} aliases, so `tmp = self.fn; tmp(x)` is checked like `self.fn(x)`
        self._alias_stack: list[dict[str, str]] = []

        # per-scope locals provably safe to call, beyond self.<attr> aliases -- see _track_safe_local
        self._safe_locals_stack: list[dict[str, bool]] = []

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
            self._safe_locals_stack.append({})
        self._mark_annotation_subtrees(node)

        for child in ast.iter_child_nodes(node):
            self.visit(child)

        if is_scope:
            self._scope_stack.pop()
            self._alias_stack.pop()
            self._safe_locals_stack.pop()

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
        """Run the one check that applies to this node type (recursion is
        handled by visit)."""
        if isinstance(node, _BANNED_NODES):
            self.report(
                node,
                "banned-construct",
                f"{type(node).__name__} is not allowed in the private region",
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
            self._check_lambda(node)
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
            self._check_for_loop_assignment(node)
        elif isinstance(
            node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        ):
            self._check_comprehension_target(node)
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
        """Guards against:

        - defining magic/hook methods (__getattr__, __reduce__, …) that Python
          runs automatically without an explicit call
        - shadowing a trusted module alias or public wrapper name with a local
          def

        """

        # are the function decorators in the list of allowed decorators?
        self._check_decorators(node)

        # is the function name not a reserved name?
        self._check_reserved_name(node, node.name)

        # if it's a dunder, is it an allowed dunder?
        if is_dunder(node.name) and node.name not in ALLOWED_DUNDER_DEFS:
            self.report(
                node,
                "dunder-def",
                f"defining magic method {node.name!r} is not allowed",
            )
        # if the function has cls or self, only allow if it is the first argument for a method
        self._check_arguments_dont_abuse_self_or_cls(node.args)

    def _check_lambda(self, node: ast.Lambda) -> None:
        """Guards against:

        - a lambda's parameters being named self/cls, forging the self/cls
        trust exemption."""
        self._check_arguments_dont_abuse_self_or_cls(node.args)

    def _check_class(self, node: ast.ClassDef) -> None:
        """Guards against:

        - banned base classes
        - a class decorator running attacker code when the class is reached
        - shadowing a trusted module alias with a local class name.
        """
        # check only allowed decorators
        self._check_decorators(node)

        # is the class name not a reserved name?
        self._check_reserved_name(node, node.name)

        # does it use class keyword arguments (like metaclass=)?
        if node.keywords:
            self.report(
                node,
                "class-keyword",
                "class keyword arguments (e.g. metaclass=) are not allowed",
            )

        # if it has base classes, are they allow-listed?
        for base in node.bases:
            if not self._is_base_class_allowed(base):
                self.report(
                    base,
                    "class-base",
                    f"base class {describe(base)!r} is not allow-listed",
                )

    def _is_base_class_allowed(self, base: ast.AST) -> bool:
        """Only an allow-listed class may be a base class."""
        path = dotted_name(base)
        return bool(path) and self._resolved_allowed(path)

    def _check_decorators(self, node) -> None:
        """Guards against:

        - a decorator running attacker code when a def/class is reached.

        """
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
        """Guards against:

        - a parameter named self/cls anywhere except the first parameter of a
          method in a class definition

        """
        # is this def/lambda a method defined directly in a class body?
        is_direct_method = bool(self._scope_stack) and isinstance(
            self._scope_stack[-1], ast.ClassDef
        )

        # which parameter, if any, is genuinely first (posonly takes precedence
        # over plain args)
        first = (
            args.posonlyargs[0]
            if args.posonlyargs
            else (args.args[0] if args.args else None)
        )

        # every parameter this def/lambda declares, including *args/**kwargs
        all_params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
        if args.vararg is not None:
            all_params.append(args.vararg)
        if args.kwarg is not None:
            all_params.append(args.kwarg)

        # flag self/cls named anywhere except the first position
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
        """Guards against: calling a non-vetted callable.

        Default-deny semantics. A call target is allowed only if it's provably
        one of a small set of safe shapes:

        - a safe builtin
        - an allow-listed import
        - a def/class defined in this file
        - a self.<attr> vetted by _SelfAttrTrust
        - a local/value traced back to one of those (see _is_safe_local_source)

        Anything else (a parameter, an untraceable local, an unknown name) is
        rejected outright: we can't prove it's safe, so per policy we'd rather
        force the author to route it through self.<attr> or a public wrapper
        than risk trusting an opaque callable

        (docs/verify.md#the-full-call-target-rule).

        Attribute-position calls route to a stricter check."""

        self.n_calls += 1
        func = node.func

        if isinstance(func, ast.Name):
            # banned builtins (eval, open, ...) are caught by _check_name
            if func.id in self.scan.import_bindings:
                # resolves through the same import-binding table as a dotted call
                if not self._resolved_allowed(func.id):
                    self.report(
                        node,
                        "call-not-allowed",
                        f"call to {self._resolve(func.id)!r} is not allow-listed",
                    )
                return

            # a local aliasing self.<attr> -- vetted the same way self.<attr>(...) would be
            aliased_attr = self._current_aliases().get(func.id)
            if aliased_attr is not None:
                if not self._self_attr.is_safe(aliased_attr, self._enclosing_class()):
                    self.report(
                        node,
                        "attr-on-value",
                        f"{func.id!r} was assigned from self.{aliased_attr!r}, which isn't an "
                        f"allow-listed constructor or a locally-defined class; calling it is not "
                        f"allowed",
                    )
                return
            # a def/class here, a safe builtin, or a local traced to one (_is_safe_local_source)
            if (
                func.id in self.scan.private_defs
                or func.id in self.scan.public_defs
                or func.id in SAFE_BUILTIN_CALLS
                or self._current_safe_locals().get(func.id, False)
            ):
                return
            self.report(
                node,
                "call-unresolved",
                f"{func.id!r}: could not unambiguously identify what this calls; only an "
                f"allow-listed import, a def/class defined in this file, a safe builtin, or a "
                f"local traced to one of those may be called",
            )
            return

        # given x.y.z(), this checks x.y
        if isinstance(func, ast.Attribute):
            self._check_call_attribute(node, func)
            return

        # given x[i].z(), this checks x[i])
        if isinstance(func, ast.Subscript) and rooted_in_self(func):
            self._check_self_subscript_call(node, func)
            return

        # the call is trusted only if it's a provably-safe source
        if not self._is_safe_local_source(func):
            self.report(
                node,
                "call-unresolved",
                "could not unambiguously identify what this calls; route the callee through "
                "self.<attr>, a local traced to an allow-listed constructor, or a public wrapper "
                "instead",
            )

    # Intentionally parallel with _check_self_subscript_call/_check_attribute -- don't merge them.
    def _check_call_attribute(self, call: ast.Call, func: ast.Attribute) -> None:
        """Guards against:

        - calling a dunder attribute off self/cls (self.__class__(...))
        - calling self.<attr>(...) that wasn't inherited or assigned a vetted-safe source (see
          _SelfAttrTrust)
        - calling a deeper self-rooted chain (self.a.b(...)), not a single self.<attr> level
        - calling a non-allow-listed library path
        - calling a named method on an opaque value (x.reshape(...)) whose type — and thus what
          the call does — we can't pin
        """
        self._checked_call_targets.add(
            id(func)
        )  # so _check_attribute won't re-flag this node
        path = dotted_name(func)
        if path is not None:
            root = path.split(".")[0]
            if root in ("self", "cls"):
                attr = self_attr_name(func)
                # a dunder off self/cls (self.__class__(...)) is always denied
                if attr is not None and is_dunder(attr):
                    self.report(
                        call,
                        "dunder-attr",
                        f"access to dunder attribute {attr!r} is not allowed",
                    )
                    return
                # self.<attr>(...) — allowed only if <attr> is inherited or vetted safe
                if attr is not None and self._self_attr.is_safe(
                    attr, self._enclosing_class()
                ):
                    return  # self.<name>(...) — <name> is inherited or was assigned a vetted source
                # attr is None: a deeper chain (self.a.b(...)); otherwise attr was assigned unsafely
                message = (
                    f"self.{attr!r} was assigned a value that isn't an allow-listed constructor "
                    f"or a locally-defined class; calling it is not allowed"
                    if attr is not None
                    else f"{path!r}: only a single self.<name> attribute may be called, "
                    f"not a deeper attribute chain"
                )
                self.report(call, "attr-on-value", message)
                return
            # a non-self dotted path: must resolve to an allow-listed import
            if root in self.scan.import_bindings:
                if not self._resolved_allowed(path):
                    self.report(
                        call,
                        "call-not-allowed",
                        f"call to {self._resolve(path)!r} is not allow-listed",
                    )
                return
        # not a self/cls chain and not import-rooted: a named method on an opaque value
        self.report(
            call,
            "method-on-value",
            f"named method {func.attr!r} called on a value whose type is unknown; "
            f"route it through a public wrapper function instead",
        )

    def _check_self_subscript_call(self, call: ast.Call, func: ast.Subscript) -> None:
        """Guards against:

        - the self.layer[i](x) idiom smuggling a tainted callable. Only allowed
          when the callable is inherited or was assigned a safe source
        - calling an element of a deeper self-rooted chain (self.a.b[i](x)), not
          a single self.<name>[...] level
        """
        # only a single self.<name>[...] level is ever eligible; a deeper chain leaves attr None
        attr = (
            self_attr_name(func.value)
            if isinstance(func.value, ast.Attribute)
            else None
        )
        # self.<name>[...] — allowed only if <name> is inherited or vetted safe
        if attr is not None and self._self_attr.is_safe(attr, self._enclosing_class()):
            return
        # attr is None: a deeper chain; otherwise <name> was assigned an unsafe value
        message = (
            f"self.{attr!r}[...] was assigned a value that isn't a list/tuple of allow-listed "
            f"constructors; calling an element of it is not allowed"
            if attr is not None
            else "only self.<name>[...] may be called this way, not a deeper self-rooted chain"
        )
        self.report(call, "attr-on-value", message)

    # ── attribute reads (not the func of a call) ─────────────────────────────────────────────
    def _check_attribute(self, node: ast.Attribute) -> None:
        """Guards against:

        - a dunder attribute read (x.__class__, obj.__dict__)
        - a self-rooted chain deeper than a single self.<name>/cls.<name> level (self.a.b)
        - a non-allow-listed library path (np.dot when only jax.* is allowed)
        - an attribute read on an opaque value (x.shape, x.T, x.ndim)
        """
        if id(node) in self._checked_call_targets:
            return  # already judged as a call's function position by _check_call_attribute
        # a bare dunder attribute (x.__class__) is always denied
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
                # a single self.<name>/cls.<name> read is fine; a deeper chain is not
                if self_attr_name(node) is not None:
                    return
                self.report(
                    node,
                    "attr-on-value",
                    f"{path!r}: only a single self.<name> attribute may be accessed, "
                    f"not a deeper attribute chain",
                )
                return
            # a non-self dotted path: must resolve to an allow-listed import
            if root in self.scan.import_bindings:
                if not self._resolved_allowed(path):
                    self.report(
                        node,
                        "attr-not-allowed",
                        f"reference to {self._resolve(path)!r} is not allow-listed",
                    )
                return
        # not self/cls-rooted and not import-rooted: an attribute read on an opaque value
        self.report(
            node,
            "attr-on-value",
            f"attribute {node.attr!r} on a value is not allowed; "
            f"route it through a public wrapper function instead",
        )

    # ── bare name reads (not the func of a call) ─────────────────────────────────────────────
    def _check_name(self, node: ast.Name) -> None:
        """Guards against:

        - loading a banned builtin (`f = open; f(...)`)
        - loading bare dunder name (`__class__`)
        """
        # only a reference (Load) can leak/dispatch a name; a Store/Del target is bound elsewhere
        if not isinstance(node.ctx, ast.Load):
            return

        # ignore if already reported as banned-call by _check_call
        if id(node) in self._checked_call_targets:
            return

        # a reference to a banned builtin, anywhere it can appear, is denied
        if node.id in BANNED_NAMES:
            # `x: str` / `def f() -> str` — a type annotation, not a reference
            if id(node) in self._annotation_nodes:
                return
            self.report(node, "banned-call", f"reference to {node.id!r} is not allowed")

        # a bare dunder name
        elif is_dunder(node.id):
            self.report(
                node,
                "dunder-name",
                f"reference to dunder name {node.id!r} is not allowed",
            )

    # ── f-strings ────────────────────────────────────────────────────────────────────────────
    def _check_formatted_value(self, node: ast.FormattedValue) -> None:
        """Guards against:

        - every f-string interpolation — plain `f"{x}"` included — invoking __format__ (and, via
          default object.__format__, __str__) on the value with no Call node for _check_call to
          see; Python calls type(x).__format__(x, spec) for every FormattedValue regardless of
          conversion flag, so there is no conversion-less case that "stays allowed"
        """
        # unconditional: no FormattedValue is ever a provably-safe __format__ call
        self.report(
            node,
            "method-on-value",
            "f-string interpolation calls __format__ on a value whose type is unknown; "
            "route it through a public wrapper function instead",
        )

    # ── container literals ───────────────────────────────────────────────────────────────────
    def _check_container_literal(self, node) -> None:
        """Guards against:

        - stashing a banned-builtin reference in a list/dict/set/tuple for later dispatch
          (`d = {"o": open}; d["o"](...)`) — we don't track which slot holds what, so we reject
          the container at construction time
        """
        # generic annotations nest type names in a Tuple slice, never invoked -- exempt
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
        """Guards against:

        - using an operator bundle (arithmetic/comparison/indexing) the policy didn't enable for
          this file
        """
        if not self.policy.bundle_enabled(bundle):
            ops = "/".join(t.__name__ for t in OPERATOR_BUNDLES[bundle])
            self.report(
                node, "bundle-disabled", f"{ops} needs the {bundle!r} method bundle"
            )

    # ── assignment / reserved names ──────────────────────────────────────────────────────────
    def _check_assign_targets(self, node) -> None:
        """Guards against:

        - rebinding a reserved name (an import alias, a public wrapper, a safe builtin, or
          self/cls) — see _check_reserved_target
        - stashing a banned-builtin into a container slot (`d["k"] = open`) — here the container
          name is Load context, so the reserved-name walk skips it and the stored value is
          otherwise never inspected
        """
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for t in targets:
            # is the assigned name itself reserved?
            self._check_reserved_target(t)
            # is a banned builtin being stashed into a container slot for later dispatch?
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
        # not a guard: bookkeeping for _check_call's local-safety tracking, not a violation source
        self._track_self_attr_alias(node)
        self._track_safe_local(node)

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

    def _track_safe_local(self, node) -> None:
        """Track whether each assigned local is a provably-safe call target, beyond the self.<attr>
        aliasing _track_self_attr_alias already covers: an allow-listed constructor call, a def/class
        defined in this file, a copy of an already-tracked-safe name, or a container/comprehension of
        such (see _is_safe_local_source) -- so _check_call can allow calling it later instead of
        denying by default. Any other assignment to a previously-tracked name clears its verdict, the
        same as _track_self_attr_alias; AugAssign always clears (no single value expression to vet)."""
        if not self._safe_locals_stack:
            return
        safe = self._safe_locals_stack[-1]
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
            if value is not None and self._is_safe_local_source(value):
                safe[t.id] = True
            else:
                safe.pop(t.id, None)

    def _is_safe_local_source(self, value: ast.AST) -> bool:
        """Is a local (or a value called directly, e.g. ``Block(...)(x)``) provably safe to call?
        self.<attr> sources are handled separately by _track_self_attr_alias/_current_aliases
        (checked first in _check_call); this covers everything else: a call to an allow-listed
        constructor or a def/class in this file (delegated to _SelfAttrTrust.is_safe_value, the same
        rule self-attribute assignments are vetted against), a bare-name copy of an already-tracked
        source or another def/class in this file, or a list/tuple/set/comprehension of such (the
        layer idiom, e.g. ``blocks = [Block(cfg) for _ in range(n)]``)."""
        if isinstance(value, ast.Name):
            return (
                self._current_safe_locals().get(value.id, False)
                or value.id in self.scan.private_defs
                or value.id in self.scan.public_defs
            )
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            return all(self._is_safe_local_source(e) for e in value.elts)
        if isinstance(value, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            return self._is_safe_local_source(value.elt)
        if isinstance(value, ast.DictComp):
            return self._is_safe_local_source(value.value)
        if isinstance(value, ast.Call):
            return self._self_attr.is_safe_value(value)
        return False

    def _current_safe_locals(self) -> dict[str, bool]:
        return self._safe_locals_stack[-1] if self._safe_locals_stack else {}

    def _check_for_loop_assignment(self, node: ast.For) -> None:
        """Guards against:

        - rebinding a reserved name in a for-loop target, e.g. ``for self in [evil]: pass``
        """
        # a for-loop target binds like an assignment but isn't an ast.Assign node -- same check
        self._check_reserved_target(node.target)

    def _check_comprehension_target(
        self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
    ) -> None:
        """Guards against:

        - rebinding a reserved name in a comprehension target
        """
        # ast.comprehension has no lineno, so dispatch from the enclosing expression instead
        for generator in node.generators:
            self._check_reserved_target(generator.target)

    def _check_reserved_target(self, target: ast.AST) -> None:
        """Guards against:

        - rebinding self/cls, which would forge the identifier-based trust
          _check_arguments_dont_abuse_self_or_cls and self.<attr> vetting both rely on
        - shadowing a reserved alias / wrapper name / safe builtin (delegated to
          _check_reserved_name)
        """
        for name_node in iter_names(target):
            if isinstance(name_node.ctx, ast.Store):
                # self/cls itself: always denied, never delegated to _check_reserved_name
                if name_node.id in ("self", "cls"):
                    self.report(
                        name_node,
                        "reserved-name",
                        f"{name_node.id!r} may not be rebound; self/cls attribute access is "
                        f"trusted by identifier alone, not a verified reference to the real "
                        f"instance",
                    )
                    continue
                # any other Store target: is this name reserved by the resolver?
                self._check_reserved_name(name_node, name_node.id)

    def _check_reserved_name(self, node: ast.AST, name: str) -> None:
        """Guards against:

        - rebinding a trusted module alias (`jnp = evil`), which would poison the path resolver
        - rebinding a public wrapper name (`transpose = evil`), defeating its type guard
        - rebinding a safe builtin (`list = evil`) — _check_call trusts a bare call to any of
          these three by identifier alone, so shadowing one would silently redirect every call
          site that appears to route through it
        """
        if name in self.policy.reserved_names:
            self.report(
                node,
                "reserved-name",
                f"{name!r} is a reserved module alias and may not be rebound",
            )
        elif name in self.scan.public_defs:
            self.report(
                node,
                "reserved-name",
                f"{name!r} is a public wrapper name and may not be rebound",
            )
        elif name in SAFE_BUILTIN_CALLS:
            self.report(
                node,
                "reserved-name",
                f"{name!r} is a trusted builtin and may not be rebound",
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
            attr: attr not in disqualified and all(self.is_safe_value(v) for v in vs)
            for attr, vs in values.items()
        }
        for attr in disqualified:
            table.setdefault(attr, False)
        return table

    def is_safe_value(self, value: ast.AST) -> bool:
        """A vetted "submodule" source: a call to an allow-listed library constructor or a name defined
        in this file (private or public), or a list/tuple/set/comprehension of such (the layer idiom)."""
        if isinstance(value, ast.Call):
            func = value.func
            path = dotted_name(func)
            if path is not None and path.split(".")[0] in self._scan.import_bindings:
                return self._resolved_allowed(path)
            return isinstance(func, ast.Name) and (
                func.id in self._scan.private_defs or func.id in self._scan.public_defs
            )
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            return all(self.is_safe_value(e) for e in value.elts)
        if isinstance(value, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            return self.is_safe_value(value.elt)
        if isinstance(value, ast.DictComp):
            return self.is_safe_value(value.value)
        return False


# ── helpers ──────────────────────────────────────────────────────────────────────────────────
def _contains_banned_reference(node: ast.AST) -> bool:
    """True if a Load-context reference to a BANNED_NAMES identifier appears anywhere in node."""
    return any(
        isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in BANNED_NAMES
        for n in ast.walk(node)
    )
