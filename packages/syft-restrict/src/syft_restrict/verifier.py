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
    node_in_ranges,
    normalize_ranges,
    rooted_in_self,
    scan_file,
    self_attr_name,
)
from .policy import (
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
    # operator/cmpop/boolop/unaryop singletons are leaf nodes under the above; always fine.
    ast.operator,
    ast.cmpop,
    ast.boolop,
    ast.unaryop,
    ast.expr_context,
)

# node-type deny-list: constructs deliberately, permanently denied (docs/blacklist.md) ──
# Listed explicitly (rather than just left off the allow-list) so their violation names them
# clearly, distinct from "node-type" (unreviewed/future syntax).
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
    # f-strings: even with no interpolation, just use a plain string; with interpolation, every
    # {expr} invokes type(expr).__format__(expr, spec) with no Call node for _check_call to see.
    ast.JoinedStr,
    ast.FormattedValue,
)

# violation-code registry: every code a check can raise, one line each (docs/blacklist.md)
ViolationCode = Literal[
    "banned-construct",  # _enforce (node type on the permanent deny-list)
    "node-type",  # _enforce (node type outside the always-on allow-list)
    "dunder-def",  # _check_def (defining a magic/hook method)
    "class-keyword",  # _check_class (metaclass= or other class keyword arg)
    "class-base",  # _check_class (non-allow-listed base class)
    "reserved-name",  # _check_name, _check_arguments_dont_abuse_self_or_cls, _check_reserved_name
    "banned-name",  # _check_call (banned bare call) / _check_name (any other Load reference)
    "call-not-allowed",  # _check_call, _check_call_attribute
    "call-unresolved",  # _check_call (bare-name/value call not traceable to a safe source)
    "dunder-attr",  # _check_call_attribute, _check_attribute
    "attr-on-value",  # _check_call_attribute, _check_self_subscript_call, _check_attribute
    "method-on-value",  # _check_call_attribute
    "attr-not-allowed",  # _check_attribute
    "dunder-name",  # _check_name
    "operator-disabled",  # _require_bundle
    "duplicate-method",  # _forbid_duplicate_methods
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
    # Copy policy so caller's reserved_names is never mutated across files.
    policy = policy.model_copy(update={"reserved_names": set(scan.import_bindings)})
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
#   f-strings (any shape)                -> _BANNED_NODES (in _enforce)
#   forged self/cls trust                -> _check_arguments_dont_abuse_self_or_cls, _check_name
#   aliasing a banned callable           -> _check_name (any Load reference, regardless of position)
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

        # answers "is self.<attr> safe to call?" -- see _SelfAttrTrust below. allow_base_class_attributes
        # controls whether a never-assigned attr is presumed inherited-safe or rejected.
        self._self_attr = _SelfAttrTrust(
            scan,
            self._resolved_allowed,
            allow_base_class=policy.allow_base_class_attributes,
        )

        # nodes inside a type annotation (never invoked), exempt from name/container checks
        self._annotation_nodes: set[int] = set()

        # per-scope locals provably safe to call. see _track_safe_local /
        # _is_safe_local_source. Starts with one frame already present because Module itself never
        # pushes a scope in visit() (only ClassDef/FunctionDef/Lambda do)
        self._safe_locals_stack: list[dict[str, bool]] = [{}]

        # when False, _track_safe_local records nothing, so a local aliased to a safe callable is
        # never itself trusted as a bare-name call target -- the callee must be called directly.
        self._allow_local_assignments = policy.allow_local_assignments

    def report(self, node: ast.AST, code: ViolationCode, message: str) -> None:
        self.violations.append(
            Violation(line=getattr(node, "lineno", 0), code=code, message=message)
        )

    # ── tree walk ───────────────────────────────────────────────────────────────────────────
    def visit(self, node: ast.AST) -> None:
        """Walk the whole tree; enforce only on nodes inside the private ranges, recurse everywhere."""
        if node_in_ranges(node, self.ranges):
            self._enforce(node)
        else:
            # private-defined names are reserved everywhere, including in public
            # region. Bare calls trust private_defs by identifier alone, so a
            # public-region rebind (helper = evil between private chunks) would
            # otherwise reopen the call-target hole.
            self._forbid_private_def_shadow_anywhere(node)

        if isinstance(node, ast.ClassDef):
            # self.<attr> trust reasoning (_SelfAttrTrust) looks at the whole class regardless
            # of the public/private split, so a duplicate method name must be caught the same
            # way, regardless of whether the class statement or either definition is private.
            self._forbid_duplicate_methods(node)

        # push/pop scope stack for ClassDef/FunctionDef/Lambda, so
        # _enclosing_class() works
        is_scope = isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.Lambda))
        if is_scope:
            self._scope_stack.append(node)
            self._safe_locals_stack.append({})
        self._mark_annotation_subtrees(node)

        # recurse to children, which may be outside the private ranges
        for child in ast.iter_child_nodes(node):
            self.visit(child)

        # pop scope stack after children, so _enclosing_class() sees the right frame
        if is_scope:
            self._scope_stack.pop()
            self._safe_locals_stack.pop()

    def _forbid_private_def_shadow_anywhere(self, node: ast.AST) -> None:
        """Reject rebinding a private-region class/def name even on public lines."""

        # only check rebinding (Store) and definitions (FunctionDef/ClassDef).
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            name_reserved = node.id in self.scan.private_defs
            name_rebound = name_reserved and id(node) not in self.scan.private_def_ids
            if name_rebound:
                self.report(
                    node,
                    "reserved-name",
                    f"{node.id!r} is a private-region class/def and may not be rebound",
                )
            return

        # methods are not bare-call targets and therefore not shadowed; shared names like `setup`` are fine
        if isinstance(node, ast.FunctionDef) and id(node) in self.scan.method_ids:
            return

        # only check rebinding (Store) and definitions (FunctionDef/ClassDef). A
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            name_reserved = node.name in self.scan.private_defs
            name_rebound = name_reserved and id(node) not in self.scan.private_def_ids
            if name_rebound:
                self.report(
                    node,
                    "reserved-name",
                    f"{node.name!r} is a private-region class/def and may not be rebound",
                )

    def _forbid_duplicate_methods(self, node: ast.ClassDef) -> None:
        """Reject defining the same method name twice directly in one class body.

        Python silently keeps only the last definition and discards the rest -- a reviewer (and
        _SelfAttrTrust) could be looking at a method body that never actually runs.
        """
        seen: set[str] = set()
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                if child.name in seen:
                    self.report(
                        child,
                        "duplicate-method",
                        f"{child.name!r} is already defined earlier in this class; Python keeps "
                        "only the last definition and silently discards the rest",
                    )
                seen.add(child.name)

    def _enclosing_class(self) -> ast.ClassDef | None:
        """The nearest enclosing ClassDef, skipping any FunctionDef/Lambda frames above it."""
        for frame in reversed(self._scope_stack):
            if isinstance(frame, ast.ClassDef):
                return frame
        return None

    def _mark_annotation_subtrees(self, node: ast.AST) -> None:
        # a type annotation (`x: str`, `x: list[str]`, `def f() -> dict[str,
        # bytes]`) is never invoked, so it can hold a name that is banned in a
        # real reference.
        for ann in (getattr(node, "annotation", None), getattr(node, "returns", None)):
            if ann is not None:
                # Mark the WHOLE subtree, not just its top node: generics nest the
                # type names one or more levels down
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

        # --- name binding (assignments, params) ---
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            self._track_safe_local(node)
        elif isinstance(node, ast.arg):
            self._check_reserved_name(node, node.arg)

        # --- literals & names ---
        elif isinstance(node, ast.Name):
            self._check_name(node)

    # ── definitions & classes ────────────────────────────────────────────────────────────────
    def _check_def(self, node: ast.FunctionDef) -> None:
        """Guards against:

        - a decorator running attacker code when the def is reached
        - defining magic/hook methods (__getattr__, __reduce__, …) that Python
          runs automatically without an explicit call
        - shadowing a trusted module alias or public wrapper name with a local
          def

        """

        # decorators are not allowed in the private region
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
        - a decorator running attacker code when the class is reached
        - shadowing a trusted module alias with a local class name.
        """
        # decorators are not allowed in the private region
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
        """Reject every decorator on a def/class: a decorator runs code the moment the def/class
        is reached, so it is banned outright in the private region."""
        for dec in node.decorator_list:
            self.report(
                dec,
                "banned-construct",
                "decorators are not allowed in the private region",
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

        # count total calls for the final result
        self.n_calls += 1
        func = node.func

        # if it's a bare name, check it against the allow-list and the private defs
        if isinstance(func, ast.Name):
            # banned builtins: report once here and mark so _check_name does not double-count
            if func.id in BANNED_NAMES:
                self.report(node, "banned-name", f"call to {func.id!r} is not allowed")
                self._checked_call_targets.add(id(func))
                return
            if func.id in self.scan.import_bindings:
                # resolves through the same import-binding table as a dotted call
                if not self._resolved_allowed(func.id):
                    self.report(
                        node,
                        "call-not-allowed",
                        f"call to {self._resolve(func.id)!r} is not allow-listed",
                    )
                return

            # a def/class here, a safe builtin, or a local traced to one
            if (
                func.id in self.scan.private_defs
                or func.id in self.scan.public_defs
                or func.id in SAFE_BUILTIN_CALLS
                or self._current_safe_locals.get(func.id, False)
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

        # given x.y.z(...), this checks the Attribute call target x.y
        if isinstance(func, ast.Attribute):
            self._check_call_attribute(node, func)
            return

        # given self.x[i](...), this checks the self-rooted subscript call
        if isinstance(func, ast.Subscript) and rooted_in_self(func):
            self._check_self_subscript_call(node, func)
            return

        # if we get here, the call is not banned by name or attribute, but must
        # still resolve to a safe source
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
        # add it to the checked set so _check_attribute doesn't double-count it
        self._checked_call_targets.add(id(func))

        # resolve the dotted path to a string, if possible. If not, it's a named
        # method on an opaque value (x.reshape(...)) whose type we can't pin, so deny it.
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

        - rebinding self/cls
        - rebinding a reserved name
        - loading a banned builtin (`open`, `eval`, `exec`, ...)
        - loading a bare dunder name (`__class__`, `__dict__`, ...)
        """
        if isinstance(node.ctx, ast.Store):
            # self/cls itself: always denied, never delegated to _check_reserved_name
            if node.id in ("self", "cls"):
                self.report(
                    node,
                    "reserved-name",
                    f"{node.id!r} may not be rebound; self/cls attribute access is "
                    f"trusted by identifier alone, not a verified reference to the real "
                    f"instance",
                )
                return
            # any other Store target: is this name reserved by the resolver?
            self._check_reserved_name(node, node.id)
            return

        # only a reference (Load) can leak/dispatch a name; a Del target is bound elsewhere
        if not isinstance(node.ctx, ast.Load):
            return

        # skip if already reported as the func of a banned call by _check_call
        if id(node) in self._checked_call_targets:
            return

        # a reference to a banned builtin, anywhere it can appear, is denied
        if node.id in BANNED_NAMES:
            # `x: str` / `def f() -> str` — a type annotation, not a reference
            if id(node) in self._annotation_nodes:
                return
            self.report(node, "banned-name", f"reference to {node.id!r} is not allowed")

        # a bare dunder name
        elif is_dunder(node.id):
            self.report(
                node,
                "dunder-name",
                f"reference to dunder name {node.id!r} is not allowed",
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
                node,
                "operator-disabled",
                f"{ops} needs the {bundle!r} operator group in allow_operators",
            )

    # ── assignment / reserved names ──────────────────────────────────────────────────────────
    @staticmethod
    def _assignment_targets_and_value(node) -> tuple[list[ast.expr], ast.expr | None]:
        """Normalize Assign/AnnAssign/AugAssign into a uniform (targets, value) pair."""
        if isinstance(node, ast.Assign):
            return node.targets, node.value
        if isinstance(node, ast.AnnAssign):
            return ([node.target] if node.value is not None else []), node.value
        return [node.target], None  # AugAssign

    def _track_safe_local(self, node) -> None:
        """Track whether each assigned local is a provably-safe call target.

        Any other assignment to a previously-tracked name clears its verdict --
        it no longer traces to that source."""

        # Local-alias tracking disabled: never record any local as safe, so a bare-name call to a
        # local (block = self.layers[0]; block(x)) falls through to call-unresolved. Callables must
        # be called directly (self.layers[0](x)) instead.
        if not self._allow_local_assignments:
            return

        # Module itself never pushes a scope in visit(), so we added a frame at
        # init and never pop it. The stack should never be empty.
        if not self._safe_locals_stack:
            raise RuntimeError(
                "safe_locals_stack is empty; visit() should have pushed a frame"
            )

        # Track the verdict in the top frame of the stack, which is the current
        # scope.
        safe = self._safe_locals_stack[-1]
        targets, value = self._assignment_targets_and_value(node)

        # check each target in the assignment. If it's a Name, track whether
        # it's provably safe to call. If it's not a Name (e.g., a tuple
        # unpacking), skip it. If the value is None (AnnAssign without a value),
        # clear the verdict.
        for t in targets:
            if not isinstance(t, ast.Name):
                continue
            if value is not None and self._is_safe_local_source(value):
                safe[t.id] = True
            else:
                safe.pop(t.id, None)

    def _is_safe_local_source(self, value: ast.AST) -> bool:
        """Is a local provably safe to call?"""
        attr = self_attr_name(value)
        if attr is None and isinstance(value, ast.Subscript):
            attr = self_attr_name(value.value)
        if attr is not None:
            return self._self_attr.is_safe(attr, self._enclosing_class())
        return _all_leaves_safe(value, self._is_safe_local_leaf)

    def _is_safe_local_leaf(self, value: ast.AST) -> bool:
        if isinstance(value, ast.Name):
            return (
                self._current_safe_locals.get(value.id, False)
                or value.id in self.scan.private_defs
                or value.id in self.scan.public_defs
                or value.id in SAFE_BUILTIN_CALLS
            )
        if isinstance(value, ast.Call):
            return self._self_attr.is_safe_value(value)
        return False

    @property
    def _current_safe_locals(self) -> dict[str, bool]:
        return self._safe_locals_stack[-1] if self._safe_locals_stack else {}

    def _check_reserved_name(self, node: ast.AST, name: str) -> None:
        """Guards against:

        - rebinding a trusted module alias (`jnp = evil`), which would poison the path resolver
        - rebinding a private-region class/def (`Attn = evil`), which would silently redirect
          every bare call that appears to route through the vetted original -- private_defs is
          a scope-blind, name-only whole-file scan, so nothing else notices the shadow
        - rebinding a public wrapper name (`transpose = evil`), defeating its type guard
        - rebinding a safe builtin (`list = evil`) — _check_call trusts a bare call to any of
          these four by identifier alone, so shadowing one would silently redirect every call
          site that appears to route through it
        """
        if name in self.policy.reserved_names:
            self.report(
                node,
                "reserved-name",
                f"{name!r} is a reserved module alias and may not be rebound",
            )
        elif (
            name in self.scan.private_defs and id(node) not in self.scan.private_def_ids
        ):
            self.report(
                node,
                "reserved-name",
                f"{name!r} is a private-region class/def and may not be rebound",
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
        """Rewrite a dotted path's import alias to its fully-qualified form
        (`jnp.einsum` -> `jax.numpy.einsum`)."""
        root, _, rest = path.partition(".")
        base = self.scan.import_bindings.get(root, root)
        return f"{base}.{rest}" if rest else base

    def _resolved_allowed(self, path: str) -> bool:
        """Resolve an import alias, then apply the policy allow-list (disallow
        beats allow)."""
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


def _all_leaves_safe(value: ast.AST, is_safe_leaf) -> bool:
    """Recurse through list/tuple/set literals and list/set/dict comprehensions/generator
    expressions down to their leaf elements (the layer idiom, e.g. ``[Block(cfg) for _ in
    range(n)]``), checking each leaf with ``is_safe_leaf``. Shared by
    ``_SelfAttrTrust.is_safe_value`` and ``_Checker._is_safe_local_source``."""
    if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
        return all(_all_leaves_safe(e, is_safe_leaf) for e in value.elts)
    if isinstance(value, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        return _all_leaves_safe(value.elt, is_safe_leaf)
    if isinstance(value, ast.DictComp):
        return _all_leaves_safe(value.value, is_safe_leaf)
    return is_safe_leaf(value)


class _SelfAttrTrust:
    """Used by ``_check_call_attribute``/``_check_self_subscript_call`` to
    determine if ``self.<attr>`` (in the given enclosing class) safe to call or
    subscript.

    An attribute the class never assigns is presumed inherited from its (already
    vetted, since bases are allow-listed) base class,  nn.Module's
    ``self.param``/``self.variable``.

    Only attributes the class itself assigns are vetted against what was
    actually stored there. Results are memoized per class for the lifetime of
    one ``_Checker`` (one ``verify()`` call).
    """

    def __init__(self, scan: FileScan, resolved_allowed, allow_base_class: bool = True) -> None:
        self._scan = scan
        self._resolved_allowed = resolved_allowed
        # the verdict for a self.<attr> never assigned in the class body: True presumes it is
        # inherited from the (already-vetted) base class; False rejects it (docs/verify.md).
        self._allow_base_class = allow_base_class
        self._cache: dict[int, dict[str, bool]] = {}  # id(ClassDef) -> {attr: is-safe}

    def is_safe(self, attr: str, cls_node: ast.ClassDef | None) -> bool:
        # if the class is None, we're not in a class body, so self.<attr> is not
        # allowed
        if cls_node is None:
            return False

        # check the memoized table for this class; if not present, build it
        table = self._cache.get(id(cls_node))
        if table is None:
            table = self._build_table(cls_node)
            self._cache[id(cls_node)] = table

        # a safe/unsafe verdict for an attr the class assigns; for one it never assigns, fall back
        # to _allow_base_class (presumed inherited from a vetted base, unless the caller disabled it)
        return table.get(attr, self._allow_base_class)

    def _build_table(self, cls_node: ast.ClassDef) -> dict[str, bool]:
        """For every self.<name> assigned anywhere in the class, record whether
        EVERY value ever stored there is a vetted-safe source.
        """

        # AugAssign always disqualifies (compound-mutating a submodule reference
        # isn't a real pattern and is inherently suspect)
        #
        # So does a self.<attr> found nested in a tuple/list-unpack target or
        # bound via a for-loop/comprehension target: neither is a real pattern
        # either, and unlike a plain `self.x = value` there's no single value
        # expression to vet, so we can't tell what ends up in the attribute.
        #
        # So does a class-level dataclass-style field (`name: Type` / `name: Type = default`,
        # directly in the class body, not `self.name`): its runtime value comes from whatever
        # constructs the instance, never from anything textually inside the class, so there's no
        # expression here to vet at all -- unlike a genuinely inherited base-class attribute
        # (e.g. nn.Module's self.param), which has no class-level annotation of its own.

        values: dict[str, list[ast.AST]] = {}
        disqualified: set[str] = {
            node.target.id
            for node in cls_node.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }

        for node in ast.walk(cls_node):
            # if the node is an assignment, record the value(s) assigned to self.<attr>
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    attr = self_attr_name(t)
                    if attr is not None:
                        values.setdefault(attr, []).append(node.value)
                    elif isinstance(t, (ast.Tuple, ast.List)):
                        disqualified.update(_iter_self_attrs_in_target(t))

            # if the node is an annotated assignment with a value, record the
            # value assigned to self.<attr>
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                attr = self_attr_name(node.target)
                if attr is not None:
                    values.setdefault(attr, []).append(node.value)
            # if the node is an AugAssign, disqualify the attribute
            elif isinstance(node, ast.AugAssign):
                attr = self_attr_name(node.target)
                if attr is not None:
                    disqualified.add(attr)
            # if the node is a for-loop or comprehension, disqualify any
            # self.<attr> in the target
            elif isinstance(node, (ast.For, ast.comprehension)):
                disqualified.update(_iter_self_attrs_in_target(node.target))
        table = {
            attr: attr not in disqualified and all(self.is_safe_value(v) for v in vs)
            for attr, vs in values.items()
        }

        # any self.<attr> that was disqualified but never assigned a value is
        # still disqualified
        for attr in disqualified:
            table.setdefault(attr, False)

        return table

    def is_safe_value(self, value: ast.AST) -> bool:
        """A vetted "submodule" source: a call to an allow-listed library constructor or a name defined
        in this file (private or public), or a list/tuple/set/comprehension of such (the layer idiom)."""
        return _all_leaves_safe(value, self._is_safe_call)

    def _is_safe_call(self, value: ast.AST) -> bool:
        if not isinstance(value, ast.Call):
            return False
        func = value.func
        path = dotted_name(func)
        if path is not None and path.split(".")[0] in self._scan.import_bindings:
            return self._resolved_allowed(path)
        return isinstance(func, ast.Name) and (
            func.id in self._scan.private_defs or func.id in self._scan.public_defs
        )
