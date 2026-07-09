import pytest
from syft_restrict import verify

from .conftest import error_codes, make_policy

# Regression tests from the bypass-hunting pass. Each documents a specific
# escape shape the verifier must reject, and why a narrower fix could miss it.


def test_alias(verify_all):
    # Aliasing a banned builtin to a local name must not let it evade the call-site ban.
    src = ["e = eval", "e('1/0')"]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_alias_in_container(verify_all):
    # A list/dict literal must not be usable to stash a banned callable for later use.
    src = ["con = [eval]", "con[0]('1/0')"]
    assert "banned-construct" in error_codes(verify_all("\n".join(src)))


def test_self_stash_and_call(verify_all):
    # self.<name> is not a blanket exemption: a callable stashed there in setup and invoked
    # later in __call__ must still be checked, not waved through just because of the prefix.
    src = [
        "class M(object):",
        "    def setup(self):",
        "        self.open = open",
        "    def __call__(self, x):",
        "        self.open('/etc/passwd')",
        "",
        "m = M()",
    ]
    assert "attr-on-value" in error_codes(verify_all("\n".join(src)))


def test_aliased_import_getattr(verify_all):
    # Aliasing __import__ and getattr must not enable a dynamic import + invoke chain.
    src = ["i = __import__", "g = getattr", "g(i('os'), 'getcwd')()"]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_aliased_getattr_on_builtins(verify_all):
    # getattr, even via alias, must not be usable to reach a banned builtin off __builtins__.
    src = ["g = getattr", "ev = g(__builtins__, 'eval')", "ev('1/0')"]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_unicode_homoglyphs_should_not_be_allowed(verify_all):
    # Stashing the real builtin then calling through a visually-identical homoglyph name
    # (Cyrillic о р е — renders like "open" in many fonts) must not evade the ban (#9).
    src = ["ореn = open", "ореn('/etc/passwd')"]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_print_should_not_be_allowed(verify_all):
    # print is a stdout exfiltration channel (disallowed-ast-examples.md) and must be banned
    # the same way open/input are.
    src = ["print('hello')"]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_class_creation_with_type(verify_all):
    # The 3-argument type() form must not bypass ClassDef base/decorator restrictions (#1).
    src = ["M = type('M', (BannedBase,), {})"]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_class_creation_with_build_class(verify_all):
    # __build_class__ must not bypass ClassDef restrictions any more than type() may (#2).
    src = ["M = __build_class__(lambda self, x: x, 'M', (BannedBase,), {})"]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_bare_name_import_bypass(policy):
    # A bare name imported via `from X import f` in public must resolve through the same
    # allowlist/denylist a dotted call (`os.system(...)`) gets, not just BANNED_NAMES.
    src = [
        "from os import system",
        "",
        "def run():",
        "    system('id')",
    ]
    result = verify("\n".join(src), [[3, 4]], policy)
    assert "call-not-allowed" in error_codes(result)


@pytest.mark.parametrize("fname", ["repr", "str", "ascii", "format", "bytes"])
def test_dunder_proxy_builtin_should_not_be_allowed(fname, verify_all):
    # repr()/str()/ascii()/format() invoke the same dunder methods "named method on a value"
    # already bans via attribute syntax (x.__repr__()); calling them bare must be banned too (#10).
    src = ["def f(x):", f"    return {fname}(x)"]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_fstring_conversion_flag_should_not_be_allowed(verify_all):
    # f"{x!r}" invokes x.__repr__() via FormattedValue's conversion flag, with no Call node —
    # must be rejected the same as repr(x) (#10).
    src = ["def f(x):", "    return f'{x!r}'"]
    assert "method-on-value" in error_codes(verify_all("\n".join(src)))


def test_bare_class_dunder_name_should_not_be_allowed(verify_all):
    # __class__ is an implicit bare Name in every method body; the dunder ban must cover bare
    # Name reads, not just Attribute.attr (#11). Flagged as its own "dunder-name" code, distinct
    # from "dunder-attr", since this isn't an attribute access.
    src = [
        "class M(object):",
        "    def __call__(self, x):",
        "        c = __class__",
        "        return x",
    ]
    assert "dunder-name" in error_codes(verify_all("\n".join(src)))


def test_container_subscript_store_alias(verify_all):
    # Subscript assignment (d["k"] = open) must not be a way to smuggle a banned reference into
    # a container slot just because the container name itself is Load, not Store, context.
    src = ["d = {}", "d['k'] = open", "d['k']('/etc/passwd')"]
    assert "banned-construct" in error_codes(verify_all("\n".join(src)))


def test_chained_assignment_alias(verify_all):
    # Chained assignment (a = b = open) must alias a banned callable to every target, not just
    # whichever one a shape-specific check happens to look at.
    src = ["a = b = open", "b('/etc/passwd')"]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_tuple_unpack_alias(verify_all):
    # Tuple unpacking must alias a banned callable to its target the same way plain assignment
    # does.
    src = ["a, b = (1, open)", "b('/etc/passwd')"]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_return_value_alias_with_no_local_name(verify_all):
    # A banned reference smuggled out via `return` and invoked immediately (leak()(...)) must be
    # caught even with no alias variable anywhere near the dangerous call — the stealthiest
    # variant of the aliasing family.
    src = [
        "def leak():",
        "    return open",
        "",
        "def run(x):",
        "    return leak()('/etc/passwd')",
    ]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_parameter_passthrough_alias(verify_all):
    # A banned callable passed positionally into a generic helper must be caught even though no
    # suspicious name ever appears in the helper's own body.
    src = [
        "def apply(fn, x):",
        "    return fn(x)",
        "",
        "def run():",
        "    return apply(open, '/etc/passwd')",
    ]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_self_nested_attribute_chain(verify_all):
    # Only a single self.<name> level is exempt from call-target policy — self.<a>.<b>(...) must
    # still be checked like any attribute chain on an opaque value.
    src = [
        "class M(object):",
        "    def setup(self):",
        "        self.sub = object()",
        "        self.sub.evil = open",
        "    def __call__(self, x):",
        "        self.sub.evil('/etc/passwd')",
        "",
        "m = M()",
    ]
    assert "attr-on-value" in error_codes(verify_all("\n".join(src)))


def test_self_augassign_alias(verify_all):
    # AugAssign to a self attribute (self.evil += open) must be tracked the same way a plain
    # assignment to it is.
    src = [
        "class M(object):",
        "    def setup(self):",
        "        self.evil = print",
        "    def __call__(self, x):",
        "        self.evil += open",
        "        self.evil('/etc/passwd')",
        "",
        "m = M()",
    ]
    assert "attr-on-value" in error_codes(verify_all("\n".join(src)))


def test_dict_literal_container_alias(verify_all):
    # Inline dict literal dispatch (d = {"o": open}) must be closed the same way list[0]
    # dispatch is (#3).
    src = ['d = {"o": open}', 'd["o"]("/etc/passwd")']
    assert "banned-construct" in error_codes(verify_all("\n".join(src)))


def test_inline_container_subscript_call(verify_all):
    # Subscript-then-call on a bare literal, with no assignment step at all, must still be
    # caught (#12).
    src = ["([eval][0])('1/0')"]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_ifexp_branch_alias(verify_all):
    # A banned reference selected via either branch of an IfExp must be tracked (#12).
    src = ["c = True", "fn = open if c else eval", "fn('/etc/passwd')"]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_for_loop_container_alias(verify_all):
    # A for-loop/comprehension target bound from a literal container of banned callables must
    # be tracked (#12).
    src = ["for fn in [eval]:", "    fn('1/0')"]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_function_default_alias(verify_all):
    # A banned callable smuggled in via a parameter default must be tracked (#12).
    src = [
        "def run(op=open):",
        "    return op('/etc/passwd')",
        "",
        "run()",
    ]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_bare_name_disallowed_leaf_bypass():
    # A disallowed API reached via a bare public import must still be rejected (#6b).
    policy = make_policy(disallow=["jax.numpy.save"])
    src = [
        "from jax.numpy import save",
        "",
        "def run(x):",
        "    save(x, 'stolen_data.zip')",
    ]
    result = verify("\n".join(src), [[3, 4]], policy)
    assert "call-not-allowed" in error_codes(result)


def test_bare_name_import_alias_bypass():
    # Renaming a public import (`as persist`) must not let the resulting bare call bypass a
    # disallow entry that names the underlying leaf (#6).
    policy = make_policy(disallow=["jax.numpy.save"])
    src = [
        "from jax.numpy import save as persist",
        "",
        "def run(x):",
        "    persist(x, 'stolen_data.zip')",
    ]
    result = verify("\n".join(src), [[3, 4]], policy)
    assert "call-not-allowed" in error_codes(result)


def test_importlib_import_module_bypass(policy):
    # A dynamic loader (importlib.import_module) reached via one public import line must still
    # be denied (#7b).
    src = [
        "from importlib import import_module",
        "",
        "def run():",
        "    import_module('os')",
    ]
    result = verify("\n".join(src), [[3, 4]], policy)
    assert "call-not-allowed" in error_codes(result)


def test_aliased_import_getattr_system(verify_all):
    # A full import+invoke chain built entirely in the private region, with no public glue at
    # all, must be caught (#7).
    src = [
        "i = __import__",
        "g = getattr",
        "g(g(i('os'), 'system'))('id')",
    ]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_vars_alias_on_builtins(verify_all):
    # vars, aliased, must not reach banned builtins off __builtins__ any more than getattr
    # can (#5).
    src = ["v = vars", "ev = v(__builtins__)['eval']", "ev('1/0')"]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_self_dynamic_import_chain(verify_all):
    # The idiomatic Flax setup/__call__ shape must not be a shield for a full dynamic-import
    # chain (#8 + #7).
    src = [
        "class M(object):",
        "    def setup(self):",
        "        self.imp = __import__",
        "        self.get = getattr",
        "    def __call__(self, x):",
        "        os = self.imp('os')",
        "        self.get(os, 'system')('id')",
    ]
    assert "attr-on-value" in error_codes(verify_all("\n".join(src)))


def test_self_layer_subscript_call(verify_all):
    # The Flax self.layer[i](x) idiom must still be vetted — a tainted layer list must not slip
    # through just because it matches a recognized pattern shape (#8).
    src = [
        "class M(object):",
        "    def setup(self):",
        "        self.layer = [open]",
        "    def __call__(self, x):",
        "        self.layer[0]('/etc/passwd')",
    ]
    assert "attr-on-value" in error_codes(verify_all("\n".join(src)))


def test_homoglyph_self_stash_and_call(verify_all):
    # A homoglyph attribute name stashed on self must not evade BANNED_NAMES at the call site
    # (#9 + #8).
    src = [
        "class M(object):",
        "    def setup(self):",
        "        self.ореn = open",
        "    def __call__(self, x):",
        "        self.ореn('/etc/passwd')",
    ]
    assert "attr-on-value" in error_codes(verify_all("\n".join(src)))


def test_homoglyph_two_hop_alias(verify_all):
    # An ASCII stash followed by a homoglyph copy — no banned name textually at the call site —
    # must still be caught (#9).
    src = ["x = open", "ореn = x", "ореn('/etc/passwd')"]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_dunder_proxy_format_should_not_be_allowed(verify_all):
    # format() invokes x.__format__() without an Attribute node; calling it bare must be
    # banned too (#10).
    src = ["def f(x):", "    return format(x, 'd')"]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_bytes_buffer_exfiltration_should_not_be_allowed(verify_all):
    # bytes(x) serializes an array's entire raw memory buffer losslessly in one call (verified:
    # round-trips exactly via np.frombuffer) — a complete data dump, worse than repr/str since
    # there's no formatting/truncation, and with no legitimate use on a JAX array in inference
    # code — must be banned (#10).
    src = ["def f(x):", "    return bytes(x)"]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_fstring_conversion_s_flag_should_not_be_allowed(verify_all):
    # f"{x!s}" invokes x.__str__() with no Call node — must be rejected the same as str(x) (#10).
    src = ["def f(x):", "    return f'{x!s}'"]
    assert "method-on-value" in error_codes(verify_all("\n".join(src)))


def test_fstring_debug_specifier_should_not_be_allowed(verify_all):
    # f"{x=}" implicitly invokes repr(x) with no Call node — must be rejected too (#10).
    src = ["def f(x):", "    return f'{x=}'"]
    assert "method-on-value" in error_codes(verify_all("\n".join(src)))


def test_nested_generic_annotation_is_not_flagged(verify_all):
    # A banned-name identifier nested inside a subscripted/generic annotation (list[str],
    # dict[str, bytes]) must not be flagged -- the whole annotation subtree is exempt, not
    # just a bare top-level Name annotation like `x: str`.
    src = [
        "def f(x: list[str]) -> dict[str, bytes]:",
        "    return {}",
    ]
    result = verify_all("\n".join(src))
    assert result.ok, [(v.code, v.message) for v in result.violations]


def test_future_annotations_import_does_not_change_annotation_handling(policy):
    # `from __future__ import annotations` only changes runtime evaluation of annotations, not
    # how they parse -- ast.parse produces the identical Name/Subscript tree either way, so a
    # str-annotated function in the private region must still verify cleanly.
    src = [
        "from __future__ import annotations",
        "",
        "def f(x: str) -> list[bytes]:",
        "    return []",
    ]
    result = verify("\n".join(src), [[3, 4]], policy)
    assert result.ok, [(v.code, v.message) for v in result.violations]


def test_stringized_annotation_is_not_flagged(verify_all):
    # A quoted forward-reference annotation (`x: "str"`) is a Constant, not a Name -- it was
    # never reachable by the BANNED_NAMES Name check in the first place, but is worth locking
    # in as a passing case since it's a common style for forward references.
    src = ['def f(x: "str") -> "bytes":', "    return x"]
    result = verify_all("\n".join(src))
    assert result.ok, [(v.code, v.message) for v in result.violations]


# ── self/cls trust is a lexical string match, not a verified binding -- it must not be ──
# ── forgeable by reassigning "self"/"cls" or reusing them as an unrelated parameter name ──


def test_self_reassignment_must_not_grant_trust(verify_all):
    # Rebinding "self" to an arbitrary object must not let the self.<name> exemption apply to
    # that object instead of the real instance.
    src = [
        "class M(object):",
        "    def __call__(self, x):",
        "        self = x",
        "        return self.anything_at_all()",
    ]
    assert "reserved-name" in error_codes(verify_all("\n".join(src)))


def test_cls_as_local_variable_must_not_grant_trust(verify_all):
    # "cls" used as an ordinary local variable name (not a real classmethod parameter, which
    # can't occur anyway since @classmethod isn't allow-listed) must not grant the self.<name>
    # exemption to whatever it happens to be assigned.
    src = [
        "class M(object):",
        "    def __call__(self, x):",
        "        cls = x",
        "        return cls.anything_at_all()",
    ]
    assert "reserved-name" in error_codes(verify_all("\n".join(src)))


def test_nested_function_self_parameter_must_not_grant_trust(policy):
    # The full realistic chain: a private nn.Module stashes a non-builtin dangerous import (not
    # in BANNED_NAMES) via setup(), and a nested helper function's parameter is coincidentally
    # named "self" -- at runtime this actually calls os.system through the real instance's
    # attribute, entirely outside the enclosing class M's own self-attribute safety table. Must
    # be rejected regardless of what "self" is coincidentally named inside the nested function.
    src = [
        "from os import system",
        "from flax import linen as nn",
        "",
        "class Wrapper(nn.Module):",
        "    def setup(self):",
        "        self.run = system",
        "",
        "class M(nn.Module):",
        "    def __call__(self, x):",
        "        w = Wrapper()",
        "        def helper(self):",
        "            return self.run('id')",
        "        return helper(w)",
    ]
    result = verify("\n".join(src), [[4, 13]], policy)
    assert "reserved-name" in error_codes(result)


def test_self_only_trusted_as_first_param_of_a_direct_method(verify_all):
    # A parameter literally named "self" is only trustworthy as the genuine first parameter of
    # a method defined directly in a class body -- not a non-first parameter, and not a
    # lambda's parameter.
    src = [
        "class M(object):",
        "    def __call__(self, x):",
        "        def helper(x, self):",
        "            return self.evil()",
        "        return helper(1, 2)",
    ]
    assert "reserved-name" in error_codes(verify_all("\n".join(src)))

    lambda_src = [
        "class M(object):",
        "    def __call__(self, x):",
        "        f = lambda self: self.evil()",
        "        return f(x)",
    ]
    assert "reserved-name" in error_codes(verify_all("\n".join(lambda_src)))


def test_comprehension_target_rebinding_is_checked(verify_all):
    # ast.comprehension carries no lineno/col_offset (verified directly), so dispatching the
    # reserved-name check from that node type was dead code -- it never actually ran, for
    # self/cls rebinding or for the original import-alias-rebinding case. Must be checked from
    # the enclosing comprehension expression instead.
    src = ["y = [cls for cls in [1, 2]]"]
    assert "reserved-name" in error_codes(verify_all("\n".join(src)))


def test_self_dunder_call_should_not_be_allowed(verify_all):
    # _check_call_attribute grants self/cls the same "inherited, presumed safe" trust as a plain
    # self.<name> attribute read, but unlike _check_attribute it never checks is_dunder(attr)
    # first. self.__getattribute__('__dict__') reaches the *call* path, not the read path, so it
    # skips the dunder-attr ban entirely and reads back the whole instance __dict__.
    src = [
        "class M(object):",
        "    def __call__(self, x):",
        "        d = self.__getattribute__('__dict__')",
        "        return x",
    ]
    assert "dunder-attr" in error_codes(verify_all("\n".join(src)))


def test_fstring_plain_interpolation_should_not_be_allowed(verify_all):
    # A plain f"{x}" (conversion -1, no !r/!s/!a and no {x=} debug form) still invokes
    # type(x).__format__(x, '') on the value -- verified directly that a custom __format__
    # override fires for a bare f'{x}' -- so it is the same dunder-invocation-with-no-Call-node
    # escape as f"{x!r}"/f"{x!s}"/f"{x=}", not a safe case as the current code assumes.
    src = ["def f(x):", "    return f'{x}'"]
    assert "method-on-value" in error_codes(verify_all("\n".join(src)))


def test_class_name_shadowing_reserved_import_alias(verify_all):
    # A class statement's own name is a Store-context binding just like an assignment target --
    # shadowing a trusted import alias (jnp) with a local class must be caught the same way
    # `jnp = evil` already is, since LEGB scoping makes later `jnp.einsum(...)` calls resolve to
    # the shadowing class, not the real jax.numpy module.
    src = [
        "import jax.numpy as jnp",
        "",
        "def helper(x, secret_sink):",
        "    class jnp:",
        "        einsum = secret_sink",
        '    return jnp.einsum("ij->i", x)',
    ]
    assert "reserved-name" in error_codes(verify_all("\n".join(src)))


def test_def_name_shadowing_public_wrapper(policy):
    # Same as the class case, but for a def statement shadowing a public-region wrapper
    # name instead of an import alias. `transpose` must be genuinely public (outside the private
    # range) for it to land in scan.public_defs at all.
    src = [
        "def transpose(x):",
        "    return x",
        "",
        "def helper(x, secret_sink):",
        "    def transpose(y):",
        "        return secret_sink(y)",
        "    return transpose(x)",
    ]
    result = verify("\n".join(src), [[4, 7]], policy)
    assert "reserved-name" in error_codes(result)


def test_self_attr_tuple_unpack_must_not_grant_trust(verify_all):
    # self.fn, self.other = fn, 0 must be vetted the same way the equivalent single-target
    # `self.fn = fn` already is -- tuple-unpacking must not be a way to hide an unvetted callable
    # from _SelfAttrTrust's assignment-table builder.
    src = [
        "class M(object):",
        "    def setup(self):",
        "        pass",
        "    def __call__(self, x, fn):",
        "        self.fn, self.other = fn, 0",
        "        return self.fn(x)",
    ]
    assert "attr-on-value" in error_codes(verify_all("\n".join(src)))


def test_self_attr_for_loop_target_must_not_grant_trust(verify_all):
    # Binding self.<attr> via a for-loop target must be vetted the same way a plain assignment to
    # it is -- otherwise the attribute is never recorded as assigned and defaults to "presumed
    # inherited/safe", letting an arbitrary caller-supplied callable be invoked through it.
    src = [
        "class Model(object):",
        "    def setup(self, cb):",
        "        for self.leak in (cb,):",
        "            pass",
        "    def __call__(self, x, secret):",
        "        self.leak(secret)",
        "        return x",
    ]
    assert "attr-on-value" in error_codes(verify_all("\n".join(src)))


def test_self_attr_local_alias_must_not_grant_trust(verify_all):
    # Copying self.<attr> into a local variable before calling it must not defeat _SelfAttrTrust --
    # reading self.<attr> is unconditionally allowed, and calling a local variable is
    # unconditionally allowed, so without alias tracking this trivially bypasses the same check
    # that correctly rejects calling self.fn directly.
    src = [
        "class M(object):",
        "    def setup(self):",
        "        self.fn = lambda x: x + 1",
        "    def __call__(self, x):",
        "        tmp = self.fn",
        "        return tmp(x)",
    ]
    assert "attr-on-value" in error_codes(verify_all("\n".join(src)))


def test_self_attr_subscript_local_alias_must_not_grant_trust(verify_all):
    # Same bypass via the Flax self.layer[i](x) idiom: copy the element to a local first. Uses an
    # opaque parameter (not a BANNED_NAMES literal like `open`) so the only thing standing between
    # this and a violation is the self-attr trust check the alias is meant to defeat.
    src = [
        "class M(object):",
        "    def setup(self, cb):",
        "        self.layer = [cb]",
        "    def __call__(self, x):",
        "        tmp = self.layer[0]",
        "        return tmp(x)",
    ]
    assert "attr-on-value" in error_codes(verify_all("\n".join(src)))


def test_self_attr_two_hop_local_alias_must_not_grant_trust(verify_all):
    # A second-hop copy (tmp2 = tmp) must not escape alias tracking either, mirroring the
    # multi-hop rigor already applied to BANNED_NAMES aliasing (test_homoglyph_two_hop_alias).
    src = [
        "class M(object):",
        "    def setup(self):",
        "        self.fn = lambda x: x + 1",
        "    def __call__(self, x):",
        "        tmp = self.fn",
        "        tmp2 = tmp",
        "        return tmp2(x)",
    ]
    assert "attr-on-value" in error_codes(verify_all("\n".join(src)))


def test_self_attr_reassigned_local_alias_is_not_stuck_tainted(verify_all):
    # Reassigning a variable that once aliased an unsafe self.<attr> to something else entirely
    # must clear the taint -- tracking must reflect the CURRENT value, not the first one ever seen.
    src = [
        "class M:",
        "    def setup(self):",
        "        self.fn = lambda x: x + 1",
        "    def __call__(self, x):",
        "        tmp = self.fn",
        "        tmp = x",
        "        return tmp",
    ]
    result = verify_all("\n".join(src))
    assert result.ok, [(v.code, v.message) for v in result.violations]


def test_self_attr_local_alias_of_safe_source_is_still_allowed(verify_all):
    # A local alias of a self.<attr> that WAS assigned a vetted-safe source (or is presumed
    # inherited, since the class never assigns it) must still be callable -- alias tracking must
    # track safety, not blanket-ban aliasing self.<attr> outright (see
    # test_calling_a_value_is_allowed in test_whitelist.py, which already covers this shape and
    # must keep passing).
    src = [
        "class M:",
        "    def __call__(self, x):",
        "        layer = self.layers[0]",
        "        return layer(x)",
    ]
    result = verify_all("\n".join(src))
    assert result.ok, [(v.code, v.message) for v in result.violations]


# ── call-target default-deny: a bare-name/value call must be *provably* safe, not merely  ──
# ── un-flagged (docs/verify.md#the-full-call-target-rule) ──────────────────────────────────
# Previously ANY bare-name call was allowed unless the callee's origin happened to be a hardcoded
# BANNED_NAMES reference (see test_parameter_passthrough_alias above, which only catches the `open`
# argument -- not the `fn(x)` call itself). These tests lock in the stricter model: a call target
# must resolve to an allow-listed import, a def/class in this file, a safe builtin, or a local/value
# traced to one of those -- everything else is rejected outright, even with nothing "banned" in sight.


def test_call_through_bare_parameter_is_rejected(verify_all):
    # The exact gap this model closes: a parameter is never traceable to a safe source, so calling
    # it directly must be rejected even though no banned name appears anywhere in this snippet.
    src = ["def apply(fn, x):", "    return fn(x)"]
    assert "call-unresolved" in error_codes(verify_all("\n".join(src)))


def test_call_through_unresolvable_name_is_rejected(verify_all):
    # A bare name that isn't an import, a def/class in this file, a safe builtin, or a tracked-safe
    # local has no provable provenance at all -- reject it, don't wave it through by default.
    src = ["def f(x):", "    return g(x)"]
    assert "call-unresolved" in error_codes(verify_all("\n".join(src)))


def test_local_bound_to_private_constructor_is_still_callable(verify_all):
    # The "layer" idiom must keep working for a PLAIN local, not just a self.<attr> alias: a
    # variable bound to a call to a class/def defined in this file is provably safe.
    src = [
        "class Attn:",
        "    def __call__(self, x):",
        "        return x",
        "",
        "def helper(x):",
        "    block = Attn()",
        "    return block(x)",
    ]
    result = verify_all("\n".join(src))
    assert result.ok, [(v.code, v.message) for v in result.violations]


def test_safe_builtin_call_is_allowed(verify_all):
    src = ["def helper(n):", "    return list(range(n))"]
    result = verify_all("\n".join(src))
    assert result.ok, [(v.code, v.message) for v in result.violations]


def test_safe_builtin_name_cannot_be_rebound(verify_all):
    # _check_call trusts a bare call to `list`/`range`/etc. by identifier alone (like a trusted
    # import alias or public wrapper) -- shadowing one would silently redirect every call site that
    # appears to route through it.
    src = ["def helper():", "    list = None", "    return list"]
    assert "reserved-name" in error_codes(verify_all("\n".join(src)))


def test_chained_call_through_private_constructor_is_allowed(verify_all):
    # Block()(x): the callee is itself a Call to a class/def defined in this file -- unambiguous by
    # construction, no local-variable indirection needed.
    src = [
        "class Block:",
        "    def __call__(self, x):",
        "        return x",
        "",
        "def helper(x):",
        "    return Block()(x)",
    ]
    result = verify_all("\n".join(src))
    assert result.ok, [(v.code, v.message) for v in result.violations]


def test_chained_call_through_unresolved_value_is_rejected(verify_all):
    # d['k'](x): the callee is a subscript on an arbitrary parameter -- not self-rooted, not traced
    # to any safe source. Must be rejected, not waved through as "calling a value".
    src = ["def helper(d, x):", "    return d['k'](x)"]
    assert "call-unresolved" in error_codes(verify_all("\n".join(src)))
