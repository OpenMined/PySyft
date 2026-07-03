import pytest
from syft_restrict import verify

from .conftest import error_codes

# The following tests fail because the current implementation of the verifier
# does not detect these cases.


def test_alias(verify_all):
    # failing: a banned callable can be assigned to a local var and called
    src = ["e = eval", "e('1/0')"]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_alias_in_container(verify_all):
    # failing: list or dict literals can store banned callables for later use
    src = ["con = [eval]", "con[0]('1/0')"]
    assert "banned-construct" in error_codes(verify_all("\n".join(src)))


def test_self_stash_and_call(verify_all):
    # failing: a class can stash a banned callable in an attribute and call it
    # later, since the self prefix whitelists the attribute access
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
    # failing: a banned callable can be imported and called via an alias
    src = ["i = __import__", "g = getattr", "g(i('os'), 'getcwd')()"]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_aliased_getattr_on_builtins(verify_all):
    # failing: a banned builtin callable can be accessed via getattr on
    # __builtins__
    src = ["g = getattr", "ev = g(__builtins__, 'eval')", "ev('1/0')"]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_unicode_homoglyphs_should_not_be_allowed(verify_all):
    # failing (#9): stash the real builtin, call through a homoglyph name
    # (Cyrillic о р е — renders like "open" in many fonts)
    src = ["ореn = open", "ореn('/etc/passwd')"]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_print_should_not_allowed(verify_all):
    # failing: print is listed on disallowed-ast-examples.md, but it's not
    # banned.
    src = ["print('hello')"]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_class_creation_with_type(verify_all):
    # failing (#1): dynamic class creation via type() bypasses ClassDef checks
    src = ["M = type('M', (BannedBase,), {})"]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_class_creation_with_build_class(verify_all):
    # failing: a class with a disallowed base class can be created using the
    # __build_class__ builtin
    src = ["M = __build_class__(lambda self, x: x, 'M', (BannedBase,), {})"]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_bare_name_import_bypass(policy):
    # failing: `from X import f` in public + a bare `f(...)` call in private
    # skips both the JAX denylist and the function allowlist that the dotted
    # equivalent (`os.system(...)`) is subject to
    src = [
        "from os import system",
        "",
        "def run():",
        "    system('id')",
    ]
    result = verify("\n".join(src), [[3, 4]], policy)
    assert "call-not-allowed" in error_codes(result)


def test_dunder_proxy_builtin_should_not_be_allowed(verify_all):
    # failing: repr()/format()/hash()/etc. invoke the same dunder methods
    # that "named method on a value" already bans via attribute syntax
    # (x.__repr__()), but as bare-name calls they are never checked at all
    src = ["def f(x):", "    return repr(x)"]
    assert "method-on-value" in error_codes(verify_all("\n".join(src)))


def test_fstring_conversion_flag_should_not_be_allowed(verify_all):
    # failing: f"{x!r}" invokes x.__repr__() via FormattedValue's conversion
    # flag, with no Call node at all for any existing check to run against
    src = ["def f(x):", "    return f'{x!r}'"]
    assert "method-on-value" in error_codes(verify_all("\n".join(src)))


def test_bare_class_dunder_name_should_not_be_allowed(verify_all):
    # failing: __class__ is an implicit bare Name in every method body; the
    # dunder ban only inspects Attribute.attr, never a bare Name.id
    src = [
        "class M(object):",
        "    def __call__(self, x):",
        "        c = __class__",
        "        return x",
    ]
    assert "dunder-attr" in error_codes(verify_all("\n".join(src)))


def test_container_subscript_store_alias(verify_all):
    # failing: d["k"] = open stores a banned callable via subscript
    # assignment; the container name is Load context in a subscript store,
    # so even the existing reserved-name check never inspects this target,
    # and the RHS is never checked at all
    src = ["d = {}", "d['k'] = open", "d['k']('/etc/passwd')"]
    assert "banned-construct" in error_codes(verify_all("\n".join(src)))


def test_chained_assignment_alias(verify_all):
    # failing: chained assignment (a = b = open) aliases a banned callable
    # to a second target that assignment-target checking never revisits
    src = ["a = b = open", "b('/etc/passwd')"]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_tuple_unpack_alias(verify_all):
    # failing: tuple unpacking aliases a banned callable the same way plain
    # assignment does, but isn't covered by the same shape-specific checks
    src = ["a, b = (1, open)", "b('/etc/passwd')"]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_return_value_alias_with_no_local_name(verify_all):
    # failing: a banned reference can be smuggled out via `return` and
    # invoked immediately (leak()(...)) with no alias variable anywhere
    # near the call -- the stealthiest variant of the aliasing family
    src = [
        "def leak():",
        "    return open",
        "",
        "def run(x):",
        "    return leak()('/etc/passwd')",
    ]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_parameter_passthrough_alias(verify_all):
    # failing: a banned callable passed positionally into a generic helper
    # is invoked there with no suspicious name anywhere in the helper body
    src = [
        "def apply(fn, x):",
        "    return fn(x)",
        "",
        "def run():",
        "    return apply(open, '/etc/passwd')",
    ]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_self_nested_attribute_chain(verify_all):
    # failing: self.<a>.<b>(...) skips call-target policy at any attribute
    # depth, not just the single-level self.<name>(...) case
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
    # failing: AugAssign to a self attribute is never checked either --
    # _iter_names finds no Store-context Name inside an Attribute target
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
    # failing (#3): inline dict literal dispatch — same gap as list[0]
    src = ['d = {"o": open}', 'd["o"]("/etc/passwd")']
    assert "banned-construct" in error_codes(verify_all("\n".join(src)))


def test_inline_container_subscript_call(verify_all):
    # failing (#12): no assignment at all — subscript-then-call on a literal
    src = ["([eval][0])('1/0')"]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_ifexp_branch_alias(verify_all):
    # failing (#12): banned reference selected via IfExp branch
    src = ["c = True", "fn = open if c else eval", "fn('/etc/passwd')"]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_for_loop_container_alias(verify_all):
    # failing (#12): comprehension/for target binds a banned callable from a container
    src = ["for fn in [eval]:", "    fn('1/0')"]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_function_default_alias(verify_all):
    # failing (#12): banned callable smuggled via a parameter default
    src = [
        "def run(op=open):",
        "    return op('/etc/passwd')",
        "",
        "run()",
    ]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_bare_name_jax_denylist_bypass(policy):
    # failing (#6b): denylisted JAX API reachable via bare public import
    src = [
        "from jax.numpy import save",
        "",
        "def run(x):",
        "    save(x, 'stolen_data.zip')",
    ]
    result = verify("\n".join(src), [[3, 4]], policy)
    assert "call-not-allowed" in error_codes(result)


def test_bare_name_import_alias_bypass(policy):
    # failing (#6): renamed public import still bypasses policy on bare call
    src = [
        "from jax.numpy import save as persist",
        "",
        "def run(x):",
        "    persist(x, 'stolen_data.zip')",
    ]
    result = verify("\n".join(src), [[3, 4]], policy)
    assert "call-not-allowed" in error_codes(result)


def test_importlib_import_module_bypass(policy):
    # failing (#7b): dynamic loader via one public import line
    src = [
        "from importlib import import_module",
        "",
        "def run():",
        "    import_module('os')",
    ]
    result = verify("\n".join(src), [[3, 4]], policy)
    assert "call-not-allowed" in error_codes(result)


def test_aliased_import_getattr_system(verify_all):
    # failing (#7): full private-region import+invoke chain (no public glue)
    src = [
        "i = __import__",
        "g = getattr",
        "g(g(i('os'), 'system'))('id')",
    ]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_vars_alias_on_builtins(verify_all):
    # failing (#5): vars alias reaches banned builtins the same way getattr does
    src = ["v = vars", "ev = v(__builtins__)['eval']", "ev('1/0')"]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_self_dynamic_import_chain(verify_all):
    # failing (#8 + #7): idiomatic Flax setup/__call__ with full dynamic import
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
    # failing (#8): Flax-shaped self.layer[i](x) with a tainted layer list
    src = [
        "class M(object):",
        "    def setup(self):",
        "        self.layer = [open]",
        "    def __call__(self, x):",
        "        self.layer[0]('/etc/passwd')",
    ]
    assert "attr-on-value" in error_codes(verify_all("\n".join(src)))


def test_homoglyph_self_stash_and_call(verify_all):
    # failing (#9 + #8): homoglyph attribute on self evades BANNED_NAMES at call site
    src = [
        "class M(object):",
        "    def setup(self):",
        "        self.ореn = open",
        "    def __call__(self, x):",
        "        self.ореn('/etc/passwd')",
    ]
    assert "attr-on-value" in error_codes(verify_all("\n".join(src)))


def test_homoglyph_two_hop_alias(verify_all):
    # failing (#9): ASCII stash then homoglyph copy — no banned name at call site
    src = ["x = open", "ореn = x", "ореn('/etc/passwd')"]
    assert "banned-call" in error_codes(verify_all("\n".join(src)))


def test_dunder_proxy_format_should_not_be_allowed(verify_all):
    # failing (#10): format() invokes x.__format__() without an Attribute node
    src = ["def f(x):", "    return format(x, 'd')"]
    assert "method-on-value" in error_codes(verify_all("\n".join(src)))


def test_dunder_proxy_hash_should_not_be_allowed(verify_all):
    # failing (#10): hash() invokes x.__hash__() without an Attribute node
    src = ["def f(x):", "    return hash(x)"]
    assert "method-on-value" in error_codes(verify_all("\n".join(src)))


def test_fstring_conversion_s_flag_should_not_be_allowed(verify_all):
    # failing (#10): f"{x!s}" invokes x.__str__() with no Call node
    src = ["def f(x):", "    return f'{x!s}'"]
    assert "method-on-value" in error_codes(verify_all("\n".join(src)))


def test_fstring_debug_specifier_should_not_be_allowed(verify_all):
    # failing (#10): f"{x=}" invokes repr(x) with no Call node
    src = ["def f(x):", "    return f'{x=}'"]
    assert "method-on-value" in error_codes(verify_all("\n".join(src)))
