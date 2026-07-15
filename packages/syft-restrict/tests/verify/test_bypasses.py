"""Regression tests for known escape / edge shapes.

Each documents a specific attack the verifier must reject, and why a narrower
fix could miss it.
"""

from verify.helpers import get_error_codes, make_policy


def _assert_error_code(verify_all, src, expected, private=None, pol=None, strict=False):
    err = get_error_codes(verify_all(src, pol=pol, private=private))

    assert err
    # if strict, we fail cases with extra violations
    if strict:
        assert {expected} == err
    else:
        assert expected in err


# ── banned-name alias surfaces ───────────────────────────────────────────────


def test_private_alias(verify_all):
    # Aliasing a banned name to a private name must not let it evade the ban.
    _assert_error_code(verify_all, "e = eval", "banned-name")


def test_public_alias(verify_all):
    # Stash in public, call via untracked name in private — call-unresolved, not
    # a free pass because the private call site has no banned identifier.
    src = """
    e = eval
    e('1/0')
    """
    _assert_error_code(verify_all, src, "call-unresolved", private=[[2, 2]])


def test_alias_in_container(verify_all):
    _assert_error_code(verify_all, "con = [eval]", "banned-name")


def test_container_subscript_store_alias(verify_all):
    # Subscript assignment (d["k"] = open) must not smuggle a banned reference
    # just because the container name is Load, not Store.
    src = """
    d = {}
    d['k'] = open
    """
    _assert_error_code(verify_all, src, "banned-name")


def test_chained_assignment_alias(verify_all):
    src = """
    a = b = open
    b('/etc/passwd')
    """
    _assert_error_code(verify_all, src, "banned-name")


def test_tuple_unpack_alias(verify_all):
    src = """
    a, b = (1, open)
    b('/etc/passwd')
    """
    _assert_error_code(verify_all, src, "banned-name")


def test_return_value_alias_with_no_local_name(verify_all):
    # Banned reference smuggled via `return` and invoked immediately — no alias
    # variable near the dangerous call.
    src = """
    def leak():
        return open

    def run(x):
        return leak()('/etc/passwd')
    """
    _assert_error_code(verify_all, src, "banned-name")


def test_parameter_passthrough_alias(verify_all):
    # Banned callable passed into a generic helper — no suspicious name in the
    # helper body itself (the call of `fn` is also call-unresolved).
    src = """
    def apply(fn, x):
        return fn(x)

    def run():
        return apply(open, '/etc/passwd')
    """
    _assert_error_code(verify_all, src, "banned-name")


def test_dict_literal_container_alias(verify_all):
    src = """
    d = {"o": open}
    d["o"]("/etc/passwd")
    """
    _assert_error_code(verify_all, src, "banned-name")


def test_inline_container_subscript_call(verify_all):
    # Subscript-then-call on a bare literal, no assignment step.
    _assert_error_code(verify_all, "([eval][0])('1/0')", "banned-name")


def test_ifexp_branch_alias(verify_all):
    src = """
    c = True
    fn = open if c else eval
    fn('/etc/passwd')
    """
    _assert_error_code(verify_all, src, "banned-name")


def test_for_loop_container_alias(verify_all):
    src = """
    for fn in [eval]:
        fn('1/0')
    """
    _assert_error_code(verify_all, src, "banned-name")


def test_function_default_alias(verify_all):
    src = """
    def run(op=open):
        return op('/etc/passwd')

    run()
    """
    _assert_error_code(verify_all, src, "banned-name")


# ── dynamic import / reflection chains ───────────────────────────────────────


def test_aliased_import_getattr(verify_all):
    src = """
    i = __import__
    g = getattr
    g(i('os'), 'getcwd')()
    """
    _assert_error_code(verify_all, src, "banned-name")


def test_aliased_getattr_on_builtins(verify_all):
    _assert_error_code(verify_all, "g = getattr", "banned-name")


def test_aliased_import_getattr_system(verify_all):
    # Full import+invoke chain built entirely in the private region.
    src = """
    i = __import__
    g = getattr
    g(g(i('os'), 'system'))('id')
    """
    _assert_error_code(verify_all, src, "banned-name")


def test_vars_alias_on_builtins(verify_all):
    src = """
    v = vars
    ev = v(__builtins__)['eval']
    ev('1/0')
    """
    _assert_error_code(verify_all, src, "banned-name")


def test_self_dynamic_import_chain(verify_all):
    # Public setup stashes a dynamic-import chain; private __call__ must not run it.
    src = """
    class M:
        def setup(self):
            self.imp = __import__
            self.get = getattr
        def __call__(self, x):
            os = self.imp('os')
            self.get(os, 'system')('id')
    """
    _assert_error_code(verify_all, src, "attr-on-value", private=[[5, 7]])


# ── public import resolution / disallow ──────────────────────────────────────


def test_bare_name_import_bypass(verify_all):
    # Bare name from `from X import f` must use the same allow/deny as a dotted path.
    src = """
    from os import system

    def run():
        system('id')
    """
    _assert_error_code(verify_all, src, "call-not-allowed", private=[[3, 4]])


def test_bare_name_disallowed_leaf_bypass(verify_all):
    src = """
    from jax.numpy import save

    def run(x):
        save(x, 'stolen_data.zip')
    """
    _assert_error_code(
        verify_all,
        src,
        "call-not-allowed",
        private=[[3, 4]],
        pol=make_policy(disallow=["jax.numpy.save"]),
    )


def test_bare_name_import_alias_bypass(verify_all):
    # Renaming (`as persist`) must not bypass a disallow on the underlying leaf.
    src = """
    from jax.numpy import save as persist

    def run(x):
        persist(x, 'stolen_data.zip')
    """
    _assert_error_code(
        verify_all,
        src,
        "call-not-allowed",
        private=[[3, 4]],
        pol=make_policy(disallow=["jax.numpy.save"]),
    )


def test_importlib_import_module_bypass(verify_all):
    src = """
    from importlib import import_module

    def run():
        import_module('os')
    """
    _assert_error_code(verify_all, src, "call-not-allowed", private=[[3, 4]])


# ── homoglyphs ───────────────────────────────────────────────────────────────


def test_unicode_homoglyphs_should_not_be_allowed(verify_all):
    # Public stash + private call through a lookalike name (Cyrillic о р е).
    src = """
    ореn = open
    f = ореn('/etc/passwd')
    """
    _assert_error_code(verify_all, src, "call-unresolved", private=[[2, 3]])


def test_homoglyph_self_stash_and_call(verify_all):
    # Public setup stashes open under a homoglyph attr; private __call__ must not invoke it.
    src = """
    class M:
        def setup(self):
            self.ореn = open
        def __call__(self, x):
            self.ореn('/etc/passwd')
    """
    _assert_error_code(verify_all, src, "attr-on-value", private=[[4, 5]])


def test_homoglyph_two_hop_alias(verify_all):
    # ASCII stash then homoglyph copy — no banned name textually at the call site.
    src = """
    x = open
    ореn = x
    ореn('/etc/passwd')
    """
    _assert_error_code(verify_all, src, "call-unresolved", private=[[3, 4]])


# ── self-attribute trust ─────────────────────────────────────────────────────


def test_self_stash_and_call(verify_all):
    # Public setup stashes open on self; private __call__ must not get a free pass via self.*
    src = """
    class M:
        def setup(self):
            self.open = open
        def __call__(self, x):
            self.open('/etc/passwd')

    m = M()
    """
    _assert_error_code(verify_all, src, "attr-on-value", private=[[4, 5]])


def test_self_nested_attribute_chain(verify_all):
    # Only a single self.<name> level is exempt — self.a.b(...) is not.
    # Stash in public setup; deeper chain call is private.
    src = """
    class M:
        def setup(self):
            self.sub = object()
            self.sub.evil = open
        def __call__(self, x):
            self.sub.evil('/etc/passwd')

    m = M()
    """
    _assert_error_code(verify_all, src, "attr-on-value", private=[[5, 6]])


def test_self_augassign_alias(verify_all):
    # Public setup seeds self.evil; private __call__ mutates and calls it.
    src = """
    class M:
        def setup(self):
            self.evil = print
        def __call__(self, x):
            self.evil += open
            self.evil('/etc/passwd')

    m = M()
    """
    _assert_error_code(verify_all, src, "attr-on-value", private=[[4, 6]])


def test_self_layer_subscript_call(verify_all):
    # Public setup builds a tainted layer list; private Flax-style call must still be vetted.
    src = """
    class M:
        def setup(self):
            self.layer = [open]
        def __call__(self, x):
            self.layer[0]('/etc/passwd')
    """
    _assert_error_code(verify_all, src, "attr-on-value", private=[[4, 5]])


def test_self_attr_tuple_unpack_must_not_grant_trust(verify_all):
    # setup is public/empty; the unpack+call that forges trust is private.
    src = """
    class M:
        def setup(self):
            pass
        def __call__(self, x, fn):
            self.fn, self.other = fn, 0
            return self.fn(x)
    """
    _assert_error_code(verify_all, src, "attr-on-value", private=[[4, 6]])


def test_self_attr_for_loop_target_must_not_grant_trust(verify_all):
    # Public setup binds self.leak via for-target; private __call__ must not treat it as inherited.
    src = """
    class Model:
        def setup(self, cb):
            for self.leak in (cb,):
                pass
        def __call__(self, x, secret):
            self.leak(secret)
            return x
    """
    _assert_error_code(verify_all, src, "attr-on-value", private=[[5, 7]])


def test_self_attr_local_alias_must_not_grant_trust(verify_all):
    # Public setup assigns unsafe self.fn; private tmp = self.fn; tmp(x) must not defeat trust.
    src = """
    class M:
        def setup(self):
            self.fn = lambda x: x + 1
        def __call__(self, x):
            tmp = self.fn
            return tmp(x)
    """
    _assert_error_code(verify_all, src, "call-unresolved", private=[[4, 6]])


def test_self_attr_subscript_local_alias_must_not_grant_trust(verify_all):
    # Public setup stashes opaque cb; private layer[i] alias must not defeat trust.
    src = """
    class M:
        def setup(self, cb):
            self.layer = [cb]
        def __call__(self, x):
            tmp = self.layer[0]
            return tmp(x)
    """
    _assert_error_code(verify_all, src, "call-unresolved", private=[[4, 6]])


def test_self_attr_two_hop_local_alias_must_not_grant_trust(verify_all):
    src = """
    class M:
        def setup(self):
            self.fn = lambda x: x + 1
        def __call__(self, x):
            tmp = self.fn
            tmp2 = tmp
            return tmp2(x)
    """
    _assert_error_code(verify_all, src, "call-unresolved", private=[[4, 7]])


def test_self_dunder_call_should_not_be_allowed(verify_all):
    # self.__getattribute__(...) must hit the dunder-attr ban on the call path.
    src = """
    class M:
        def __call__(self, x):
            d = self.__getattribute__('__dict__')
            return x
    """
    _assert_error_code(verify_all, src, "dunder-attr")


# ── self/cls lexical trust must not be forgeable ─────────────────────────────


def test_self_reassignment_must_not_grant_trust(verify_all):
    src = """
    class M:
        def __call__(self, x):
            self = x
            return self.anything_at_all()
    """
    _assert_error_code(verify_all, src, "reserved-name")


def test_cls_as_local_variable_must_not_grant_trust(verify_all):
    src = """
    class M:
        def __call__(self, x):
            cls = x
            return cls.anything_at_all()
    """
    _assert_error_code(verify_all, src, "reserved-name")


def test_nested_function_self_parameter_must_not_grant_trust(verify_all):
    # Public Wrapper.setup stashes system; private M.__call__ must not grant nested
    # helper(self) the self.<attr> exemption for a foreign object.
    src = """
    from os import system
    from flax import linen as nn

    class Wrapper(nn.Module):
        def setup(self):
            self.run = system

    class M(nn.Module):
        def __call__(self, x):
            w = Wrapper()
            def helper(self):
                return self.run('id')
            return helper(w)
    """
    _assert_error_code(verify_all, src, "reserved-name", private=[[9, 13]])


def test_self_only_trusted_as_first_param_of_a_direct_method(verify_all):
    src = """
    class M:
        def __call__(self, x):
            def helper(x, self):
                return self.evil()
            return helper(1, 2)
    """
    _assert_error_code(verify_all, src, "reserved-name")

    lambda_src = """
    class M:
        def __call__(self, x):
            f = lambda self: self.evil()
            return f(x)
    """
    _assert_error_code(verify_all, lambda_src, "reserved-name")


def test_comprehension_target_rebinding_is_checked(verify_all):
    # ast.comprehension has no lineno; targets must still be checked as Store names.
    _assert_error_code(verify_all, "y = [cls for cls in [1, 2]]", "reserved-name")


def test_class_name_shadowing_reserved_import_alias(verify_all):
    # Class name is a Store binding — shadowing jnp must be caught like jnp = evil.
    src = """
    import jax.numpy as jnp

    def helper(x, secret_sink):
        class jnp:
            einsum = secret_sink
        return jnp.einsum("ij->i", x)
    """
    _assert_error_code(verify_all, src, "reserved-name")


def test_def_name_shadowing_public_wrapper(verify_all):
    # Def shadowing a public wrapper name must be caught.
    src = """
    def transpose(x):
        return x

    def helper(x, secret_sink):
        def transpose(y):
            return secret_sink(y)
        return transpose(x)
    """
    _assert_error_code(verify_all, src, "reserved-name", private=[[4, 7]])


# ── private_defs trusted by identifier: must not be rebindable ───────────────


def test_private_def_assign_rebind_must_not_grant_call(verify_all):
    # Bare calls trust private_defs by name alone. Rebinding helper = evil then
    # helper(x) must fail at the rebind.
    src = """
    def helper(x):
        return x
    def f(x, evil):
        helper = evil
        return helper(x)
    """
    _assert_error_code(verify_all, src, "reserved-name")


def test_private_def_for_rebind_must_not_grant_call(verify_all):
    src = """
    def helper(x):
        return x
    def f(items):
        for helper in items:
            helper(1)
    """
    _assert_error_code(verify_all, src, "reserved-name")


def test_private_def_comprehension_rebind_must_not_grant_call(verify_all):
    src = """
    def helper(x):
        return x
    def f(items):
        return [helper for helper in items]
    """
    _assert_error_code(verify_all, src, "reserved-name")


def test_private_def_nested_def_shadow_must_not_grant_call(verify_all):
    src = """
    def helper(x):
        return x
    def f(x):
        def helper(y):
            return y
        return helper(x)
    """
    _assert_error_code(verify_all, src, "reserved-name")


def test_private_def_public_region_rebind_must_not_grant_call(verify_all):
    # Multi-range hole: private def, public rebind, private call. Public glue
    # must not be allowed to shadow a private def name that call sites still trust.
    src = """
    def helper(x):
        return x

    helper = evil

    def run(x):
        return helper(x)
    """
    _assert_error_code(verify_all, src, "reserved-name", private=[[1, 2], [7, 8]])


def test_declared_field_callable_must_not_grant_trust(verify_all):
    # A class-level dataclass-style field (`fn: object`, never assigned via self.fn = ... inside
    # the class) is populated by whatever constructs the instance -- unlike a self.<attr> = value
    # assignment, there's no expression here for the verifier to vet, so it must not default to
    # "safe" the way a genuinely inherited base-class attribute would.
    src = """
    class M:
        fn: object

        def __call__(self, x):
            return self.fn(x)
    """
    _assert_error_code(verify_all, src, "attr-on-value")


def test_duplicate_method_name_in_same_class_must_be_flagged(verify_all):
    # Python silently keeps only the last definition and discards the rest -- a reviewer (and
    # _SelfAttrTrust, which reasons about the whole class regardless of the public/private split)
    # could be looking at a method body that never actually runs.
    src = """
    class M:
        def __call__(self, x):
            return x

        def __call__(self, x):
            return x + 1
    """
    _assert_error_code(verify_all, src, "duplicate-method")


def test_duplicate_method_name_split_across_public_and_private_must_be_flagged(verify_all):
    # The class statement and one of the two `setup` methods are public; the check must not be
    # gated on the class itself being in the private range, since methods routinely straddle both.
    src = """
    class M:
        def setup(self):
            pass

        def setup(self):
            self.h = object()

        def __call__(self, x):
            return x
    """
    _assert_error_code(verify_all, src, "duplicate-method", private=[[8, 9]])


def test_hook_name_rebound_to_call_result_must_not_grant_trust(verify_all):
    # Rebinding a hook/method name at class-body level to the result of a call reproduces
    # decorator behaviour (`__call__ = jax.grad(_inner)` == `@jax.grad`) while sidestepping the
    # decorator ban: the call is checked as an ordinary allow_functions call, and the rebind of
    # __call__ isn't caught because methods are excluded from reserved-name protection (the same
    # exclusion behind the duplicate-method gap). jax.grad passes only via the broad "jax.*" glob.
    # CURRENTLY FAILS: verify() reports zero violations for this.
    src = """
    import jax
    from flax import linen as nn

    class M(nn.Module):
        def _inner(self, x):
            return x
        __call__ = jax.grad(_inner)
    """
    _assert_error_code(verify_all, src, "reserved-name", private=[[4, 7]])
