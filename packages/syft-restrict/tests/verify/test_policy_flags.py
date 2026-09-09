"""The optional strictness flags: allow_local_assignments / allow_base_class_attributes.

Both default True (the behavior the rest of the suite assumes). These tests pin what flipping each
to False does: it turns a construct that is normally accepted into a rejection.
"""

from verify.helpers import get_error_codes, make_policy

# ── allow_local_assignments ──────────────────────────────────────────────────

_LOCAL_ALIAS = """
class M:
    def __call__(self, x):
        block = self.layers[0]
        return block(x)
"""


def test_local_alias_call_allowed_by_default(verify_all):
    # block = self.layers[0]; block(x) — the alias is tracked as a safe call target.
    assert not get_error_codes(verify_all(_LOCAL_ALIAS))


def test_local_alias_call_rejected_when_local_assignments_disabled(verify_all):
    # With tracking off, the alias is no longer a trusted callee; the call is unresolved.
    codes = get_error_codes(
        verify_all(_LOCAL_ALIAS, pol=make_policy(allow_local_assignments=False))
    )
    assert "call-unresolved" in codes


def test_direct_self_subscript_call_still_allowed_when_local_assignments_disabled(
    verify_all,
):
    # The flag only removes the *alias* convenience; calling the value directly still works.
    src = """
    class M:
        def __call__(self, x):
            return self.layers[0](x)
    """
    assert not get_error_codes(
        verify_all(src, pol=make_policy(allow_local_assignments=False))
    )


# ── allow_base_class_attributes ─────────────────────────────────────────────────

_UNKNOWN_ATTR = """
class M:
    def __call__(self, x):
        return self.norm(x)
"""


def test_unknown_self_attr_call_allowed_by_default(verify_all):
    # self.norm is never assigned in the class -> presumed inherited from the vetted base.
    assert not get_error_codes(verify_all(_UNKNOWN_ATTR))


def test_unknown_self_attr_call_rejected_when_base_class_attributes_disabled(
    verify_all,
):
    # With the base-class assumption off, a never-assigned self.<attr> is not callable.
    codes = get_error_codes(
        verify_all(_UNKNOWN_ATTR, pol=make_policy(allow_base_class_attributes=False))
    )
    assert "attr-on-value" in codes


def test_assigned_self_attr_still_callable_when_base_class_attributes_disabled(
    verify_all,
):
    # The flag rejects only *unknown* attrs; one the class assigns a vetted source stays callable.
    src = """
    class Block:
        def __call__(self, x):
            return x
    class M:
        def setup(self):
            self.blk = Block()
        def __call__(self, x):
            return self.blk(x)
    """
    assert not get_error_codes(
        verify_all(src, pol=make_policy(allow_base_class_attributes=False))
    )
