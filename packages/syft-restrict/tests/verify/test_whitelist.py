"""Whitelisted cases: things the private region is explicitly allowed to do."""

from syft_restrict import verify

from verify.helpers import FIXTURES, get_error_codes, make_policy, normalize_source


def _ok(result):
    return result.ok, [f"L{v.line} {v.code}: {v.message}" for v in result.violations]


# ── fixtures / whole-file green path ─────────────────────────────────────────


def test_compliant_fixture_passes(policy):
    """The compliant fixture (model definition) passes with no violations."""
    source = (FIXTURES / "compliant_model.py").read_text()
    config_line = next(
        i for i, ln in enumerate(source.splitlines(), 1) if ln.startswith("CONFIG")
    )
    result = verify(source, [[config_line, len(source.splitlines())]], policy)
    ok, detail = _ok(result)
    assert ok, detail
    assert result.n_calls_checked > 0


# ── bare names, builtins, local defs ─────────────────────────────────────────


def test_bare_name_calls_allowed(verify_all):
    """Calling a local var / private def / safe builtin by bare name is fine."""
    src = """
    def helper(n):
        rows = list(range(n))
        vals = tuple(rows)
        total = sum(vals)
        return helper(total)
    """
    assert _ok(verify_all(src))[0]


def test_safe_builtin_call_is_allowed(verify_all):
    src = """
    def helper(n):
        return list(range(n))
    """
    assert _ok(verify_all(src))[0]


def test_safe_aliased_builtin_calls(verify_all):
    """A local bound to a safe builtin is still callable."""
    src = """
    g = len
    g([1, 2, 3])
    """
    assert _ok(verify_all(src))[0]


def test_public_call_is_allowed(verify_all):
    """A public-region def may be called from the private region."""
    src = """
    def helper():
        return 1
    helper()
    """
    # def is public (lines 1–2); call is private (line 3)
    result = verify_all(src, private=[[3, 3]])
    assert _ok(result)[0]


def test_public_class_constructor_is_allowed(verify_all):
    """A public-region class may be constructed by bare name from the private region."""
    src = """
    class Block:
        def __call__(self, x):
            return x
    def run(x):
        b = Block()
        return b(x)
    """
    # class Block is public (lines 1–3); run is private (lines 4–6)
    result = verify_all(src, private=[[4, 6]])
    assert _ok(result)[0]


def test_verify_does_not_mutate_caller_policy(policy):
    """verify() must not write reserved_names onto the caller's Policy instance."""
    assert policy.reserved_names == set()
    src = normalize_source("""
    import jax.numpy as jnp
    def f():
        return 1
    """)
    verify(src, [[3, 4]], policy)
    assert policy.reserved_names == set()


def test_private_call_is_allowed(verify_all):
    """A private-region def may be called from the private region."""
    src = """
    def helper():
        return 1
    helper()
    """
    assert _ok(verify_all(src))[0]


def test_local_bound_to_private_constructor_is_still_callable(verify_all):
    """``block = Attn(); block(x)`` — local traced to a class defined here."""
    src = """
    class Attn:
        def __call__(self, x):
            return x

    def helper(x):
        block = Attn()
        return block(x)
    """
    assert _ok(verify_all(src))[0]


def test_chained_call_through_private_constructor_is_allowed(verify_all):
    """``Block()(x)``: callee is itself a Call to a class defined here."""
    src = """
    class Block:
        def __call__(self, x):
            return x

    def helper(x):
        return Block()(x)
    """
    assert _ok(verify_all(src))[0]


# ── self / cls ───────────────────────────────────────────────────────────────


def test_self_method_calls_allowed(verify_all):
    """``self.method(...)`` / inherited ``self.param`` — receiver is the module class."""
    src = """
    class M:
        def setup(self):
            self.w = self.param('w')
        def __call__(self, x):
            return self.norm(x)
    """
    assert _ok(verify_all(src))[0]


def test_calling_a_value_through_self_is_allowed(verify_all):
    """Calling a trusted self-rooted subscript/local alias is allowed."""
    src = """
    class M:
        def __call__(self, x):
            layer = self.layers[0]
            return self.layers[0](x) + layer(x)
    """
    assert _ok(verify_all(src))[0]


def test_self_attr_local_alias_of_safe_source_is_still_allowed(verify_all):
    src = """
    class M:
        def __call__(self, x):
            layer = self.layers[0]
            return layer(x)
    """
    assert _ok(verify_all(src))[0]


def test_self_attr_assigned_allowlisted_import_call_is_callable(policy):
    src = normalize_source("""
    from flax import linen as nn

    class M(nn.Module):
        def setup(self):
            self.dense = nn.Dense(8)
        def __call__(self, x):
            return self.dense(x)
    """)
    result = verify(src, [[3, 7]], policy)
    assert _ok(result)[0]


def test_self_attr_list_of_constructors_element_alias_is_callable(verify_all):
    """The Gemma-style ``block = self.layer[0]; block(x)`` pattern."""
    src = """
    class Block:
        def __call__(self, x):
            return x

    class M:
        def setup(self):
            self.layer = [Block() for _ in range(3)]
        def __call__(self, x):
            block = self.layer[0]
            return block(x)
    """
    assert _ok(verify_all(src))[0]


def test_self_attr_reassigned_local_alias_is_not_stuck_tainted(verify_all):
    """Reassigning a local that once aliased self.<attr> clears the taint."""
    src = """
    class M:
        def setup(self):
            self.fn = lambda x: x + 1
        def __call__(self, x):
            tmp = self.fn
            tmp = x
            return tmp
    """
    assert _ok(verify_all(src))[0]


# ── decorators / dunder defs / bases ─────────────────────────────────────────


def test_allowed_decorators(policy):
    """Only the allow-listed decorators may sit above a def/class (imports stay public)."""
    src = normalize_source("""
    from flax import linen as nn
    import jax

    class M:
        @nn.compact
        @jax.jit
        def __call__(self, x):
            return x
    """)
    result = verify(src, [[4, len(src.splitlines())]], policy)
    assert _ok(result)[0]


def test_allowed_dunder_defs(verify_all):
    """``__call__``, ``setup`` and ``__post_init__`` are the only definable hooks."""
    src = """
    class M:
        def setup(self):
            return None
        def __post_init__(self):
            return None
        def __call__(self, x):
            return x
    """
    assert _ok(verify_all(src))[0]


def test_allowlisted_import_as_base_class(policy):
    """A base class must be an allow-listed import (e.g. ``nn.Module``)."""
    src = normalize_source("""
    from flax import linen as nn
    class M(nn.Module):
        def __call__(self, x):
            return x
    """)
    result = verify(src, [[2, len(src.splitlines())]], make_policy())
    assert _ok(result)[0]


# ── control flow / assignment / annotations ──────────────────────────────────


def test_control_flow_and_comprehensions(verify_all):
    """if/for/while/ternary, comprehensions and a lambda are all on the node allow-list.

    Uses ``abs`` (a safe builtin) rather than an arbitrary bare name for the comprehension's call;
    unresolvable bare-name calls are covered in ``test_disallowed.py``.
    """
    src = """
    def f(xs):
        acc = [abs(v) for v in xs if v]
        pairs = {k: k for k in xs}
        seen = {v for v in xs}
        h = (lambda z: z)
        out = 0
        for v in xs:
            out = v
            if out:
                break
        while out:
            out = 0
        return out if xs else acc
    """
    assert _ok(verify_all(src))[0]


def test_non_reserved_rebind_is_fine(verify_all):
    """Reassigning ordinary locals (not import aliases / wrapper names) is allowed."""
    src = """
    def f(x):
        y = x
        y += 1
        z: int = y
        return z
    """
    result = verify_all(src)
    assert _ok(result)[0]
    assert "reserved-name" not in get_error_codes(result)


def test_arithmetic_only_policy_still_passes_pure_math(verify_all):
    """A bundle the code doesn't use being disabled doesn't cause spurious failures."""
    pol = make_policy(methods=["arithmetic"])
    src = """
    def f(a, b):
        return a + b - (-a)
    """
    assert _ok(verify_all(src, pol))[0]


def test_nested_generic_annotation_is_not_flagged(verify_all):
    """Banned-name identifiers inside generic annotations are not references."""
    src = """
    def f(x: list[str]) -> dict[str, bytes]:
        return {}
    """
    assert _ok(verify_all(src))[0]


def test_future_annotations_import_does_not_change_annotation_handling(policy):
    """``from __future__ import annotations`` does not change the AST shape of annotations."""
    src = normalize_source("""
    from __future__ import annotations

    def f(x: str) -> list[bytes]:
        return []
    """)
    result = verify(src, [[3, 4]], policy)
    assert _ok(result)[0]


def test_stringized_annotation_is_not_flagged(verify_all):
    """Quoted forward-reference annotations are Constants, not Names."""
    src = """
    def f(x: "str") -> "bytes":
        return x
    """
    assert _ok(verify_all(src))[0]
