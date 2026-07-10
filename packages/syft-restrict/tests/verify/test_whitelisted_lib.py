"""Things we *manually* allow: library calls resolved by name, and operator bundles.

These need real imports, which are themselves banned inside the private region — so the import
lines stay *public* (glue) and only the usage below them is marked private via ``private=``.
"""

import pytest

from verify.helpers import get_error_codes, make_policy

# ── library calls allowed BY NAME (resolved against the public imports) ─────────────


@pytest.mark.parametrize(
    "src",
    [
        """
        import jax.numpy as jnp
        r = jnp.einsum('ij,jk->ik', a, b)
        """,
        """
        import jax
        r = jax.nn.softmax(x)
        """,
        """
        import jax
        r = jax.lax.rsqrt(x)
        """,
        """
        from flax import linen as nn
        layer = nn.Dense(8)
        """,
        """
        from flax import linen as nn
        layer = nn.LayerNorm()
        """,
    ],
)
def test_allowlisted_library_calls(verify_all, src):
    # line 1 = public import; rest = private
    result = verify_all(src, private=[[2, 2]])
    assert result.ok, [(v.code, v.message) for v in result.violations]


def test_attribute_read_on_allowlisted_namespace(verify_all):
    """A constant read off an allow-listed module (``jnp.pi``) resolves and is allowed."""
    src = """
    import jax.numpy as jnp
    r = jnp.pi
    """
    result = verify_all(src, private=[[2, 2]])
    assert result.ok, [(v.code, v.message) for v in result.violations]


# ── operator bundles: enabled => allowed ─────────────────────────────────────


@pytest.mark.parametrize(
    "src",
    [
        "r = a + b - (-a)",  # arithmetic: BinOp / UnaryOp
        "r = (a < b) and (b > a)",  # comparison: Compare / BoolOp
        "r = x[0] + x[1:3]",  # indexing: Subscript / Slice
    ],
)
def test_enabled_operator_bundles(verify_all, src):
    result = verify_all(src)
    assert result.ok, [(v.code, v.message) for v in result.violations]


# ── operator bundles: disabled => bundle-disabled ────────────────────────────


@pytest.mark.parametrize(
    "methods, src",
    [
        (["indexing", "comparison"], "r = a + b"),  # arithmetic off
        (["arithmetic", "indexing"], "r = a < b"),  # comparison off
        (["arithmetic", "comparison"], "r = x[0]"),  # indexing off
    ],
)
def test_disabled_operator_bundle_is_rejected(verify_all, methods, src):
    result = verify_all(src, make_policy(methods=methods))
    assert "bundle-disabled" in get_error_codes(result)


# ── user-supplied disallow beats the allow ───────────────────────────────────


@pytest.mark.parametrize(
    "src",
    [
        """
        import jax
        q = jax.experimental.io_callback(send, x)
        """,
        """
        import jax.numpy as jnp
        q = jnp.save('f', x)
        """,
        """
        import jax
        q = jax.debug.print('x', x)
        """,
    ],
)
def test_user_disallow_beats_allow(verify_all, src):
    # Under a broad `jax.*` allow, an explicit disallow still rejects the dangerous leaves.
    pol = make_policy(disallow=["jax.experimental.*", "jax.numpy.save", "jax.debug.*"])
    result = verify_all(src, pol, private=[[2, 2]])
    assert "call-not-allowed" in get_error_codes(result)


# ── with no disallow, a broad allow permits a formerly-denied leaf ───────────


def test_no_disallow_permits_leaf_under_broad_allow(verify_all):
    # Intentional behavior change: safety comes from a *specific* allow-list or an explicit
    # disallow list; a bare `jax.*` allow with no disallow now permits `jax.numpy.save`.
    src = """
    import jax.numpy as jnp
    q = jnp.save('f', x)
    """
    result = verify_all(src, private=[[2, 2]])
    assert "call-not-allowed" not in get_error_codes(result)


# ── a library that simply isn't on the allow-list ────────────────────────────


def test_non_allowlisted_library_call(verify_all):
    src = """
    import numpy as np
    r = np.dot(a, b)
    """
    result = verify_all(src, private=[[2, 2]])
    assert "call-not-allowed" in get_error_codes(result)


def test_non_allowlisted_library_attribute(verify_all):
    src = """
    import numpy as np
    r = np.pi
    """
    result = verify_all(src, private=[[2, 2]])
    assert "attr-not-allowed" in get_error_codes(result)
