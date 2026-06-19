"""Things we *manually* allow: library calls resolved by name, and operator bundles.

These need real imports, which are themselves banned inside the hidden region — so the import lines
stay *visible* (glue) and only the usage below them is marked private.
"""

import pytest

from syft_restrict import verify

from .conftest import error_codes, make_policy


def _verify(header: str, body: str, policy):
    """Keep ``header`` lines visible (imports), mark the ``body`` lines below as private."""
    head_lines = header.splitlines()
    source = header + body
    lo = len(head_lines) + 1
    hi = len(source.splitlines())
    return verify(source, [[lo, hi]], policy)


# ── library calls allowed BY NAME (resolved against the imports) ──────────────────────────
@pytest.mark.parametrize(
    "header, body",
    [
        ("import jax.numpy as jnp\n", "r = jnp.einsum('ij,jk->ik', a, b)\n"),
        ("import jax\n", "r = jax.nn.softmax(x)\n"),
        ("import jax\n", "r = jax.lax.rsqrt(x)\n"),
        ("from flax import linen as nn\n", "layer = nn.Dense(8)\n"),
        ("from flax import linen as nn\n", "layer = nn.LayerNorm()\n"),
    ],
)
def test_allowlisted_library_calls(policy, header, body):
    result = _verify(header, body, policy)
    assert result.ok, [(v.code, v.message) for v in result.violations]


def test_attribute_read_on_allowlisted_namespace(policy):
    """A constant read off an allow-listed module (``jnp.pi``) resolves and is allowed."""
    result = _verify("import jax.numpy as jnp\n", "r = jnp.pi\n", policy)
    assert result.ok, [(v.code, v.message) for v in result.violations]


# ── operator bundles: enabled => allowed ──────────────────────────────────────────────────
@pytest.mark.parametrize(
    "body",
    [
        "r = a + b - (-a)\n",  # arithmetic: BinOp / UnaryOp
        "r = (a < b) and (b > a)\n",  # comparison: Compare / BoolOp
        "r = x[0] + x[1:3]\n",  # indexing: Subscript / Slice
    ],
)
def test_enabled_operator_bundles(verify_all, body):
    result = verify_all(body)
    assert result.ok, [(v.code, v.message) for v in result.violations]


# ── operator bundles: disabled => bundle-disabled ─────────────────────────────────────────
@pytest.mark.parametrize(
    "methods, body",
    [
        (["indexing", "comparison"], "r = a + b\n"),  # arithmetic off
        (["arithmetic", "indexing"], "r = a < b\n"),  # comparison off
        (["arithmetic", "comparison"], "r = x[0]\n"),  # indexing off
    ],
)
def test_disabled_operator_bundle_is_rejected(verify_all, methods, body):
    result = verify_all(body, make_policy(methods=methods))
    assert "bundle-disabled" in error_codes(result)


# ── denylist beats the allow, even under an otherwise-allowed module ──────────────────────
@pytest.mark.parametrize(
    "header, body",
    [
        ("import jax\n", "q = jax.experimental.io_callback(send, x)\n"),
        ("import jax.numpy as jnp\n", "q = jnp.save('f', x)\n"),
        ("import jax\n", "q = jax.debug.print('x', x)\n"),
    ],
)
def test_denylist_beats_allow(policy, header, body):
    result = _verify(header, body, policy)
    assert "call-not-allowed" in error_codes(result)


# ── a library that simply isn't on the allow-list ─────────────────────────────────────────
def test_non_allowlisted_library_call(policy):
    result = _verify("import numpy as np\n", "r = np.dot(a, b)\n", policy)
    assert "call-not-allowed" in error_codes(result)


def test_non_allowlisted_library_attribute(policy):
    result = _verify("import numpy as np\n", "r = np.pi\n", policy)
    assert "attr-not-allowed" in error_codes(result)
