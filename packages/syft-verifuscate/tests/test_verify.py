"""Tests for the static checker (approach B)."""

from pathlib import Path

import pytest

from syft_verifuscate import Policy, verify

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).parents[3]

ALLOW_FUNCTIONS = "jax.*, flax.linen.*"
ALLOW_METHODS = "arithmetic, indexing, comparison, metadata"


def _policy():
    return Policy.parse(ALLOW_FUNCTIONS, ALLOW_METHODS)


def _verify_all(source: str):
    n = len(source.splitlines())
    return verify(source, [[1, n]], _policy())


def test_compliant_fixture_passes():
    source = (FIXTURES / "compliant_model.py").read_text()
    # mark the model definition (everything from CONFIG onward) as private
    config_line = next(
        i for i, ln in enumerate(source.splitlines(), 1) if ln.startswith("CONFIG")
    )
    result = verify(source, [[config_line, len(source.splitlines())]], _policy())
    assert result.ok, [f"L{v.line} {v.code}: {v.message}" for v in result.violations]
    assert result.n_calls_checked > 0


@pytest.mark.parametrize(
    "code, snippet",
    [
        ("banned-construct", "import os\n"),
        ("banned-call", "y = eval('1 + 1')\n"),
        ("banned-call", "z = getattr(obj, name)\n"),
        ("method-on-value", "a = x.reshape(8, -1)\n"),
        ("method-on-value", "b = '{0.__class__}'.format(payload)\n"),
        ("decorator", "@evil\ndef f():\n    return 1\n"),
        ("dunder-attr", "c = obj.__class__\n"),
    ],
)
def test_rejections(code, snippet):
    result = _verify_all(snippet)
    assert not result.ok
    assert code in {v.code for v in result.violations}, [
        (v.code, v.message) for v in result.violations
    ]


def test_reserved_module_alias_cannot_be_rebound():
    source = "import jax.numpy as jnp\njnp = make_evil()\n"
    # only the rebind line is private (the import is visible glue)
    result = verify(source, [[2, 2]], _policy())
    assert "reserved-name" in {v.code for v in result.violations}


def test_jax_denylist_beats_allow():
    # io_callback is under an allowed module (jax.*) but on the denylist
    source = "import jax\nq = jax.experimental.io_callback(send, x)\n"
    result = verify(source, [[2, 2]], _policy())
    assert "call-not-allowed" in {v.code for v in result.violations}


def test_operator_bundle_must_be_enabled():
    source = "r = a + b\n"
    # arithmetic NOT enabled -> the BinOp is rejected
    policy = Policy.parse(ALLOW_FUNCTIONS, "indexing")
    result = verify(source, [[1, 1]], policy)
    assert "bundle-disabled" in {v.code for v in result.violations}


def test_real_gemma_flags_named_methods_on_values():
    """Matches research approach-B §3.6.3: the real file still has method/attr-on-value spots."""
    gemma = REPO_ROOT / "koen" / "gemma_inference.py"
    if not gemma.exists():
        pytest.skip("gemma_inference.py not present")
    source = gemma.read_text()
    result = verify(source, [[22, 280]], _policy())
    assert not result.ok
    messages = " | ".join(v.message for v in result.violations)
    # the documented spots: `module.variable(...).value` in _get, and `embed_table.T`
    assert "'variable'" in messages
    assert "'T'" in messages
    assert {"method-on-value", "attr-on-value"} <= {v.code for v in result.violations}
