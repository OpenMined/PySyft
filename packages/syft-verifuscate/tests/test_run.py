"""End-to-end tests for run()."""

import shutil
from pathlib import Path

import pytest

from syft_verifuscate import PolicyViolation, run

FIXTURES = Path(__file__).parent / "fixtures"
ALLOW_FUNCTIONS = "jax.*, flax.linen.*"
ALLOW_METHODS = "arithmetic, indexing, comparison, metadata"


def _private(source: str):
    config_line = next(
        i for i, ln in enumerate(source.splitlines(), 1) if ln.startswith("CONFIG")
    )
    return [[config_line, len(source.splitlines())]]


def test_run_success_writes_obfuscated_and_certificate(tmp_path):
    src = tmp_path / "model.py"
    shutil.copy(FIXTURES / "compliant_model.py", src)
    result = run(
        src,
        private=_private(src.read_text()),
        allow_functions=ALLOW_FUNCTIONS,
        allow_methods=ALLOW_METHODS,
    )
    assert result.ok
    out = Path(result.obfuscated_path)
    assert out.exists() and out.name == "model.obfuscated.py"
    assert result.certificate["source_sha256"]
    assert result.certificate["policy_id"]
    assert result.certificate["verifuscate_version"]
    assert result.certificate["n_calls_checked"] > 0


def test_run_strict_raises_and_writes_nothing(tmp_path):
    src = tmp_path / "bad.py"
    src.write_text("CONFIG = dict(dim=8)\nimport os\nleak = os.getcwd()\n")
    with pytest.raises(PolicyViolation) as exc:
        run(
            src,
            private=[[1, 3]],
            allow_functions=ALLOW_FUNCTIONS,
            allow_methods=ALLOW_METHODS,
        )
    assert exc.value.violations
    assert not (tmp_path / "bad.obfuscated.py").exists()


def test_run_nonstrict_returns_violations(tmp_path):
    src = tmp_path / "bad.py"
    src.write_text("CONFIG = dict(dim=8)\nleak = x.reshape(1)\n")
    result = run(
        src,
        private=[[1, 2]],
        allow_functions=ALLOW_FUNCTIONS,
        allow_methods=ALLOW_METHODS,
        strict=False,
    )
    assert not result.ok
    assert any(v.code == "method-on-value" for v in result.violations)
    assert result.obfuscated_path is None
    assert not (tmp_path / "bad.obfuscated.py").exists()
