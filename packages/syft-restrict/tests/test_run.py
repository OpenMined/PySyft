"""End-to-end tests for run()."""

import shutil
from pathlib import Path

import pytest
from syft_restrict import MarkerError, PolicyViolation, run
from syft_restrict.runner import _run
from verify.helpers import normalize_source

FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLE_CATALOG = Path(__file__).parent.parent / "examples" / "catalog"
ALLOW_FUNCTIONS = ["jax.*", "flax.linen.*"]
ALLOW_OPERATORS = ["arithmetic", "indexing", "comparison"]


def _private(source: str):
    config_line = next(
        i for i, ln in enumerate(source.splitlines(), 1) if ln.startswith("CONFIG")
    )
    return [[config_line, len(source.splitlines())]]


def test_run_success_writes_obfuscated_and_certificate(tmp_path):
    src = tmp_path / "model.py"
    shutil.copy(FIXTURES / "compliant_model.py", src)
    result = _run(
        src,
        obfuscate=_private(src.read_text()),
        allow_functions=ALLOW_FUNCTIONS,
        allow_operators=ALLOW_OPERATORS,
    )
    assert result.ok
    assert result.obfuscated_path is not None and result.certificate is not None
    out = Path(result.obfuscated_path)
    assert out.exists() and out.name == "model.obfuscated.py"
    assert result.certificate["source_sha256"]
    assert result.certificate["policy_id"]
    assert result.certificate["restrict_version"]
    assert result.certificate["n_calls_checked"] > 0


def test_run_attaches_advisory_audit_without_affecting_ok(tmp_path):
    src = tmp_path / "model.py"
    shutil.copy(FIXTURES / "compliant_model.py", src)
    result = _run(
        src,
        obfuscate=_private(src.read_text()),
        allow_functions=ALLOW_FUNCTIONS,  # globs -> the audit flags them unsafe
        allow_operators=ALLOW_OPERATORS,
    )
    assert result.ok  # a clean verify still passes
    assert result.audit is not None
    assert result.audit.unsafe  # the globs are flagged unsafe by the audit
    assert (
        not result.audit.ok
    )  # the audit is unhappy, yet run().ok stayed True (advisory only)


def test_run_audit_attached_on_verification_failure(tmp_path):
    src = tmp_path / "bad.py"
    src.write_text("CONFIG = dict(dim=8)\nleak = x.reshape(1)\n")
    result = _run(
        src,
        obfuscate=[[1, 2]],
        allow_functions=["jax.numpy.einsum"],
        allow_operators=ALLOW_OPERATORS,
        strict=False,
        catalog_dir=EXAMPLE_CATALOG,
    )
    assert not result.ok  # verification failed
    assert result.audit is not None  # the audit is still attached
    assert any(e.path == "jax.numpy.einsum" for e in result.audit.safe)


def test_run_survives_a_broken_catalog_dir(tmp_path):
    # The advisory audit must never fail the run: a malformed catalog degrades, it does not raise.
    src = tmp_path / "model.py"
    shutil.copy(FIXTURES / "compliant_model.py", src)
    bad = tmp_path / "cat" / "jax" / "0.11"
    bad.mkdir(parents=True)
    (bad / "catalog.json").write_text("{ broken json")
    result = _run(
        src,
        obfuscate=_private(src.read_text()),
        allow_functions=ALLOW_FUNCTIONS,
        allow_operators=ALLOW_OPERATORS,
        catalog_dir=tmp_path / "cat",
    )
    assert result.ok  # a clean verify still passes despite the broken catalog
    assert result.audit is not None


def test_run_strict_raises_and_writes_nothing(tmp_path):
    src = tmp_path / "bad.py"
    src.write_text("CONFIG = dict(dim=8)\nimport os\nleak = os.getcwd()\n")
    with pytest.raises(PolicyViolation) as exc:
        _run(
            src,
            obfuscate=[[1, 3]],
            allow_functions=ALLOW_FUNCTIONS,
            allow_operators=ALLOW_OPERATORS,
        )
    assert exc.value.violations
    assert not (tmp_path / "bad.obfuscated.py").exists()


def test_run_nonstrict_returns_violations(tmp_path):
    src = tmp_path / "bad.py"
    src.write_text("CONFIG = dict(dim=8)\nleak = x.reshape(1)\n")
    result = _run(
        src,
        obfuscate=[[1, 2]],
        allow_functions=ALLOW_FUNCTIONS,
        allow_operators=ALLOW_OPERATORS,
        strict=False,
    )
    assert not result.ok
    assert any(v.code == "method-on-value" for v in result.violations)
    assert result.obfuscated_path is None
    assert not (tmp_path / "bad.obfuscated.py").exists()


def test_run_auto_detects_markers_when_ranges_omitted(tmp_path):
    src = tmp_path / "model.py"
    shutil.copy(FIXTURES / "marked_model.py", src)
    result = run(
        src,
        allow_functions=ALLOW_FUNCTIONS,
        allow_operators=ALLOW_OPERATORS,
    )
    assert result.ok
    assert result.obfuscated_path is not None and result.certificate is not None
    out = Path(result.obfuscated_path)
    assert out.exists()
    # marked_model.py: line 9 is "# syft-restrict: obfuscate-start", line 40 is "...-end" -- the
    # resolved range excludes both marker lines, so it's (10, 39), not the marker lines themselves.
    assert result.certificate["obfuscate_ranges"] == [[10, 41]]
    obfuscated_lines = out.read_text().splitlines()
    # marker lines fall outside the resolved range, so they pass through untouched -- the reader
    # can still see exactly where the private region was, even though its contents are renamed.
    assert obfuscated_lines[8] == "# syft-restrict: obfuscate-start"
    assert obfuscated_lines[41] == "# syft-restrict: obfuscate-end"
    assert (
        "CONFIG" not in out.read_text()
    )  # the private region's own identifiers were renamed


def test_run_without_markers_raises_marker_error(tmp_path):
    # run() is markers-only: a file with no `# syft-restrict:` markers has no private region to
    # resolve, so parse_markers() raises rather than silently verifying nothing.
    src = tmp_path / "unmarked.py"
    src.write_text(
        normalize_source("""
    CONFIG = dict(dim=8)
    def f(x):
        return x
    """)
    )
    with pytest.raises(MarkerError):
        run(src, allow_functions=ALLOW_FUNCTIONS, allow_operators=ALLOW_OPERATORS)


def test__run_uses_explicit_ranges_and_ignores_markers(tmp_path):
    # _run() takes ranges directly and never scans markers, so a stray/unmatched marker in the
    # source (which would make run()'s parse_markers raise) is simply ignored.
    src = tmp_path / "model.py"
    src.write_text(
        normalize_source("""
    # syft-restrict: obfuscate-start
    CONFIG = dict(dim=8)
    x = CONFIG  # unmatched block, no obfuscate-end
    """)
    )
    result = _run(
        src,
        obfuscate=[[1, 3]],
        allow_functions=ALLOW_FUNCTIONS,
        allow_operators=ALLOW_OPERATORS,
    )
    assert result.ok
