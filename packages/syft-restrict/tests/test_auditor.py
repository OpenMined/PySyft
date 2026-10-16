"""Tests for the advisory allow-list audit (syft_restrict.auditor)."""

import json
from pathlib import Path

from syft_restrict import AuditReport, audit_allow_functions
from syft_restrict.auditor import _best_version_key
from syft_restrict.catalog_lint import _canonical
from syft_restrict.catalog_lint import main as lint_main

# The example catalog is not bundled in the package; tests point the audit at it explicitly.
EXAMPLE_CATALOG = Path(__file__).resolve().parent.parent / "examples" / "catalog"


def _entry(report: AuditReport, path: str):
    return next(e for e in report.entries if e.path == path)


def test_without_catalog_dir_everything_is_review():
    # No catalog ships with the package. With no catalog_dir there are no rules, so every non-glob
    # path is deferred to review (never silently safe).
    report = audit_allow_functions(
        ["jax.numpy.einsum", "jax.numpy.save", "flax.linen.Module"]
    )
    assert all(e.verdict == "review" for e in report.entries)
    assert report.ok  # review does not fail the report


def test_glob_allow_is_flagged_unsafe():
    # Globs are flagged by the classifier itself, so this holds with or without a catalog.
    report = audit_allow_functions(["jax.*", "flax.linen.*"])
    assert _entry(report, "jax.*").verdict == "unsafe"
    assert _entry(report, "flax.linen.*").verdict == "unsafe"
    assert "glob" in _entry(report, "jax.*").reason


def test_uncatalogued_path_is_deferred_to_review_without_assumptions():
    # Even with a catalog present, an unknown path is neither safe nor unsafe: the audit makes no
    # guess and defers to a human, regardless of whether the path is importable.
    report = audit_allow_functions(
        ["totally.made.up.symbol", "shutil.copyfile"], catalog_dir=EXAMPLE_CATALOG
    )
    for path in ("totally.made.up.symbol", "shutil.copyfile"):
        e = _entry(report, path)
        assert e.verdict == "review"
        assert (
            "catalog" in e.reason
        )  # reported as uncatalogued, deferred to human review
    assert report.ok  # review entries do not fail the report; they need a human


def test_cross_library_pattern_matches_any_library():
    # `*.io_callback` lives in the library-agnostic _common catalog
    report = audit_allow_functions(["somelib.io_callback"], catalog_dir=EXAMPLE_CATALOG)
    assert _entry(report, "somelib.io_callback").verdict == "unsafe"


def test_lint_accepts_a_path_and_fixes(tmp_path):
    cat = tmp_path / "mylib" / "1.0"
    cat.mkdir(parents=True)
    f = cat / "catalog.json"
    f.write_text(
        '{\n  "safe": {"b": "two", "a": "one"}\n}\n'
    )  # unsorted, not canonical
    assert lint_main([str(tmp_path)]) == 1  # check mode flags it
    assert lint_main([str(tmp_path), "--fix"]) == 0  # --fix rewrites it
    assert lint_main([str(tmp_path)]) == 0  # now canonical
    assert list(json.loads(f.read_text())["safe"]) == ["a", "b"]  # keys sorted


def test_lint_reports_entries_a_stricter_bucket_already_matches(tmp_path, capsys):
    # A safe entry swallowed by an unsafe glob audits as unsafe, so the safe claim never applies.
    cat = tmp_path / "mylib" / "1.0"
    cat.mkdir(parents=True)
    (cat / "catalog.json").write_text(
        _canonical(
            {
                "unsafe": {"mylib.*.test": "test runner", "mylib.io.save": "writes"},
                "safe": {"mylib.fft.test": "pure", "mylib.io.save": "pure"},
            }
        )
    )
    assert lint_main([str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "'mylib.fft.test' in 'safe' is already matched by 'mylib.*.test'" in err
    assert "'mylib.io.save' in 'safe' is already matched by 'mylib.io.save'" in err


def test_lint_fix_does_not_silence_an_overlap(tmp_path):
    # --fix rewrites formatting; choosing a bucket is a human call, so the overlap must survive it.
    cat = tmp_path / "mylib" / "1.0"
    cat.mkdir(parents=True)
    (cat / "catalog.json").write_text(
        '{"unsafe": {"mylib.b": "x", "mylib.a": "y"}, "safe": {"mylib.a": "z"}}'
    )
    assert lint_main([str(tmp_path), "--fix"]) == 1
    assert lint_main([str(tmp_path)]) == 1  # formatting fixed, overlap still reported


def test_lint_accepts_disjoint_buckets(tmp_path):
    cat = tmp_path / "mylib" / "1.0"
    cat.mkdir(parents=True)
    (cat / "catalog.json").write_text(
        _canonical({"unsafe": {"mylib.io.*": "writes"}, "safe": {"mylib.add": "pure"}})
    )
    assert lint_main([str(tmp_path)]) == 0


def test_best_version_key_matches_on_dot_boundaries_only():
    # A version dir must match a whole version component, not a raw string prefix: the "0.1" dir
    # applies to 0.1.x, never to 0.11.x / 0.19.x (which look like "0.1" prefixes as bare strings).
    # There is no baseline fallback — an uncovered version resolves to None (no library rules).
    keys = ["0.1", "0.11"]
    assert _best_version_key(keys, "0.1.7") == "0.1"
    assert _best_version_key(keys, "0.11.0") == "0.11"  # not "0.1"
    assert (
        _best_version_key(keys, "0.19.2") is None
    )  # no baseline; uncovered -> no rules
    assert _best_version_key(keys, "0.2.0") is None
    assert (
        _best_version_key(keys, "") is None
    )  # unknown/undetected version matches nothing


def test_catalog_dir_supplies_the_rules(tmp_path):
    # Without a catalog_dir a path is 'review'; a catalog_dir is what provides its rules.
    assert _entry(audit_allow_functions(["mylib.add"]), "mylib.add").verdict == "review"
    common = tmp_path / "_common" / "default"
    common.mkdir(parents=True)
    (common / "catalog.json").write_text(
        json.dumps({"unsafe": {"mylib.add": "custom rule"}})
    )
    report = audit_allow_functions(["mylib.add"], catalog_dir=tmp_path)
    add = _entry(report, "mylib.add")
    assert add.verdict == "unsafe"
    assert add.reason == "custom rule"


def test_malformed_catalog_degrades_to_review(tmp_path):
    # A broken catalog.json must not crash the advisory audit; its paths fall to review.
    common = tmp_path / "_common" / "default"
    common.mkdir(parents=True)
    (common / "catalog.json").write_text("{ not valid json ")
    report = audit_allow_functions(["mylib.add"], catalog_dir=tmp_path)
    assert _entry(report, "mylib.add").verdict == "review"


def test_report_format_has_sections_and_ok_flag():
    report = audit_allow_functions(["mylib.*", "mylib.add"])
    text = report.format()
    assert "UNSAFE" in text and "REVIEW" in text
    assert "ok=False" in text  # the glob entry fails the report
