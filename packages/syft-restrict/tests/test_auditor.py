"""Tests for the advisory allow-list audit (syft_restrict.auditor)."""

import json
from pathlib import Path

import jax

from syft_restrict import AuditReport, audit_allow_functions
from syft_restrict.auditor import _best_version_key
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


def test_known_unsafe_paths_are_flagged():
    paths = [
        "jax.profiler.start_server",  # network server / disk trace (the floor-gap we found)
        "jax.distributed.initialize",  # outbound network
        "jax.numpy.save",  # disk
        "jax.experimental.io_callback",  # host callback
        "flax.training.checkpoints.save_checkpoint",  # disk
    ]
    report = audit_allow_functions(paths, catalog_dir=EXAMPLE_CATALOG)
    assert all(_entry(report, p).verdict == "unsafe" for p in paths)
    assert all(
        _entry(report, p).reason for p in paths
    )  # every entry carries an explanation
    assert not report.ok  # unsafe entries fail the report


def test_glob_allow_is_flagged_unsafe():
    # Globs are flagged by the classifier itself, so this holds with or without a catalog.
    report = audit_allow_functions(["jax.*", "flax.linen.*"])
    assert _entry(report, "jax.*").verdict == "unsafe"
    assert _entry(report, "flax.linen.*").verdict == "unsafe"
    assert "glob" in _entry(report, "jax.*").reason


def test_pure_computation_is_safe():
    # Pure math is safe, including the most expressive ops (einsum, matmul, softmax).
    paths = [
        "jax.numpy.einsum",
        "jax.numpy.matmul",
        "jax.nn.softmax",
        "jax.numpy.where",
        "jax.numpy.sort",
        "jax.random.categorical",
        "flax.linen.Dense",
        "flax.linen.relu",
    ]
    report = audit_allow_functions(paths, catalog_dir=EXAMPLE_CATALOG)
    assert all(_entry(report, p).verdict == "safe" for p in paths)
    assert all(_entry(report, p).reason for p in paths)
    assert report.ok


def test_dual_use_is_narrow():
    # dual_use is reserved for a specific capability beyond pure computation: the host/device
    # boundary crossers, and flax's attribute-access knob.
    paths = ["jax.device_get", "jax.device_put", "flax.linen.Module"]
    report = audit_allow_functions(paths, catalog_dir=EXAMPLE_CATALOG)
    assert all(_entry(report, p).verdict == "dual_use" for p in paths)
    assert all(
        _entry(report, p).reason for p in paths
    )  # each carries its own concrete reason
    assert report.ok  # dual_use does not fail the report (allowed-but-flagged)


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


def test_orbax_is_flagged_unsafe_without_a_version_dir():
    # orbax has no version-keyable import root, so its blanket rule lives in _common and must fire
    # regardless of whether any orbax version is detected.
    report = audit_allow_functions(
        ["orbax.checkpoint.save"], catalog_dir=EXAMPLE_CATALOG
    )
    assert _entry(report, "orbax.checkpoint.save").verdict == "unsafe"


def test_flax_linen_common_surface_is_catalogued():
    # Spot-check the flax.linen coverage: pure compute is safe, only Module is dual_use, io is unsafe.
    expected = {
        "flax.linen.Dense": "safe",  # pure computation
        "flax.linen.MultiHeadDotProductAttention": "safe",
        "flax.linen.relu": "safe",  # activations are pure math -> safe
        "flax.linen.softmax": "safe",
        "flax.linen.scan": "safe",  # lifted transform over pure compute
        "flax.linen.make_causal_mask": "safe",  # structural mask helper
        "flax.linen.compact": "safe",  # decorator / machinery
        "flax.linen.initializers.lecun_normal": "safe",  # init, no data channel
        "flax.linen.Module": "dual_use",  # attribute-access knob
        "flax.io.read_file": "unsafe",  # flax.io.* glob -> disk IO
    }
    report = audit_allow_functions(list(expected), catalog_dir=EXAMPLE_CATALOG)
    got = {e.path: e.verdict for e in report.entries}
    assert got == expected
    assert all(_entry(report, p).reason for p in expected)  # every entry carries a note


def test_jax_common_surface_is_catalogued():
    # Spot-check the jax coverage: pure compute (transforms, math, samplers) is safe; only the
    # host/device boundary crossers are dual_use; IO/callbacks are unsafe.
    expected = {
        "jax.jit": "safe",  # transform over pure compute
        "jax.grad": "safe",
        "jax.numpy.matmul": "safe",  # pure compute
        "jax.numpy.sum": "safe",  # reduction
        "jax.nn.relu": "safe",  # activation
        "jax.numpy.linalg.svd": "safe",
        "jax.random.categorical": "safe",  # pure compute on logits
        "jax.numpy.zeros": "safe",  # constant creation
        "jax.random.split": "safe",  # key management
        "jax.nn.initializers.he_normal": "safe",  # init
        "jax.device_get": "dual_use",  # host/device boundary crossing
        "jax.device_put": "dual_use",
        "jax.numpy.save": "unsafe",  # disk IO (jax.numpy.save* rule)
        "jax.debug.print": "unsafe",  # host callback
    }
    report = audit_allow_functions(list(expected), catalog_dir=EXAMPLE_CATALOG)
    got = {e.path: e.verdict for e in report.entries}
    assert got == expected
    assert all(_entry(report, p).reason for p in expected)


def test_catalog_dir_supplies_the_rules(tmp_path):
    # Without a catalog_dir a path is 'review'; a catalog_dir is what provides its rules.
    assert (
        _entry(audit_allow_functions(["jax.numpy.einsum"]), "jax.numpy.einsum").verdict
        == "review"
    )
    version_dir = ".".join(jax.__version__.split(".")[:2])  # e.g. "0.11"
    ext = tmp_path / "jax" / version_dir
    ext.mkdir(parents=True)
    (ext / "catalog.json").write_text(
        json.dumps({"unsafe": {"jax.numpy.einsum": "custom rule"}})
    )
    report = audit_allow_functions(["jax.numpy.einsum"], catalog_dir=tmp_path)
    einsum = _entry(report, "jax.numpy.einsum")
    assert einsum.verdict == "unsafe"
    assert einsum.reason == "custom rule"


def test_malformed_catalog_degrades_to_review(tmp_path):
    # A broken catalog.json must not crash the advisory audit; its paths fall to review.
    version_dir = ".".join(jax.__version__.split(".")[:2])
    ext = tmp_path / "jax" / version_dir
    ext.mkdir(parents=True)
    (ext / "catalog.json").write_text("{ not valid json ")
    report = audit_allow_functions(["jax.numpy.einsum"], catalog_dir=tmp_path)
    assert _entry(report, "jax.numpy.einsum").verdict == "review"


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


def test_report_format_has_sections_and_ok_flag():
    report = audit_allow_functions(
        ["jax.profiler.start_server", "jax.device_get", "jax.numpy.einsum"],
        catalog_dir=EXAMPLE_CATALOG,
    )
    text = report.format()
    assert "UNSAFE" in text and "DUAL-USE" in text and "SAFE" in text
    assert "ok=False" in text  # the unsafe profiler entry fails the report


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
