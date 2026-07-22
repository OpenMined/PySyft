"""Tests for the advisory allow-list audit (syft_restrict.audit)."""

from syft_restrict import AuditReport, audit_allow_functions
from syft_restrict.audit import _best_version_key


def _entry(report: AuditReport, path: str):
    return next(e for e in report.entries if e.path == path)


def test_known_unsafe_paths_are_flagged():
    paths = [
        "jax.profiler.start_server",  # network server / disk trace (the floor-gap we found)
        "jax.distributed.initialize",  # outbound network
        "jax.numpy.save",  # disk
        "jax.experimental.io_callback",  # host callback
        "flax.training.checkpoints.save_checkpoint",  # disk
    ]
    report = audit_allow_functions(paths)
    assert all(_entry(report, p).verdict == "unsafe" for p in paths)
    assert all(_entry(report, p).reason for p in paths)  # every entry carries an explanation
    assert not report.ok  # unsafe entries fail the report


def test_glob_allow_is_flagged_unsafe():
    report = audit_allow_functions(["jax.*", "flax.linen.*"])
    assert _entry(report, "jax.*").verdict == "unsafe"
    assert _entry(report, "flax.linen.*").verdict == "unsafe"
    assert "glob" in _entry(report, "jax.*").reason


def test_genuinely_inert_paths_are_safe():
    # constants, masks, and module refs have no residual channel of their own.
    paths = ["jax.numpy.arange", "jax.numpy.ones", "jax.numpy.tril", "jax.lax"]
    report = audit_allow_functions(paths)
    assert all(_entry(report, p).verdict == "safe" for p in paths)
    assert all(_entry(report, p).reason for p in paths)  # safe entries still explained
    assert report.ok  # only-safe list passes


def test_dual_use_paths_are_flagged_between_safe_and_unsafe():
    # useful ops that are mostly safe but abusable in combination land in dual_use, not safe.
    paths = ["jax.numpy.einsum", "jax.nn.softmax", "jax.numpy.where", "flax.linen.Module"]
    report = audit_allow_functions(paths)
    assert all(_entry(report, p).verdict == "dual_use" for p in paths)
    assert all(_entry(report, p).reason for p in paths)  # dual_use entries carry a terse note
    # the category carries the caution, so the note stays vague -- no abuse how-to / recipe wording.
    einsum_reason = _entry(report, "jax.numpy.einsum").reason
    assert "encode" not in einsum_reason and "secret" not in einsum_reason
    assert report.ok  # dual_use does not fail the report (allowed-but-flagged)


def test_uncatalogued_path_is_deferred_to_review_without_assumptions():
    # An unknown path is neither safe nor unsafe: the audit makes no guess, it defers to a human.
    # This holds regardless of whether the path is importable — no source inspection happens.
    report = audit_allow_functions(["totally.made.up.symbol", "shutil.copyfile"])
    for path in ("totally.made.up.symbol", "shutil.copyfile"):
        e = _entry(report, path)
        assert e.verdict == "review"
        assert "catalog" in e.reason  # reported as uncatalogued, deferred to human review
    assert report.ok  # review entries do not fail the report; they need a human


def test_cross_library_pattern_matches_any_library():
    # `*.io_callback` lives in the library-agnostic _common catalog
    report = audit_allow_functions(["somelib.io_callback"])
    assert _entry(report, "somelib.io_callback").verdict == "unsafe"


def test_orbax_is_flagged_unsafe_without_a_version_dir():
    # orbax has no version-keyable import root, so its blanket rule lives in _common and must fire
    # regardless of whether any orbax version is detected.
    report = audit_allow_functions(["orbax.checkpoint.save"])
    assert _entry(report, "orbax.checkpoint.save").verdict == "unsafe"


def test_report_format_has_sections_and_ok_flag():
    report = audit_allow_functions(
        ["jax.profiler.start_server", "jax.numpy.einsum", "jax.numpy.ones"]
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
    assert _best_version_key(keys, "0.19.2") is None  # no baseline; uncovered -> no rules
    assert _best_version_key(keys, "0.2.0") is None
    assert _best_version_key(keys, "") is None  # unknown/undetected version matches nothing
