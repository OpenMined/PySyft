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


def test_pure_computation_is_safe():
    # Pure math is safe, including the most expressive ops (einsum, matmul, softmax).
    paths = [
        "jax.numpy.einsum", "jax.numpy.matmul", "jax.nn.softmax", "jax.numpy.where",
        "jax.numpy.sort", "jax.random.categorical", "flax.linen.Dense", "flax.linen.relu",
    ]
    report = audit_allow_functions(paths)
    assert all(_entry(report, p).verdict == "safe" for p in paths)
    assert all(_entry(report, p).reason for p in paths)
    assert report.ok


def test_dual_use_is_narrow():
    # dual_use is reserved for a specific capability beyond pure computation: the host/device
    # boundary crossers, and flax's attribute-access knob.
    paths = ["jax.device_get", "jax.device_put", "flax.linen.Module"]
    report = audit_allow_functions(paths)
    assert all(_entry(report, p).verdict == "dual_use" for p in paths)
    assert all(_entry(report, p).reason for p in paths)  # each carries its own concrete reason
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
    report = audit_allow_functions(list(expected))
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
    report = audit_allow_functions(list(expected))
    got = {e.path: e.verdict for e in report.entries}
    assert got == expected
    assert all(_entry(report, p).reason for p in expected)


def test_report_format_has_sections_and_ok_flag():
    report = audit_allow_functions(
        ["jax.profiler.start_server", "jax.device_get", "jax.numpy.einsum"]
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
