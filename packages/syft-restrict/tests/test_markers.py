"""Tests for parse_markers() — the comment-based alternative to hand-counted line ranges."""

import pytest

from syft_restrict import MarkerError, parse_markers
from verify.helpers import normalize_source


def test_single_obfuscate_block():
    src = normalize_source("""
    import jax
    # syft-restrict: obfuscate-start
    def f(x):
        return x
    # syft-restrict: obfuscate-end
    """)
    obfuscate, hide = parse_markers(src)
    assert obfuscate == [(3, 4)]
    assert hide == []


def test_single_hide_block():
    src = normalize_source("""
    import jax
    # syft-restrict: hide-start
    SALT = 1
    SCALE = 2
    # syft-restrict: hide-end
    """)
    obfuscate, hide = parse_markers(src)
    assert obfuscate == []
    assert hide == [(3, 4)]


def test_single_line_trailing_markers():
    src = normalize_source("""
    MODEL_ID = 'gemma-2b'  # syft-restrict: obfuscate
    SALT = 1  # syft-restrict: hide
    """)
    obfuscate, hide = parse_markers(src)
    assert obfuscate == [(1, 1)]
    assert hide == [(2, 2)]


def test_multiple_disjoint_blocks_of_same_kind():
    src = normalize_source("""
    # syft-restrict: obfuscate-start
    a = 1
    # syft-restrict: obfuscate-end
    b = 2
    # syft-restrict: obfuscate-start
    c = 3
    # syft-restrict: obfuscate-end
    """)
    obfuscate, hide = parse_markers(src)
    assert obfuscate == [(2, 2), (6, 6)]
    assert hide == []


def test_obfuscate_and_hide_blocks_both_present_and_disjoint():
    src = normalize_source("""
    # syft-restrict: obfuscate-start
    a = 1
    # syft-restrict: obfuscate-end
    # syft-restrict: hide-start
    b = 2
    # syft-restrict: hide-end
    """)
    obfuscate, hide = parse_markers(src)
    assert obfuscate == [(2, 2)]
    assert hide == [(5, 5)]


def test_end_without_start_raises():
    src = normalize_source("""
    a = 1
    # syft-restrict: obfuscate-end
    """)
    with pytest.raises(MarkerError):
        parse_markers(src)


def test_start_without_end_raises():
    src = normalize_source("""
    # syft-restrict: obfuscate-start
    a = 1
    """)
    with pytest.raises(MarkerError):
        parse_markers(src)


def test_mismatched_end_kind_raises():
    src = normalize_source("""
    # syft-restrict: obfuscate-start
    a = 1
    # syft-restrict: hide-end
    """)
    with pytest.raises(MarkerError):
        parse_markers(src)


def test_hide_block_nested_inside_obfuscate_is_allowed():
    # Hide is strictly stronger than obfuscate, so carving a hide sub-region out of an open
    # obfuscate block is safe -- the surrounding obfuscate lines are unaffected.
    src = normalize_source("""
    # syft-restrict: obfuscate-start
    a = 1
    # syft-restrict: hide-start
    b = 2
    # syft-restrict: hide-end
    c = 3
    # syft-restrict: hide-start
    d = 4
    # syft-restrict: hide-end
    e = 5
    # syft-restrict: obfuscate-end
    """)
    obfuscate, hide = parse_markers(src)
    assert obfuscate == [(2, 2), (6, 6), (10, 10)]
    assert hide == [(4, 4), (8, 8)]


def test_single_line_hide_marker_nested_inside_obfuscate_is_allowed():
    src = normalize_source("""
    # syft-restrict: obfuscate-start
    a = 1  # syft-restrict: hide
    b = 2
    # syft-restrict: obfuscate-end
    """)
    obfuscate, hide = parse_markers(src)
    assert obfuscate == [(3, 3)]
    assert hide == [(2, 2)]


def test_obfuscate_block_nested_inside_hide_raises():
    # The reverse direction is not allowed: obfuscate is weaker than hide, so nesting it inside
    # an open hide block would loosen a region meant to be stricter.
    src = normalize_source("""
    # syft-restrict: hide-start
    a = 1
    # syft-restrict: obfuscate-start
    b = 2
    # syft-restrict: obfuscate-end
    # syft-restrict: hide-end
    """)
    with pytest.raises(MarkerError):
        parse_markers(src)


def test_obfuscate_block_nested_inside_obfuscate_raises():
    src = normalize_source("""
    # syft-restrict: obfuscate-start
    a = 1
    # syft-restrict: obfuscate-start
    b = 2
    # syft-restrict: obfuscate-end
    # syft-restrict: obfuscate-end
    """)
    with pytest.raises(MarkerError):
        parse_markers(src)


def test_hide_block_nested_inside_hide_raises():
    src = normalize_source("""
    # syft-restrict: hide-start
    a = 1
    # syft-restrict: hide-start
    b = 2
    # syft-restrict: hide-end
    # syft-restrict: hide-end
    """)
    with pytest.raises(MarkerError):
        parse_markers(src)


def test_single_line_obfuscate_marker_inside_open_obfuscate_block_raises():
    # Redundant same-kind marker -- not useful, so still rejected.
    src = normalize_source("""
    # syft-restrict: obfuscate-start
    a = 1  # syft-restrict: obfuscate
    # syft-restrict: obfuscate-end
    """)
    with pytest.raises(MarkerError):
        parse_markers(src)


def test_marker_inside_open_hide_block_raises():
    # A hide block cannot itself contain any nested marker, single-line or otherwise.
    src = normalize_source("""
    # syft-restrict: hide-start
    a = 1  # syft-restrict: hide
    # syft-restrict: hide-end
    """)
    with pytest.raises(MarkerError):
        parse_markers(src)


def test_empty_block_raises():
    src = normalize_source("""
    # syft-restrict: obfuscate-start
    # syft-restrict: obfuscate-end
    """)
    with pytest.raises(MarkerError):
        parse_markers(src)


def test_no_markers_at_all_raises():
    src = normalize_source("""
    import jax
    def f(x):
        return x
    """)
    with pytest.raises(MarkerError):
        parse_markers(src)


def test_marker_lookalike_inside_string_is_ignored():
    src = normalize_source("""
    NOTE = '# syft-restrict: obfuscate-start'
    # syft-restrict: obfuscate-start
    a = 1
    # syft-restrict: obfuscate-end
    """)
    obfuscate, hide = parse_markers(src)
    assert obfuscate == [(3, 3)]
    assert hide == []


def test_marker_lookalike_comment_prose_is_ignored():
    # A comment that merely mentions the word shouldn't match without the exact directive shape.
    src = normalize_source("""
    # this used to obfuscate something entirely differently
    # syft-restrict: obfuscate-start
    a = 1
    # syft-restrict: obfuscate-end
    """)
    obfuscate, hide = parse_markers(src)
    assert obfuscate == [(3, 3)]
    assert hide == []
