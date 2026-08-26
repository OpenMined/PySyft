"""Unit tests for the versioned-folder name parsing/filtering helpers.

These are pure functions -- no Drive mocks needed. They cover the path
that replaced the four format-specific parsers from the original PR.
"""

from syft.sync.connections.drive.gdrive_transport import (
    _extract_version_from_name,
    _filter_patch_compatible,
    _looks_like_version,
)


# ---------- _looks_like_version ---------------------------------------------


def test_looks_like_version_accepts_xyz():
    assert _looks_like_version("0.1.114") is True


def test_looks_like_version_rejects_non_digit():
    assert _looks_like_version("alice@example.com") is False


def test_looks_like_version_rejects_two_parts():
    assert _looks_like_version("0.1") is False


def test_looks_like_version_rejects_four_parts():
    assert _looks_like_version("0.1.2.3") is False


def test_looks_like_version_rejects_empty():
    assert _looks_like_version("") is False


# ---------- _extract_version_from_name --------------------------------------


def test_extract_from_personal_format():
    assert _extract_version_from_name("0.1.114#alice@example.com") == "0.1.114"


def test_extract_from_p2p_format():
    name = "syft_datasite#0.1.114#alice@example.com#inbox#bob@example.com"
    assert _extract_version_from_name(name) == "0.1.114"


def test_extract_from_checkpoints_format():
    assert (
        _extract_version_from_name("alice@example.com-0.1.114-checkpoints") == "0.1.114"
    )


def test_extract_from_rolling_state_format():
    assert (
        _extract_version_from_name("alice@example.com-0.1.114-rolling-state")
        == "0.1.114"
    )


def test_extract_returns_none_when_missing():
    assert _extract_version_from_name("just_a_folder_name") is None


# ---------- _filter_patch_compatible ----------------------------------------


def test_filter_keeps_same_patch():
    folders = [("id1", "0.1.114#alice@example.com")]
    assert _filter_patch_compatible(folders, current_version="0.1.114") == folders


def test_filter_keeps_different_patch_same_minor():
    folders = [("id1", "0.1.114#alice@example.com")]
    assert _filter_patch_compatible(folders, current_version="0.1.200") == folders


def test_filter_drops_minor_diff():
    folders = [
        ("id1", "0.1.114#alice@example.com"),
        ("id2", "0.2.0#alice@example.com"),
    ]
    assert _filter_patch_compatible(folders, current_version="0.1.114") == [
        ("id1", "0.1.114#alice@example.com")
    ]


def test_filter_drops_major_diff():
    folders = [
        ("id1", "0.1.114#alice@example.com"),
        ("id2", "1.0.0#alice@example.com"),
    ]
    assert _filter_patch_compatible(folders, current_version="0.1.114") == [
        ("id1", "0.1.114#alice@example.com")
    ]


def test_filter_drops_names_without_a_version():
    folders = [
        ("id1", "0.1.114#alice@example.com"),
        ("id2", "no_version_here"),
    ]
    assert _filter_patch_compatible(folders, current_version="0.1.114") == [
        ("id1", "0.1.114#alice@example.com")
    ]


def test_filter_covers_all_four_folder_formats():
    """All four formats syft uses should match when major.minor align."""
    folders = [
        ("id1", "0.1.114#alice@example.com"),
        ("id2", "syft_datasite#0.1.115#alice@example.com#inbox#bob@example.com"),
        ("id3", "alice@example.com-0.1.116-checkpoints"),
        ("id4", "alice@example.com-0.1.117-rolling-state"),
    ]
    kept = _filter_patch_compatible(folders, current_version="0.1.200")
    assert {fid for fid, _ in kept} == {"id1", "id2", "id3", "id4"}


def test_filter_returns_empty_for_bad_current_version():
    folders = [("id1", "0.1.114#alice@example.com")]
    assert _filter_patch_compatible(folders, current_version="garbage") == []


# ---------- two-digit minor (0.10.x, the first `syft` release line) ---------


def test_looks_like_version_accepts_two_digit_minor():
    assert _looks_like_version("0.10.0") is True


def test_extract_two_digit_minor_from_all_formats():
    assert _extract_version_from_name("0.10.0#alice@example.com") == "0.10.0"
    assert (
        _extract_version_from_name(
            "syft_datasite#0.10.0#alice@example.com#inbox#bob@example.com"
        )
        == "0.10.0"
    )
    assert (
        _extract_version_from_name("alice@example.com-0.10.0-checkpoints") == "0.10.0"
    )


def test_filter_two_digit_minor_is_numeric_not_lexicographic():
    """0.10.x must match 0.10.y and reject both 0.1.x and 0.9.x."""
    folders = [
        ("id1", "0.10.0#alice@example.com"),
        ("id2", "0.10.3#alice@example.com"),
        ("id3", "0.1.117#alice@example.com"),
        ("id4", "0.9.5#alice@example.com"),
    ]
    kept = _filter_patch_compatible(folders, current_version="0.10.1")
    assert {fid for fid, _ in kept} == {"id1", "id2"}
