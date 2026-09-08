"""Unit tests for the versioned-folder name parsing/filtering helpers.

These are pure functions -- no Drive mocks needed. They cover the path
that replaced the four format-specific parsers from the original PR.

Ordering and selection now live in _partition_by_version (adopt, private
folders) and _sorted_by_version (P2P lookup), each tested separately.
"""

from syft.sync.connections.drive.gdrive_transport import (
    _extract_version_from_name,
    _looks_like_version,
    _partition_by_version,
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


def test_partition_two_digit_minor_is_numeric_not_lexicographic():
    """0.10.x must match 0.10.y and reject both 0.1.x and 0.9.x."""
    folders = [
        ("id1", "0.10.0#alice@example.com"),
        ("id2", "0.10.3#alice@example.com"),
        ("id3", "0.1.117#alice@example.com"),
        ("id4", "0.9.5#alice@example.com"),
    ]
    compatible, older, _ = _partition_by_version(folders, current_version="0.10.1")
    assert {fid for fid, _ in compatible} == {"id1", "id2"}
    assert {fid for fid, _ in older} == {"id3", "id4"}
