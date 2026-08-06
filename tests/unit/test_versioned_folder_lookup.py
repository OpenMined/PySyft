"""Unit tests for the versioned-folder name parsing/filtering helpers.

These are pure functions -- no Drive mocks needed. They cover the path
that replaced the four format-specific parsers from the original PR.

Ordering and selection now live in _partition_by_version (adopt, private
folders) and _sorted_by_version (P2P lookup), each tested separately.
"""

from syft_client.sync.connections.drive.gdrive_transport import (
    _extract_version_from_name,
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
