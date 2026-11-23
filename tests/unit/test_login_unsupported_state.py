"""A login on a SyftBox of a release whose formats this one does not read.

The login stops with a way to start fresh, instead of failing later on the
old data.
"""

from pathlib import Path

import pytest
from syft.sync import login_utils
from syft.sync.version.exceptions import UnreadablePeerInfoError
from syft.version import VERSION_FILE_NAME
from tests.unit.utils import EARLIER_RELEASE_FILE


@pytest.fixture
def login_without_drive(monkeypatch, tmp_path):
    """Login helpers pointed at a local folder, with no Drive and no terminal."""
    syftbox = tmp_path / "SyftBox_do@test.org"
    monkeypatch.setattr(login_utils, "_get_default_syftbox_path", lambda email: syftbox)
    monkeypatch.setattr(login_utils, "_resolve_token_path", lambda token_path: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    return syftbox


def test_login_stops_on_unsupported_drive_syftbox(monkeypatch, login_without_drive):
    def unreadable(email: str, token_path: Path | None):
        raise UnreadablePeerInfoError("Could not read the own peer info file")

    monkeypatch.setattr(login_utils, "_read_remote_version", unreadable)

    with pytest.raises(UnreadablePeerInfoError, match="start fresh"):
        login_utils.handle_potential_version_mismatches_on_login("do@test.org")


def test_login_stops_on_unsupported_local_syftbox(monkeypatch, login_without_drive):
    monkeypatch.setattr(login_utils, "_read_remote_version", lambda e, t: None)
    login_without_drive.mkdir(parents=True)
    (login_without_drive / VERSION_FILE_NAME).write_text(EARLIER_RELEASE_FILE.decode())

    with pytest.raises(UnreadablePeerInfoError, match="start fresh"):
        login_utils.handle_potential_version_mismatches_on_login("do@test.org")
