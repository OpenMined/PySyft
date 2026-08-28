"""Tests for version mismatch check and delete_syftbox utilities."""

from pathlib import Path
from unittest.mock import patch

from syft.sync.login_utils import handle_potential_version_mismatches_on_login
from syft.sync.version.version_info import VersionInfo


EMAIL = "test@example.com"
TOKEN_PATH = Path("/fake/token.json")


def _old_version_info() -> VersionInfo:
    return VersionInfo(
        syft_client_version="0.0.1",
        min_supported_syft_client_version="0.0.1",
        protocol_version="1.0.0",
        min_supported_protocol_version="1.0.0",
    )


class TestVersionMismatchCheck:
    @patch("syft.sync.login_utils.delete_remote_syftbox")
    @patch("syft.sync.login_utils.delete_local_syftbox")
    @patch("syft.sync.login_utils._prompt_mismatch", return_value="2")
    @patch("syft.sync.login_utils._read_remote_version")
    @patch("syft.sync.login_utils.read_local_version")
    def test_delete_all(
        self,
        mock_read_local,
        mock_read_remote,
        mock_prompt,
        mock_delete_local,
        mock_delete_remote,
    ):
        """Mismatch + choice 2 (delete all) → local + remote deleted."""
        mock_read_local.return_value = _old_version_info()
        mock_read_remote.return_value = _old_version_info()

        handle_potential_version_mismatches_on_login(EMAIL, TOKEN_PATH)

        mock_delete_local.assert_called_once()
        mock_delete_remote.assert_called_once()

    @patch("syft.sync.login_utils.delete_remote_syftbox")
    @patch("syft.sync.login_utils.delete_local_syftbox")
    @patch("syft.sync.login_utils._prompt_mismatch", return_value="1")
    @patch("syft.sync.login_utils._read_remote_version")
    @patch("syft.sync.login_utils.read_local_version")
    def test_continue_keeps_local_and_remote(
        self,
        mock_read_local,
        mock_read_remote,
        mock_prompt,
        mock_delete_local,
        mock_delete_remote,
    ):
        """Mismatch + choice 1 (continue) → no deletes; data is kept for repair."""
        mock_read_local.return_value = _old_version_info()
        mock_read_remote.return_value = _old_version_info()

        handle_potential_version_mismatches_on_login(EMAIL, TOKEN_PATH)

        mock_delete_local.assert_not_called()
        mock_delete_remote.assert_not_called()

    @patch("syft.sync.login_utils.sys.exit")
    @patch("syft.sync.login_utils.delete_remote_syftbox")
    @patch("syft.sync.login_utils.delete_local_syftbox")
    @patch("syft.sync.login_utils._prompt_mismatch", return_value="3")
    @patch("syft.sync.login_utils._read_remote_version")
    @patch("syft.sync.login_utils.read_local_version")
    def test_quit_exits_without_delete(
        self,
        mock_read_local,
        mock_read_remote,
        mock_prompt,
        mock_delete_local,
        mock_delete_remote,
        mock_exit,
    ):
        """Mismatch + choice 3 (quit) → exit, no deletes."""
        mock_read_local.return_value = _old_version_info()
        mock_read_remote.return_value = _old_version_info()
        mock_exit.side_effect = SystemExit(0)

        try:
            handle_potential_version_mismatches_on_login(EMAIL, TOKEN_PATH)
        except SystemExit:
            pass

        mock_delete_local.assert_not_called()
        mock_delete_remote.assert_not_called()
        mock_exit.assert_called_once_with(0)

    @patch("syft.sync.login_utils._read_remote_version")
    @patch("syft.sync.login_utils.read_local_version")
    def test_no_mismatch_no_prompt(self, mock_read_local, mock_read_remote):
        """Both versions match installed → no prompt."""
        mock_read_local.return_value = VersionInfo.current()
        mock_read_remote.return_value = VersionInfo.current()

        handle_potential_version_mismatches_on_login(EMAIL, TOKEN_PATH)


class TestPromptWithoutATerminal:
    """A notebook or a scheduled run has no terminal to answer the prompt."""

    @patch("syft.sync.login_utils.sys.stdin")
    def test_no_terminal_keeps_data_and_continues(self, mock_stdin):
        # Choice 1 keeps every file and changes nothing, so it is safe to take
        # without an answer. A prompt would stop the run instead.
        from syft.sync.login_utils import _prompt_mismatch

        mock_stdin.isatty.return_value = False
        with patch("builtins.input", side_effect=AssertionError("must not prompt")):
            assert _prompt_mismatch(_old_version_info(), _old_version_info()) == "1"

    @patch("syft.sync.login_utils.sys.stdin")
    def test_a_terminal_still_asks(self, mock_stdin):
        from syft.sync.login_utils import _prompt_mismatch

        mock_stdin.isatty.return_value = True
        with patch("builtins.input", return_value="2"):
            assert _prompt_mismatch(_old_version_info(), _old_version_info()) == "2"


def _query_files(connection, name_contains):
    """Query mock drive for files/folders whose name contains a substring."""
    q = f"name contains '{name_contains}' and trashed=false"
    results = (
        connection.drive_service.files().list(q=q, fields="files(id, name)").execute()
    )
    return results.get("files", [])


class TestDeleteSyftboxImport:
    def test_importable_from_top_level(self):
        from syft import (
            delete_syftbox,
            delete_local_syftbox,
            delete_remote_syftbox,
        )

        assert callable(delete_syftbox)
        assert callable(delete_local_syftbox)
        assert callable(delete_remote_syftbox)
