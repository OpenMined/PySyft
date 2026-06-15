"""Tests for version mismatch check and delete_syftbox utilities."""

from pathlib import Path
from unittest.mock import patch

from syft_client.sync.connections.drive.gdrive_transport import (
    GDRIVE_P2P_FOLDER_DATASITE_PREFIX,
    SYFT_PEERS_FILE,
    SYFT_VERSION_FILE,
)
from syft_client.sync.login_utils import handle_potential_version_mismatches_on_login
from syft_client.sync.syftbox_manager import SyftboxManager
from syft_client.sync.version.version_info import VersionInfo
from syft_datasets.dataset_manager import (
    DATASET_COLLECTION_PREFIX,
    PRIVATE_DATASET_COLLECTION_PREFIX,
)
from tests.unit.utils import create_tmp_dataset_files


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
    @patch("syft_client.sync.login_utils.delete_remote_syftbox")
    @patch("syft_client.sync.login_utils.delete_local_syftbox")
    @patch("syft_client.sync.login_utils._prompt_mismatch", return_value="2")
    @patch("syft_client.sync.login_utils._read_remote_version")
    @patch("syft_client.sync.login_utils.read_local_version")
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

    @patch("syft_client.sync.login_utils._delete_remote_unversioned_state")
    @patch("syft_client.sync.login_utils.delete_remote_syftbox")
    @patch("syft_client.sync.login_utils.delete_local_syftbox")
    @patch("syft_client.sync.login_utils._prompt_mismatch", return_value="1")
    @patch("syft_client.sync.login_utils._read_remote_version")
    @patch("syft_client.sync.login_utils.read_local_version")
    def test_upgrade_deletes_local_only(
        self,
        mock_read_local,
        mock_read_remote,
        mock_prompt,
        mock_delete_local,
        mock_delete_remote,
        mock_delete_unversioned,
    ):
        """Mismatch + choice 1 (upgrade) → local deleted, unversioned state deleted, full remote preserved."""
        mock_read_local.return_value = _old_version_info()
        mock_read_remote.return_value = _old_version_info()

        handle_potential_version_mismatches_on_login(EMAIL, TOKEN_PATH)

        mock_delete_local.assert_called_once()
        mock_delete_remote.assert_not_called()
        mock_delete_unversioned.assert_called_once()

    @patch("syft_client.sync.login_utils._read_remote_version")
    @patch("syft_client.sync.login_utils.read_local_version")
    def test_no_mismatch_no_prompt(self, mock_read_local, mock_read_remote):
        """Both versions match installed → no prompt."""
        mock_read_local.return_value = VersionInfo.current()
        mock_read_remote.return_value = VersionInfo.current()

        handle_potential_version_mismatches_on_login(EMAIL, TOKEN_PATH)


def _query_files(connection, name_contains):
    """Query mock drive for files/folders whose name contains a substring."""
    q = f"name contains '{name_contains}' and trashed=false"
    results = (
        connection.drive_service.files().list(q=q, fields="files(id, name)").execute()
    )
    return results.get("files", [])


def test_delete_unversioned_state_removes_correct_folders():
    """delete_unversioned_state removes exactly the right artifacts from mock drive."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
        encryption=True,
    )

    # Create dataset so collection folders exist
    mock_path, private_path, readme_path = create_tmp_dataset_files()
    do_manager.create_dataset(
        name="my dataset",
        mock_path=mock_path,
        private_path=private_path,
        summary="Test",
        readme_path=readme_path,
        users=[ds_manager.email],
        upload_private=True,
    )
    do_manager.sync()

    do_conn = do_manager.peer_manager.connection_router.connections[0]
    do_email = do_manager.email

    # Assert artifacts exist before deletion
    do_enc_bundles = f"syft_encryption_bundles#{do_email}"
    assert len(_query_files(do_conn, do_enc_bundles)) > 0
    assert len(_query_files(do_conn, DATASET_COLLECTION_PREFIX)) > 0
    assert len(_query_files(do_conn, PRIVATE_DATASET_COLLECTION_PREFIX)) > 0
    assert len(_query_files(do_conn, SYFT_PEERS_FILE)) > 0
    assert len(_query_files(do_conn, SYFT_VERSION_FILE)) > 0

    # Assert versioned folders exist
    p2p_before = _query_files(do_conn, GDRIVE_P2P_FOLDER_DATASITE_PREFIX)
    assert len(p2p_before) > 0

    # Delete unversioned state
    do_conn.delete_unversioned_state()

    # Assert unversioned artifacts are gone
    assert len(_query_files(do_conn, do_enc_bundles)) == 0
    assert len(_query_files(do_conn, DATASET_COLLECTION_PREFIX)) == 0
    assert len(_query_files(do_conn, PRIVATE_DATASET_COLLECTION_PREFIX)) == 0
    # peers/version files: DO's are gone, DS's may still exist
    do_peers = [
        f
        for f in _query_files(do_conn, SYFT_PEERS_FILE)
        if f["id"] == do_conn._get_peers_file_id()
    ]
    assert len(do_peers) == 0
    do_version = [
        f
        for f in _query_files(do_conn, SYFT_VERSION_FILE)
        if f["id"] == do_conn._get_version_file_id()
    ]
    assert len(do_version) == 0

    # Assert versioned folders survive
    p2p_after = _query_files(do_conn, GDRIVE_P2P_FOLDER_DATASITE_PREFIX)
    assert len(p2p_after) == len(p2p_before)


class TestDeleteSyftboxImport:
    def test_importable_from_top_level(self):
        from syft_client import (
            delete_syftbox,
            delete_local_syftbox,
            delete_remote_syftbox,
        )

        assert callable(delete_syftbox)
        assert callable(delete_local_syftbox)
        assert callable(delete_remote_syftbox)
