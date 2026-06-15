"""End-to-end test for version mismatch and backup flow with mock drive."""

from unittest.mock import patch

from syft_client.sync.utils.syftbox_utils import delete_local_syftbox
from syft_client.sync.connections.drive.gdrive_transport import (
    GDRIVE_P2P_FOLDER_DATASITE_PREFIX,
    GOOGLE_FOLDER_MIME_TYPE,
    GDriveConnection,
)
from syft_client.sync.connections.drive.mock_drive_service import (
    MockDriveService,
)
from syft_client.sync.syftbox_manager import SyftboxManager, SyftboxManagerConfig
from syft_client.version import SYFT_CLIENT_VERSION

from tests.unit.utils import create_test_project_folder, create_tmp_dataset_files

NEW_VERSION = "99.0.0"


def _find_p2p_folders(connection, peer_email):
    """Find all P2P folders involving a specific peer on the mock drive."""
    q = (
        f"mimeType='{GOOGLE_FOLDER_MIME_TYPE}'"
        f" and name contains '{GDRIVE_P2P_FOLDER_DATASITE_PREFIX}'"
        f" and name contains '{peer_email}'"
        " and trashed=false"
    )
    results = (
        connection.drive_service.files().list(q=q, fields="files(id, name)").execute()
    )
    return results.get("files", [])


def _find_versioned_p2p_folders(connection, peer_email, version):
    """Find P2P folders for a specific peer and version."""
    folders = _find_p2p_folders(connection, peer_email)
    return [f for f in folders if f"#{version}#" in f["name"]]


def _get_backing_store(manager):
    """Extract the mock backing store from a manager's connection."""
    conn = manager.peer_manager.connection_router.connections[0]
    return conn.drive_service._backing_store


def _reinitialize_manager(email, backing_store, has_do_role, has_ds_role):
    """Create a new SyftboxManager connected to an existing mock backing store.

    This mirrors what pair_with_mock_drive_service_connection does for a
    single manager, reusing the same backing store so the new manager sees
    the same GDrive state.
    """
    config = SyftboxManagerConfig._base_config_for_testing(
        email=email,
        has_do_role=has_do_role,
        has_ds_role=has_ds_role,
        use_in_memory_cache=False,
    )
    manager = SyftboxManager.from_config(config)

    mock_service = MockDriveService(backing_store, email)
    conn = GDriveConnection.from_service(email, mock_service)
    manager._add_connection(conn)

    if has_ds_role:
        manager.file_writer.add_callback(
            "write_file",
            manager.datasite_watcher_syncer.on_file_change,
        )
    if has_do_role:
        manager.datasite_owner_syncer.event_cache.add_callback(
            "on_event_local_write",
            manager.job_file_change_handler._handle_file_change,
        )

    manager.peer_manager.write_own_version()
    return manager


def _simulate_upgrade(manager, backing_store):
    """Simulate handle_potential_version_mismatches_on_login with mocks.

    Patches only the I/O boundaries so that read_local_version reads from the
    manager's real syftbox folder, _read_remote_version reads from the mock
    drive, and delete operations target the correct local path / mock drive.
    """
    email = manager.email
    syftbox_folder = manager.syftbox_folder

    mock_service = MockDriveService(backing_store, email)
    mock_conn = GDriveConnection.from_service(email, mock_service)

    def read_remote(e, t):
        return mock_conn.read_own_version_file()

    def do_delete_local(**kwargs):
        delete_local_syftbox(email=email, local_syftbox_path=syftbox_folder)

    def do_delete_unversioned(e, t):
        mock_conn.delete_unversioned_state()

    with (
        patch(
            "syft_client.sync.login_utils._resolve_token_path",
            return_value=None,
        ),
        patch(
            "syft_client.sync.login_utils._get_default_syftbox_path",
            return_value=syftbox_folder,
        ),
        patch(
            "syft_client.sync.login_utils._read_remote_version",
            side_effect=read_remote,
        ),
        patch(
            "syft_client.sync.login_utils._prompt_mismatch",
            return_value="1",
        ),
        patch(
            "syft_client.sync.login_utils.delete_local_syftbox",
            side_effect=do_delete_local,
        ),
        patch(
            "syft_client.sync.login_utils._delete_remote_unversioned_state",
            side_effect=do_delete_unversioned,
        ),
    ):
        from syft_client.sync.login_utils import (
            handle_potential_version_mismatches_on_login,
        )

        handle_potential_version_mismatches_on_login(email)


def test_version_mismatch_and_backup_flow():
    """Full flow: create state on v1 -> upgrade to v2 -> old state preserved, new version works."""

    # -- Step 1: Create DO/DS on current version --
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )

    # -- Step 2: DO creates dataset --
    mock_path, private_path, readme_path = create_tmp_dataset_files()
    do_manager.create_dataset(
        name="my dataset",
        mock_path=mock_path,
        private_path=private_path,
        summary="Test dataset",
        readme_path=readme_path,
        users=[ds_manager.email],
    )
    do_manager.sync()
    ds_manager.sync()

    # -- Step 3: DS submits job, DO runs it --
    project_dir = create_test_project_folder(with_pyproject=False)
    ds_manager.submit_python_job(
        user=do_manager.email,
        code_path=str(project_dir),
        job_name="pre_upgrade.job",
        entrypoint="main.py",
    )
    do_manager.sync()
    assert len(do_manager.jobs) == 1
    do_manager.jobs[0].approve()
    do_manager.process_approved_jobs()
    do_manager.sync()
    ds_manager.sync()

    assert do_manager.jobs[0].status == "done"

    # -- Step 4: Assert only P2P folders with current version --
    do_conn = do_manager.peer_manager.connection_router.connections[0]
    do_p2p_current = _find_versioned_p2p_folders(
        do_conn, ds_manager.email, SYFT_CLIENT_VERSION
    )
    assert len(do_p2p_current) > 0
    # No folders with a different version
    all_do_p2p = _find_p2p_folders(do_conn, ds_manager.email)
    assert len(all_do_p2p) == len(do_p2p_current)

    # -- Step 5: Extract backing store --
    backing_store = _get_backing_store(do_manager)
    do_email = do_manager.email
    ds_email = ds_manager.email

    # -- Step 6+7: Upgrade DO --
    with (
        patch("syft_client.version.SYFT_CLIENT_VERSION", NEW_VERSION),
        patch(
            "syft_client.sync.connections.drive.gdrive_transport.SYFT_CLIENT_VERSION",
            NEW_VERSION,
        ),
        patch("syft_client.sync.login_utils.SYFT_CLIENT_VERSION", NEW_VERSION),
        patch("syft_client.sync.version.version_info.SYFT_CLIENT_VERSION", NEW_VERSION),
    ):
        _simulate_upgrade(do_manager, backing_store)
        do_manager = _reinitialize_manager(
            do_email, backing_store, has_do_role=True, has_ds_role=False
        )

        # -- Step 8: Assert new versioned folders for DO --
        do_conn_new = do_manager.peer_manager.connection_router.connections[0]
        do_p2p_new = _find_versioned_p2p_folders(do_conn_new, ds_email, NEW_VERSION)
        # New folders don't exist yet (no peers added), but personal folder does
        personal_folder_name = f"{NEW_VERSION}#{do_email}"
        personal_id = do_conn_new._find_folder_by_name(
            personal_folder_name,
            parent_id=do_conn_new.get_syftbox_folder_id(),
            owner_email=do_email,
        )
        assert personal_id is not None

        # -- Step 9+10: Upgrade DS --
        _simulate_upgrade(ds_manager, backing_store)
        ds_manager = _reinitialize_manager(
            ds_email, backing_store, has_do_role=False, has_ds_role=True
        )

        # -- Step 11: Assert new versioned folders for DS --
        ds_conn_new = ds_manager.peer_manager.connection_router.connections[0]
        ds_personal_name = f"{NEW_VERSION}#{ds_email}"
        ds_personal_id = ds_conn_new._find_folder_by_name(
            ds_personal_name,
            parent_id=ds_conn_new.get_syftbox_folder_id(),
            owner_email=ds_email,
        )
        assert ds_personal_id is not None

        # -- Step 12: Assert peer connection is gone --
        assert len(do_manager.peer_manager.approved_peers) == 0
        assert len(ds_manager.peer_manager.approved_peers) == 0

        # -- Step 13: Re-add peers --
        ds_manager.add_peer(do_manager.email)
        do_manager.load_peers()
        do_manager.approve_peer_request(ds_manager.email)

        # Now new versioned P2P folders should exist
        do_p2p_new = _find_versioned_p2p_folders(do_conn_new, ds_email, NEW_VERSION)
        assert len(do_p2p_new) > 0

        ds_p2p_new = _find_versioned_p2p_folders(ds_conn_new, do_email, NEW_VERSION)
        assert len(ds_p2p_new) > 0

        # -- Step 14: Re-upload dataset --
        mock_path2, private_path2, readme_path2 = create_tmp_dataset_files()
        do_manager.create_dataset(
            name="my dataset",
            mock_path=mock_path2,
            private_path=private_path2,
            summary="Test dataset v2",
            readme_path=readme_path2,
            users=[ds_manager.email],
        )
        do_manager.sync()
        ds_manager.sync()

        # -- Step 15: Re-submit job --
        project_dir2 = create_test_project_folder(with_pyproject=False)
        ds_manager.submit_python_job(
            user=do_manager.email,
            code_path=str(project_dir2),
            job_name="post_upgrade.job",
            entrypoint="main.py",
        )
        do_manager.sync()

        # -- Step 16: Assert only one job (new one), old folder still has old --
        assert len(do_manager.jobs) == 1

        # Old versioned P2P folders still have old data
        old_do_p2p = _find_versioned_p2p_folders(
            do_conn_new, ds_email, SYFT_CLIENT_VERSION
        )
        assert len(old_do_p2p) > 0

        # -- Step 17: DO runs new job --
        do_manager.jobs[0].approve()
        do_manager.process_approved_jobs()
        do_manager.sync()

        assert do_manager.jobs[0].status == "done"

        # -- Step 18: DS sees result --
        ds_manager.sync()
        ds_jobs = ds_manager.job_client.jobs
        assert len(ds_jobs) == 1
        # DS should have received the output file via sync
        ds_job = ds_jobs[0]
        assert ds_job.status == "done"
