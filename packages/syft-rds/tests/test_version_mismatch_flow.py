"""End-to-end test for a client minor upgrade that keeps local and remote data.

Login no longer deletes SyftBox state on a major/minor mismatch. The default is
to continue; private Drive folders are adopted by rename, and P2P folders of the
earlier version are reused so a peer that has not upgraded still finds them.
"""

from unittest.mock import patch

from syft.sync.connections.drive.gdrive_transport import (
    GDRIVE_P2P_FOLDER_DATASITE_PREFIX,
    GOOGLE_FOLDER_MIME_TYPE,
    GDriveConnection,
)
from syft.sync.connections.drive.mock_drive_service import (
    MockDriveService,
)
from syft_rds import SyftRDSClient
from syft_rds.config import SyftRDSClientConfig
from syft.version import SYFT_VERSION

from dataset_test_utils import create_test_project_folder, create_tmp_dataset_files

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


def _reinitialize_manager(
    email, backing_store, has_do_role, has_ds_role, syftbox_folder, write_version=True
):
    """Create a new SyftRDSClient on the same local path and mock Drive store.

    Reuses the local SyftBox directory so a continue-on-mismatch upgrade keeps
    the data that login left in place. Reuses the backing store so GDrive state
    matches the pre-upgrade client.
    """
    config = SyftRDSClientConfig._base_config_for_testing(
        email=email,
        has_do_role=has_do_role,
        has_ds_role=has_ds_role,
        use_in_memory_cache=False,
        syftbox_folder=syftbox_folder,
    )
    manager = SyftRDSClient.from_config(config)

    mock_service = MockDriveService(backing_store, email)
    conn = GDriveConnection.from_service(email, mock_service)
    manager.sync_engine._add_connection(conn)

    if has_ds_role:
        manager.sync_engine.file_writer.add_callback(
            "write_file",
            manager.sync_engine.datasite_watcher_syncer.on_file_change,
        )
    if has_do_role:
        manager.sync_engine.datasite_owner_syncer.event_cache.add_callback(
            "on_event_local_write",
            manager.sync_engine.job_file_change_handler._handle_file_change,
        )

    if write_version:
        manager.peer_manager.write_own_version()
    return manager


def _simulate_continue_on_mismatch(manager, backing_store):
    """Run the login mismatch handler with choice 1 (continue, keep data)."""
    email = manager.email
    syftbox_folder = manager.syftbox_folder

    mock_service = MockDriveService(backing_store, email)
    mock_conn = GDriveConnection.from_service(email, mock_service)

    def read_remote(e, t):
        return mock_conn.read_own_version_file()

    with (
        patch(
            "syft.sync.login_utils._resolve_token_path",
            return_value=None,
        ),
        patch(
            "syft.sync.login_utils._get_default_syftbox_path",
            return_value=syftbox_folder,
        ),
        patch(
            "syft.sync.login_utils._read_remote_version",
            side_effect=read_remote,
        ),
        patch(
            "syft.sync.login_utils._prompt_mismatch",
            return_value="1",
        ),
        patch(
            "syft.sync.login_utils.delete_local_syftbox",
        ) as mock_delete_local,
        patch(
            "syft.sync.login_utils.delete_remote_syftbox",
        ) as mock_delete_remote,
    ):
        from syft.sync.login_utils import (
            handle_potential_version_mismatches_on_login,
        )

        handle_potential_version_mismatches_on_login(email)
        mock_delete_local.assert_not_called()
        mock_delete_remote.assert_not_called()


def test_version_mismatch_continues_and_repairs():
    """Upgrade keeps peers, jobs, and data; private folders adopt; P2P reuses."""

    # -- Step 1: Create DO/DS on current version --
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
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

    # -- Step 4: Record P2P folders and personal folder id before upgrade --
    do_conn = do_manager.peer_manager.connection_router.connections[0]
    do_p2p_current = _find_versioned_p2p_folders(
        do_conn, ds_manager.email, SYFT_VERSION
    )
    assert len(do_p2p_current) > 0
    old_personal_name = f"{SYFT_VERSION}#{do_manager.email}"
    old_personal_id = do_conn._find_folder_by_name(
        old_personal_name,
        parent_id=do_conn.get_syftbox_folder_id(),
        owner_email=do_manager.email,
    )
    assert old_personal_id is not None

    # -- Step 5: Extract backing store --
    backing_store = _get_backing_store(do_manager)
    do_email = do_manager.email
    ds_email = ds_manager.email

    # -- Step 6+7: Upgrade DO (continue keeps data) --
    with (
        patch("syft.version.SYFT_VERSION", NEW_VERSION),
        patch(
            "syft.sync.connections.drive.gdrive_transport.SYFT_VERSION",
            NEW_VERSION,
        ),
        patch("syft.sync.login_utils.SYFT_VERSION", NEW_VERSION),
        patch("syft.sync.version.version_info.SYFT_VERSION", NEW_VERSION),
    ):
        do_syftbox = do_manager.syftbox_folder
        _simulate_continue_on_mismatch(do_manager, backing_store)
        do_manager = _reinitialize_manager(
            do_email,
            backing_store,
            has_do_role=True,
            has_ds_role=False,
            syftbox_folder=do_syftbox,
        )

        # Personal folder is adopted (same Drive id, new name), not recreated.
        do_conn_new = do_manager.peer_manager.connection_router.connections[0]
        new_personal_name = f"{NEW_VERSION}#{do_email}"
        new_personal_id = do_conn_new._find_folder_by_name(
            new_personal_name,
            parent_id=do_conn_new.get_syftbox_folder_id(),
            owner_email=do_email,
        )
        assert new_personal_id is not None
        assert new_personal_id == old_personal_id
        assert (
            do_conn_new._find_folder_by_name(
                old_personal_name,
                parent_id=do_conn_new.get_syftbox_folder_id(),
                owner_email=do_email,
            )
            is None
        )

        # Peers survive: continue did not wipe SYFT_peers.json.
        do_manager.load_peers()
        assert any(p.email == ds_email for p in do_manager.peer_manager.approved_peers)

        # Pre-upgrade job is still present on the kept datasite.
        assert any(job.name == "pre_upgrade.job" for job in do_manager.jobs)

        # -- Step 8+9: Upgrade DS --
        ds_syftbox = ds_manager.syftbox_folder
        _simulate_continue_on_mismatch(ds_manager, backing_store)
        ds_manager = _reinitialize_manager(
            ds_email,
            backing_store,
            has_do_role=False,
            has_ds_role=True,
            syftbox_folder=ds_syftbox,
        )

        ds_conn_new = ds_manager.peer_manager.connection_router.connections[0]
        ds_personal_name = f"{NEW_VERSION}#{ds_email}"
        ds_personal_id = ds_conn_new._find_folder_by_name(
            ds_personal_name,
            parent_id=ds_conn_new.get_syftbox_folder_id(),
            owner_email=ds_email,
        )
        assert ds_personal_id is not None

        ds_manager.load_peers()
        assert any(p.email == do_email for p in ds_manager.peer_manager.approved_peers)

        # P2P folders of the old version are reused, not replaced. A peer that
        # has not upgraded still looks for the old name.
        do_p2p_new = _find_versioned_p2p_folders(do_conn_new, ds_email, NEW_VERSION)
        assert len(do_p2p_new) == 0
        do_p2p_old = _find_versioned_p2p_folders(do_conn_new, ds_email, SYFT_VERSION)
        assert len(do_p2p_old) > 0
        assert len(do_p2p_old) == len(do_p2p_current)

        # -- Step 10: Submit a new job without re-peering --
        project_dir2 = create_test_project_folder(with_pyproject=False)
        ds_manager.submit_python_job(
            user=do_manager.email,
            code_path=str(project_dir2),
            job_name="post_upgrade.job",
            entrypoint="main.py",
        )
        do_manager.sync()

        post = [job for job in do_manager.jobs if job.name == "post_upgrade.job"]
        assert len(post) == 1
        post[0].approve()
        do_manager.process_approved_jobs()
        do_manager.sync()
        # Reload from disk; the pre-process JobState object does not update in place.
        post = [job for job in do_manager.jobs if job.name == "post_upgrade.job"]
        assert len(post) == 1
        assert post[0].status == "done"

        ds_manager.sync()
        ds_post = [
            job for job in ds_manager.job_client.jobs if job.name == "post_upgrade.job"
        ]
        assert len(ds_post) == 1
        assert ds_post[0].status == "done"


def _upgraded_manager(manager):
    """The same client after an upgrade: a new process on the same data.

    A real upgrade restarts the process, so the peer manager computes its own
    version again. Reusing the pre-upgrade object would read a cached version.
    """
    # No version write here. The test must show that login is what refreshes
    # the version files, so the new manager must not do it first.
    return _reinitialize_manager(
        manager.email,
        _get_backing_store(manager),
        has_do_role=True,
        has_ds_role=False,
        syftbox_folder=manager.syftbox_folder,
        write_version=False,
    )


def test_login_writes_the_remote_version_file_too():
    """Login must refresh both version files, not only the local one.

    A peer reads the remote file to select a job or dataset protocol version for
    us. A local-only write leaves that file at the version that first created
    it, so peers keep negotiating against a client we no longer run.
    """
    from syft.sync.login import _init_client_login
    from syft.sync.version.local_version import read_local_version

    _, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        check_versions=True,
    )
    conn = do_manager.peer_manager.connection_router.connections[0]
    assert conn.read_own_version_file().syft_client_version == SYFT_VERSION

    with (
        patch("syft.version.SYFT_VERSION", NEW_VERSION),
        patch("syft.sync.version.version_info.SYFT_VERSION", NEW_VERSION),
    ):
        upgraded = _upgraded_manager(do_manager)
        new_conn = upgraded.peer_manager.connection_router.connections[0]
        # Still the pre-upgrade version: nothing has refreshed it yet.
        assert new_conn.read_own_version_file().syft_client_version == SYFT_VERSION

        _init_client_login(upgraded.sync_engine, sync=False, load_peers=False)

        assert new_conn.read_own_version_file().syft_client_version == NEW_VERSION
        local = read_local_version(upgraded.syftbox_folder)
        assert local is not None
        assert local.syft_client_version == NEW_VERSION


def test_the_mismatch_prompt_does_not_return_after_a_login():
    """The prompt asks once per upgrade, not once per login.

    The check compares the installed client with the local and the remote
    version file. Login refreshes both, so the next login finds no mismatch.
    """
    from syft.sync.login import _init_client_login

    _, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        check_versions=True,
    )
    email = do_manager.email
    syftbox_folder = do_manager.syftbox_folder

    with (
        patch("syft.version.SYFT_VERSION", NEW_VERSION),
        patch("syft.sync.version.version_info.SYFT_VERSION", NEW_VERSION),
        patch("syft.sync.login_utils.SYFT_VERSION", NEW_VERSION),
        patch("syft.sync.login_utils._resolve_email", return_value=email),
        patch("syft.sync.login_utils._resolve_token_path", return_value=None),
        patch(
            "syft.sync.login_utils._get_default_syftbox_path",
            return_value=syftbox_folder,
        ),
        patch(
            "syft.sync.login_utils._prompt_mismatch", return_value="1"
        ) as mock_prompt,
    ):
        from syft.sync.login_utils import (
            handle_potential_version_mismatches_on_login,
        )

        conn = do_manager.peer_manager.connection_router.connections[0]
        with patch(
            "syft.sync.login_utils._read_remote_version",
            side_effect=lambda e, t: conn.read_own_version_file(),
        ):
            # The check runs before the client exists, so it reads the files
            # the previous client version left behind.
            handle_potential_version_mismatches_on_login(email)
            assert mock_prompt.call_count == 1

            upgraded = _upgraded_manager(do_manager)
            _init_client_login(upgraded.sync_engine, sync=False, load_peers=False)

            # Every login after that finds both files current, and asks nothing.
            handle_potential_version_mismatches_on_login(email)
            handle_potential_version_mismatches_on_login(email)
            assert mock_prompt.call_count == 1
