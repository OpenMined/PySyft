"""
Integration tests that have near-duplicate unit test coverage.

These tests verify the same logic as their unit test counterparts but use
real Google Drive connections instead of mocked connections.

Unit test equivalents:
- test_google_drive_connection_syncing -> test_in_memory_connection_syncing
- test_google_drive_files -> test_file_connections
- test_datasets -> test_datasets
- test_datasets_shared_with_any -> test_datasets_shared_with_any
- test_jobs -> test_jobs
- test_file_deletion_do_to_ds -> test_file_deletion_do_to_ds
"""

from syft_datasets.dataset import Dataset
import os
from pathlib import Path
import time
import json
from time import sleep
import pytest
from tests.unit.utils import create_tmp_dataset_files
from tests.integration.with_unit_coverage.utils import create_test_pair, is_mock_mode


def path_for_job(do_email: str, ds_email: str, filename: str = "test.job") -> str:
    """Return the correct path for DS to submit a job file to DO."""
    return f"{do_email}/app_data/job/inbox/{ds_email}/{filename}"


SYFT_CLIENT_DIR = Path(__file__).parent.parent.parent.parent
CREDENTIALS_DIR = SYFT_CLIENT_DIR / "credentials"

FILE_DO = os.environ.get("ai_audit_credentials_fname_do", "token_do.json")
EMAIL_DO = os.environ.get("AI_AUDIT_EMAIL_DO", "do@test.com")

FILE_DS = os.environ.get("ai_audit_credentials_fname_ds", "token_ds.json")
EMAIL_DS = os.environ.get("AI_AUDIT_EMAIL_DS", "ds@test.com")

token_path_do = CREDENTIALS_DIR / FILE_DO
token_path_ds = CREDENTIALS_DIR / FILE_DS


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.usefixtures("setup_delete_syftboxes")
def test_google_drive_connection_syncing():
    manager_ds, manager_do = create_test_pair(
        do_email=EMAIL_DO,
        ds_email=EMAIL_DS,
        do_token_path=token_path_do,
        ds_token_path=token_path_ds,
    )

    # this calls connection.send_propose_file_change_message via callbacks
    if not is_mock_mode():
        sleep(1)
    start_time = time.time()
    manager_ds._send_file_change(
        path_for_job(EMAIL_DO, EMAIL_DS, "my.job"), "Hello, world!"
    )
    end_time_sending = time.time()
    print(f"Time taken to send message: {end_time_sending - start_time} seconds")

    # wait for the message to be sent, this is not always needed
    sleep(1)

    # this is just for timing purposes, you can ignore it
    # continuing with the test

    manager_do.datasite_owner_syncer.sync(peer_emails=[EMAIL_DS])
    assert len(manager_do.datasite_owner_syncer.event_cache.get_cached_events()) > 0

    manager_ds.sync()

    events = (
        manager_ds.datasite_watcher_syncer.datasite_watcher_cache.get_cached_events()
    )
    assert len(events) > 0


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.usefixtures("setup_delete_syftboxes")
def test_google_drive_files():
    manager_ds, manager_do = create_test_pair(
        do_email=EMAIL_DO,
        ds_email=EMAIL_DS,
        do_token_path=token_path_do,
        ds_token_path=token_path_ds,
        use_in_memory_cache=False,
    )

    manager_do.datasite_owner_syncer.perm_context.open(".").grant_write_access(
        manager_ds.email
    )

    # syftbox_dir / EMAIL_DO
    datasite_dir_do = (
        manager_do.datasite_owner_syncer.event_cache.file_connection.base_dir
    )

    # syftbox_dir (ds)
    syftbox_dir_ds = manager_ds.datasite_watcher_syncer.datasite_watcher_cache.file_connection.base_dir

    assert datasite_dir_do != syftbox_dir_ds

    job_path = f"app_data/job/inbox/{EMAIL_DS}/test.job"
    job_send_path = f"{EMAIL_DO}/{job_path}"

    manager_ds._send_file_change(job_send_path, "This is a job")
    sleep(1)

    manager_do.sync()

    assert (datasite_dir_do / job_path).exists()

    result_rel_path = "test_result.result"
    result_path = datasite_dir_do / result_rel_path
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        f.write("I am a result")

    manager_do.sync()
    sleep(1)

    manager_ds.sync()

    assert (syftbox_dir_ds / EMAIL_DO / result_rel_path).exists()


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.usefixtures("setup_delete_syftboxes")
def test_datasets():
    ds_manager, do_manager = create_test_pair(
        do_email=EMAIL_DO,
        ds_email=EMAIL_DS,
        do_token_path=token_path_do,
        ds_token_path=token_path_ds,
        use_in_memory_cache=False,
    )

    mock_dset_path, private_dset_path, readme_path = create_tmp_dataset_files()
    do_manager.create_dataset(
        name="my dataset",
        mock_path=mock_dset_path,
        private_path=private_dset_path,
        summary="This is a summary",
        readme_path=readme_path,
        tags=["tag1", "tag2"],
        users=[ds_manager.email],
    )

    datasets = do_manager.datasets.get_all()
    assert len(datasets) == 1

    # Retrieve dataset by name
    dataset_do = do_manager.datasets["my dataset"]
    assert isinstance(dataset_do, Dataset)
    assert len(dataset_do.private_files) > 0
    assert len(dataset_do.mock_files) > 0

    ds_manager.sync()

    assert len(ds_manager.datasets.get_all()) == 1

    dataset_ds = ds_manager.datasets.get("my dataset", datasite=do_manager.email)
    assert dataset_ds.mock_files[0].exists()

    mock_content_ds = (dataset_ds.mock_dir / "mock.txt").read_text()
    assert len(mock_content_ds) > 0

    def has_file(root_dir, filename):
        return any(p.name == filename for p in Path(root_dir).rglob("*"))

    assert has_file(ds_manager.syftbox_folder, "mock.txt")
    assert not has_file(ds_manager.syftbox_folder, "private.txt")


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.usefixtures("setup_delete_syftboxes")
def test_datasets_shared_with_any():
    """Test that datasets shared with 'any' become discoverable after peer approval."""
    # Create managers WITHOUT auto peer setup
    ds_manager, do_manager = create_test_pair(
        do_email=EMAIL_DO,
        ds_email=EMAIL_DS,
        do_token_path=token_path_do,
        ds_token_path=token_path_ds,
        use_in_memory_cache=False,
        add_peers=False,  # Don't auto-add peers
    )

    mock_dset_path, private_dset_path, readme_path = create_tmp_dataset_files()

    # DO creates dataset with users='any' BEFORE peer is approved
    do_manager.create_dataset(
        name="any dataset",
        mock_path=mock_dset_path,
        private_path=private_dset_path,
        summary="Dataset shared with anyone",
        readme_path=readme_path,
        tags=["any"],
        users="any",
    )

    # DS should NOT see the dataset yet (not approved)
    ds_collections = ds_manager._connection_router.watcher_list_dataset_collections()
    assert not any(c["tag"] == "any dataset" for c in ds_collections)

    # DS adds peer, DO approves (this should share 'any' datasets)
    ds_manager.add_peer(do_manager.email)
    do_manager.load_peers()
    do_manager.approve_peer_request(ds_manager.email, peer_must_exist=False)

    # DS should now see the dataset
    # Google Drive search index takes time to propagate permission changes
    time.sleep(5)
    ds_collections = ds_manager._connection_router.watcher_list_dataset_collections()
    assert any(c["tag"] == "any dataset" for c in ds_collections)


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.usefixtures("setup_delete_syftboxes")
def test_jobs():
    ds_manager, do_manager = create_test_pair(
        do_email=EMAIL_DO,
        ds_email=EMAIL_DS,
        do_token_path=token_path_do,
        ds_token_path=token_path_ds,
        use_in_memory_cache=False,
    )

    test_py_path = "/tmp/test.py"
    with open(test_py_path, "w") as f:
        f.write("""
with open("outputs/result.json", "w") as f:
    f.write('{"result": 1}')
""")

    ds_manager.submit_python_job(
        user=do_manager.email,
        code_path=test_py_path,
        job_name="test.job",
    )

    do_manager.sync()
    assert len(do_manager.job_client.jobs) == 1
    job = do_manager.job_client.jobs[0]

    job.approve()

    do_manager.job_runner.process_approved_jobs()

    # Before sharing: DS should not see outputs
    do_manager.sync()
    ds_manager.sync()
    assert len(ds_manager.job_client.jobs[-1].output_paths) == 0

    # Share outputs with DS
    do_manager.job_runner.share_job_results(
        "test.job", share_outputs=True, share_logs=False
    )

    do_manager.sync()
    ds_manager.sync()

    output_path = ds_manager.job_client.jobs[-1].output_paths[0]
    with open(output_path, "r") as f:
        json_content = json.loads(f.read())

    assert json_content["result"] == 1


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.usefixtures("setup_delete_syftboxes")
def test_file_deletion_do_to_ds():
    """Test that DO can delete a file and it syncs to DS"""
    ds_manager, do_manager = create_test_pair(
        do_email=EMAIL_DO,
        ds_email=EMAIL_DS,
        do_token_path=token_path_do,
        ds_token_path=token_path_ds,
        use_in_memory_cache=False,
    )

    ctx = do_manager.datasite_owner_syncer.perm_context
    ctx.open(".").grant_read_access(ds_manager.email)

    datasite_dir_do = do_manager.syftbox_folder
    syftbox_dir_ds = ds_manager.syftbox_folder

    # DO creates a file
    result_rel_path = "test_file.txt"
    result_path = datasite_dir_do / do_manager.email / result_rel_path
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        f.write("This is a test file")

    # DO syncs (sends file to DS)
    do_manager.sync()

    # DS syncs (receives file from DO)
    ds_manager.sync()

    # Verify file exists on DS side
    ds_file_path = syftbox_dir_ds / do_manager.email / result_rel_path
    assert ds_file_path.exists(), "File should exist on DS side after sync"

    # DO deletes the file
    result_path.unlink()
    assert not result_path.exists(), "File should be deleted on DO side"

    # DO syncs (propagates deletion)
    do_manager.sync()

    # DS syncs (receives deletion)
    ds_manager.sync()

    # Verify file is deleted on DS side
    assert not ds_file_path.exists(), (
        "File should be deleted on DS side after DO deletes and both sync"
    )

    # Verify hash is removed from caches
    do_cache = do_manager.datasite_owner_syncer.event_cache
    assert result_rel_path not in do_cache.file_hashes, (
        "Hash should be removed from DO cache"
    )

    ds_cache = ds_manager.datasite_watcher_syncer.datasite_watcher_cache
    expected_path = Path(do_manager.email) / result_rel_path
    assert expected_path not in ds_cache.file_hashes, (
        "Hash should be removed from DS cache"
    )


@pytest.mark.skipif(is_mock_mode(), reason="State persistence requires real GDrive")
@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.usefixtures("setup_delete_syftboxes")
def test_google_drive_connection_load_state():
    # create the state

    # load the clients and add the peers
    manager_ds1, manager_do1 = create_test_pair(
        do_email=EMAIL_DO,
        ds_email=EMAIL_DS,
        do_token_path=token_path_do,
        ds_token_path=token_path_ds,
        add_peers=True,
        load_peers=False,
    )

    # make some changes
    manager_ds1._send_file_change(
        path_for_job(EMAIL_DO, EMAIL_DS, "my.job"), "Hello, world!"
    )
    manager_ds1._send_file_change(
        path_for_job(EMAIL_DO, EMAIL_DS, "my_second.job"), "Hello, world!"
    )

    # create a dataset with "any" permission to test loading and cache
    mock_dset_path, private_dset_path, readme_path = create_tmp_dataset_files()
    manager_do1.create_dataset(
        name="load_state_dataset",
        mock_path=mock_dset_path,
        private_path=private_dset_path,
        summary="Dataset for load state test",
        readme_path=readme_path,
        tags=["test"],
        users="any",
    )

    # verify dataset was created and cache was populated
    assert len(manager_do1.datasets.get_all()) == 1
    assert len(manager_do1.datasite_owner_syncer._any_shared_datasets) == 1

    # test loading the peers and loading the inbox
    manager_ds2, manager_do2 = create_test_pair(
        do_email=EMAIL_DO,
        ds_email=EMAIL_DS,
        do_token_path=token_path_do,
        ds_token_path=token_path_ds,
        add_peers=False,
        load_peers=False,
    )

    manager_do2.load_peers()
    assert len(manager_do2.peers) == 1

    manager_ds2.load_peers()
    assert len(manager_ds2.peers) == 1

    # sync so we have something in the syftbox and do outbox
    manager_do2.sync()

    assert len(manager_do2.datasite_owner_syncer.event_cache.get_cached_events()) == 2

    # we have created some state now, so now we can log in again and load the state
    # use a fresh syftbox folder to simulate clean filesystem
    manager_ds3, manager_do3 = create_test_pair(
        do_email=EMAIL_DO,
        ds_email=EMAIL_DS,
        do_token_path=token_path_do,
        ds_token_path=token_path_ds,
        add_peers=False,
        load_peers=True,
    )

    manager_do3.sync()
    manager_ds3.sync()

    loaded_events_do = manager_do3.datasite_owner_syncer.event_cache.get_cached_events()
    assert len(loaded_events_do) == 2

    loaded_events_ds = (
        manager_ds3.datasite_watcher_syncer.datasite_watcher_cache.get_cached_events()
    )
    assert len(loaded_events_ds) == 2

    # verify datasets were loaded from GDrive
    loaded_datasets = manager_do3.datasets.get_all()
    assert len(loaded_datasets) == 1
    assert loaded_datasets[0].name == "load_state_dataset"

    # verify _any_shared_datasets cache was populated during pull_initial_state
    assert len(manager_do3.datasite_owner_syncer._any_shared_datasets) == 1
    assert (
        manager_do3.datasite_owner_syncer._any_shared_datasets[0][0]
        == "load_state_dataset"
    )


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.usefixtures("setup_delete_syftboxes")
def test_version_upgrade_breaks_communication():
    """
    Integration test for version negotiation:
    1. Initialize peers with matching versions and verify they are compatible
    2. Update one peer's version file (simulating an upgrade)
    3. Reload the version and verify peers are now incompatible
    """
    from syft_client.sync.version.version_info import VersionInfo

    # Phase 1: Create managers with compatible versions
    ds_manager, do_manager = create_test_pair(
        do_email=EMAIL_DO,
        ds_email=EMAIL_DS,
        do_token_path=token_path_do,
        ds_token_path=token_path_ds,
        use_in_memory_cache=False,
        check_versions=True,
    )

    # Wait for GDrive permissions to propagate
    sleep(2)

    # Verify initial version compatibility
    ds_manager.peer_manager.load_peer_version(do_manager.email)
    do_manager.peer_manager.load_peer_version(ds_manager.email)

    assert ds_manager.peer_manager.is_peer_version_compatible(do_manager.email), (
        "DS should see DO as compatible initially"
    )
    assert do_manager.peer_manager.is_peer_version_compatible(ds_manager.email), (
        "DO should see DS as compatible initially"
    )

    # Phase 2: Simulate DS "upgrading" to a new incompatible version
    ds_connection = ds_manager._connection_router.connections[0]

    current = VersionInfo.current()
    new_version = VersionInfo(
        syft_client_version="99.0.0",  # Incompatible version
        min_supported_syft_client_version="99.0.0",
        protocol_version=current.protocol_version,
        min_supported_protocol_version=current.min_supported_protocol_version,
        updated_at=current.updated_at,
    )

    # Write the new version to GDrive (simulating DS upgrading their client)
    ds_connection.write_version_file(new_version)

    # Phase 3: Clear DO's cached version of DS and reload from GDrive
    do_manager.peer_manager.clear_peer_version(ds_manager.email)

    # Reload DS's version (this fetches from GDrive)
    reloaded_version = do_manager.peer_manager.load_peer_version(ds_manager.email)

    # Verify the new version was loaded
    assert reloaded_version is not None, "Should be able to reload peer version"
    assert reloaded_version.syft_client_version == "99.0.0", (
        "Reloaded version should be the upgraded version"
    )

    # Verify versions are now incompatible
    assert not do_manager.peer_manager.is_peer_version_compatible(ds_manager.email), (
        "DO should now see DS as incompatible after version upgrade"
    )

    # Cleanup: Restore DS's version file to the original version
    ds_connection.write_version_file(current)
