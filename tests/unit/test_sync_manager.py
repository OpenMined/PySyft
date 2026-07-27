from pathlib import Path

import pytest
import yaml

from syft_client.sync.connections.connection_router import ConnectionRouter
from syft_client.sync.connections.drive.gdrive_transport import GDriveConnection
from syft_client.sync.peers.peer_store import PeerStore
from syft_client.sync.messages.proposed_filechange import ProposedFileChangesMessage
from syft_client.sync.messages.proposed_filechange import ProposedFileChange
from syft_client.sync.syftbox_manager import SyftboxManager
from syft_client.sync.sync.caches.datasite_owner_cache import (
    ProposedEventFileOutdatedException,
)
from syft_client.sync.sync.collection_spec import CollectionSyncSpec
from tests.unit.utils import (
    TEST_COLLECTION_PREFIX,
    TEST_COLLECTION_SUBPATH,
    get_mock_events_messages,
    get_mock_proposed_events_messages,
    grant_job_inbox_access,
)


def path_for_job(do_email: str, ds_email: str, filename: str = "test.job") -> str:
    """Return the correct path for DS to submit a job file to DO."""
    return f"{do_email}/app_data/job/inbox/{ds_email}/{filename}"


def test_sync_to_syftbox_eventlog():
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()

    grant_job_inbox_access(do_manager, ds_manager.email)
    file_path = path_for_job(do_manager.email, ds_manager.email, "my.job")

    events_in_backing_platform = do_manager._get_all_accepted_events_do()
    assert len(events_in_backing_platform) == 0

    ds_manager._send_file_change(file_path, "Hello, world!")
    do_manager.sync()

    # second event is present
    events_in_backing_platform = do_manager._get_all_accepted_events_do()
    assert len(events_in_backing_platform) > 0


def test_valid_and_invalid_proposed_filechange_event():
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
    do_manager.datasite_owner_syncer.perm_context.open(".").grant_write_access(
        ds_manager.email
    )
    ds_email = ds_manager.email
    do_email = do_manager.email

    path_from_syftbox = f"{do_email}/test.job"
    path_in_datasite = path_from_syftbox.split("/")[-1]

    # create first message to create a hash
    message_1 = ProposedFileChangesMessage(
        sender_email=ds_email,
        proposed_file_changes=[
            ProposedFileChange(
                old_hash=None,
                path_in_datasite=path_in_datasite,
                content="Content 1",
                datasite_email=do_email,
            )
        ],
    )

    # create modification that corresponds to the first message
    hash1 = message_1.proposed_file_changes[0].new_hash
    do_manager.datasite_owner_syncer.handle_proposed_filechange_events_message(
        ds_email, message_1
    )

    message_2 = ProposedFileChangesMessage(
        sender_email=ds_email,
        proposed_file_changes=[
            ProposedFileChange(
                old_hash=hash1,
                path_in_datasite=path_in_datasite,
                content="Content 2",
                datasite_email=do_email,
            )
        ],
    )
    do_manager.datasite_owner_syncer.handle_proposed_filechange_events_message(
        ds_email, message_2
    )

    content = do_manager.datasite_owner_syncer.event_cache.file_connection.read_file(
        path_in_datasite
    )
    assert content == "Content 2"

    message_3_outdated = ProposedFileChangesMessage(
        sender_email=ds_email,
        proposed_file_changes=[
            ProposedFileChange(
                old_hash=hash1,
                path_in_datasite=path_in_datasite,
                content="Content 3",
                datasite_email=do_email,
            )
        ],
    )

    # This should fail, as the event is outdated
    with pytest.raises(ProposedEventFileOutdatedException):
        do_manager.datasite_owner_syncer.handle_proposed_filechange_events_message(
            ds_email, message_3_outdated
        )

    content = do_manager.datasite_owner_syncer.event_cache.file_connection.read_file(
        path_in_datasite
    )
    assert content == "Content 2"


def test_sync_back_to_ds_cache():
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()

    grant_job_inbox_access(do_manager, ds_manager.email)
    file_path = path_for_job(do_manager.email, ds_manager.email)
    ds_manager._send_file_change(file_path, "Hello, world!")

    do_manager.sync()  # DO processes inbox and pushes to outbox
    ds_manager.sync()  # DS pulls from DO's outbox
    assert (
        len(
            ds_manager.datasite_watcher_syncer.datasite_watcher_cache.get_cached_events()
        )
        == 1
    )


def test_sync_existing_datasite_state_do():
    """Test that DO can sync and cache events from DS.

    Creates state via DS sending file changes to DO, then verifies DO's cache.
    """
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )
    connection_ds = ds_manager._connection_router.connections[0]
    events_messages = get_mock_events_messages(2)

    for message in events_messages:
        do_manager._connection_router.owner_write_events_message_to_syftbox(message)
        do_manager._connection_router.owner_write_event_messages_to_outbox(
            ds_manager.email, events_messages[0]
        )

    # DO syncs to receive the changes
    do_manager.sync()

    # Verify DO's cache has the events
    n_messages_in_cache = len(
        do_manager.datasite_owner_syncer.event_cache.events_messages_connection
    )
    n_files_in_cache = len(do_manager.datasite_owner_syncer.event_cache.file_connection)
    hashes_in_cache = len(do_manager.datasite_owner_syncer.event_cache.file_hashes)

    n_outbox = connection_ds.watcher_get_outbox_file_metadatas(do_manager.email, None)
    assert n_messages_in_cache >= 1  # At least 1 message with the 2 file changes
    assert n_files_in_cache == 2  # 2 data files
    assert hashes_in_cache == 2  # 2 data files
    assert len(n_outbox) >= 1


def test_sync_existing_inbox_state_do():
    """Test that DO processes inbox messages from DS and creates events.

    DS sends file changes which arrive in DO's inbox. DO syncs and processes them.
    """
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
    )
    do_manager.datasite_owner_syncer.perm_context.open(".").grant_write_access(
        ds_manager.email
    )
    connection_ds = ds_manager._connection_router.connections[0]

    proposed_events_messages = get_mock_proposed_events_messages(
        2, email=ds_manager.email
    )
    for message in proposed_events_messages:
        connection_ds.watcher_send_proposed_file_changes_message(
            do_manager.email, message
        )

    # DO syncs to process inbox messages
    do_manager.sync()

    # Verify DO's cache has processed the events
    n_events_message_in_cache = len(
        do_manager.datasite_owner_syncer.event_cache.events_messages_connection
    )
    n_files_in_cache = len(do_manager.datasite_owner_syncer.event_cache.file_connection)
    hashes_in_cache = len(do_manager.datasite_owner_syncer.event_cache.file_hashes)
    assert n_events_message_in_cache >= 1  # At least 1 message with 2 file changes
    assert n_files_in_cache == 3  # 2 data files + root syft.pub.yaml
    assert hashes_in_cache == 3  # 2 data files + root syft.pub.yaml


def test_sync_existing_datasite_state_ds():
    """Test that DS can sync events from DO's outbox.

    Creates state via DO creating files and syncing, then verifies DS receives them.
    """
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )
    events_messages = get_mock_events_messages(2)
    for message in events_messages:
        do_manager._connection_router.owner_write_event_messages_to_outbox(
            ds_manager.email, message
        )

    ds_manager.sync()

    # Verify DS received events (files may be batched into fewer messages)
    ds_events_in_cache = len(
        ds_manager.datasite_watcher_syncer.datasite_watcher_cache.get_cached_events()
    )
    assert ds_events_in_cache == 2


def test_load_peers():
    """Test peer loading and persistence across restarts."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        add_peers=False
    )

    ds_manager.add_peer("peer1@email.com")
    ds_manager.add_peer(do_manager.email)

    do_manager.load_peers()

    do_manager.approve_peer_request(ds_manager.email)

    # reset the peers and load them from connection
    do_manager._approved_peers = []
    do_manager._peer_requests = []
    do_manager._outstanding_peer_requests = []
    ds_manager._approved_peers = []
    ds_manager._peer_requests = []
    ds_manager._outstanding_peer_requests = []

    do_manager.load_peers()
    ds_manager.load_peers()

    assert len(ds_manager.peers) == 2
    assert len(do_manager.peers) == 1


def test_file_connections():
    """Test file sync between DS and DO using filesystem caches."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )
    do_manager.datasite_owner_syncer.perm_context.open(".").grant_write_access(
        ds_manager.email
    )

    datasite_dir_do = (
        do_manager.datasite_owner_syncer.event_cache.file_connection.base_dir
    )

    syftbox_dir_ds = ds_manager.datasite_watcher_syncer.datasite_watcher_cache.file_connection.base_dir

    assert datasite_dir_do != syftbox_dir_ds

    job_path = path_for_job(do_manager.email, ds_manager.email)
    job_path_in_datasite = "/".join(job_path.split("/")[1:])

    ds_manager._send_file_change(job_path, "Hello, world!")
    do_manager.sync()

    assert (datasite_dir_do / job_path_in_datasite).exists()

    result_rel_path = "test_result.job"
    result_path = datasite_dir_do / result_rel_path
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        f.write("I am a result")

    do_manager.sync()

    ds_manager.sync()

    assert (syftbox_dir_ds / do_manager.email / result_rel_path).exists()


def test_file_deletion_do_to_ds():
    """Test that DO can delete a file and it syncs to DS"""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )

    datasite_dir_do = do_manager.syftbox_folder / do_manager.email
    syftbox_dir_ds = ds_manager.syftbox_folder

    # Grant DS read access at root level
    ctx = do_manager.datasite_owner_syncer.perm_context
    ctx.open(".").grant_read_access(ds_manager.email)

    # DO creates a file
    result_rel_path = "test_file.txt"
    result_path = datasite_dir_do / result_rel_path
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


def test_in_memory_deletion():
    """Test deletion works with in-memory cache"""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=True
    )

    grant_job_inbox_access(do_manager, ds_manager.email)

    # Create file via send_file_change
    job_path = path_for_job(do_manager.email, ds_manager.email)
    job_path_in_datasite = "/".join(job_path.split("/")[1:])

    ds_manager._send_file_change(job_path, "Hello, world!")
    do_manager.sync()

    # Verify file exists in DO cache
    do_cache = do_manager.datasite_owner_syncer.event_cache
    assert job_path_in_datasite in [
        str(p) for p, _ in do_cache.file_connection.get_items()
    ]

    # Simulate deletion by removing from DO cache
    do_cache.file_connection.delete_file(job_path_in_datasite)

    # Process deletion
    do_manager.sync()
    ds_manager.sync()

    # Verify deletion propagated
    ds_cache = ds_manager.datasite_watcher_syncer.datasite_watcher_cache
    ds_path = Path(do_manager.email) / job_path_in_datasite
    assert str(ds_path) not in [str(p) for p, _ in ds_cache.file_connection.get_items()]


def test_collection_files_excluded_from_outbox_sync():
    """Test that files in a registered collection folder are excluded from outbox sync.

    Collections have their own dedicated sync channel with proper permissions,
    so they should not be broadcast to all peers via the general outbox.
    """
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        collection_specs=[
            CollectionSyncSpec.public(TEST_COLLECTION_PREFIX, TEST_COLLECTION_SUBPATH)
        ],
    )

    datasite_dir_do = do_manager.syftbox_folder / do_manager.email

    # Grant DS read access to public/ so regular_file.txt syncs
    ctx = do_manager.datasite_owner_syncer.perm_context
    ctx.open("public/").grant_read_access(ds_manager.email)

    # Create a regular file (should be synced)
    regular_file = datasite_dir_do / "public" / "regular_file.txt"
    regular_file.parent.mkdir(parents=True, exist_ok=True)
    regular_file.write_text("regular content")

    # Create a collection file (should NOT be synced via outbox)
    collection_file = (
        datasite_dir_do / TEST_COLLECTION_SUBPATH / "my_collection" / "data.yaml"
    )
    collection_file.parent.mkdir(parents=True, exist_ok=True)
    collection_file.write_text("name: my_collection")

    # Sync DO to generate events
    do_manager.sync()

    # Check which files are in the DO's event cache (i.e., what gets sent to outbox)
    do_cache = do_manager.datasite_owner_syncer.event_cache
    cached_paths = [str(p) for p in do_cache.file_hashes.keys()]

    # Regular file should be in cache (will be synced)
    assert any("regular_file.txt" in p for p in cached_paths), (
        "Regular files should be included in outbox sync"
    )

    # Collection file should NOT be in cache (excluded from outbox)
    assert not any("my_collection" in p for p in cached_paths), (
        "Files in a registered collection should be excluded from outbox sync"
    )


def test_job_files_sync_to_submitter_only():
    """Test that job files only sync to the peer who submitted the job.

    When a DO has multiple approved peers, job results should only be sent
    to the peer who submitted that specific job, not broadcast to all peers.
    Uses permission-based routing: submitter gets read access via share_outputs(),
    non-submitter has no read access to job outputs.
    """
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
    )

    submitter_email = "submitter_peer@example.com"
    non_submitter_email = ds_manager.email

    datasite_dir_do = do_manager.syftbox_folder / do_manager.email

    # Create job directory structure
    job_name = "test_job_123"
    job_dir = datasite_dir_do / "app_data" / "job" / job_name
    job_dir.mkdir(parents=True, exist_ok=True)

    # Write config.yaml
    config_path = job_dir / "config.yaml"
    config_data = {"submitted_by": submitter_email, "status": "completed"}
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    # Grant submitter read access to job outputs (simulates share_outputs)
    ctx = do_manager.datasite_owner_syncer.perm_context
    ctx.open(f"app_data/job/{job_name}/outputs/").grant_read_access(submitter_email)

    # Create job result file
    result_file = job_dir / "outputs" / "result.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text('{"result": 42}')

    # Grant both peers read access to public/ so shared.txt goes to both
    public_folder = ctx.open("public/")
    public_folder.grant_read_access(submitter_email)
    public_folder.grant_read_access(non_submitter_email)

    # Create a regular file (non-job) that should go to all peers
    regular_file = datasite_dir_do / "public" / "shared.txt"
    regular_file.parent.mkdir(parents=True, exist_ok=True)
    regular_file.write_text("shared content")

    # Process local changes with both recipients
    recipients = [submitter_email, non_submitter_email]
    submitter_conn = GDriveConnection.from_service(
        submitter_email, ds_manager._connection_router.connections[0].drive_service
    )
    submitter_conn.add_peer(do_manager.email)
    submitter_router = ConnectionRouter(
        connections=[submitter_conn],
        peer_store=PeerStore(email=submitter_email),
    )
    do_manager.datasite_owner_syncer.process_local_changes(recipients)

    messages_for_non_submitter = (
        ds_manager._connection_router.watcher_get_events_messages(
            do_manager.email, None
        )
    )

    paths_for_non_submitter = [
        str(event.path_in_datasite)
        for msg in messages_for_non_submitter
        for event in msg.events
    ]

    messages_for_submitter = submitter_router.watcher_get_events_messages(
        do_manager.email, None
    )
    paths_for_submitter = [
        str(event.path_in_datasite)
        for msg in messages_for_submitter
        for event in msg.events
    ]
    # Job output files should ONLY be in submitter's outbox
    assert any(
        "app_data/job" in p and "result.json" in p for p in paths_for_submitter
    ), "Job output files should be sent to submitter"
    assert not any("result.json" in p for p in paths_for_non_submitter), (
        "Job output files should NOT be sent to non-submitter peers"
    )

    # Regular files should be in BOTH outboxes
    assert any("shared.txt" in p for p in paths_for_submitter), (
        "Regular files should be sent to submitter"
    )
    assert any("shared.txt" in p for p in paths_for_non_submitter), (
        "Regular files should be sent to non-submitter peers"
    )


def test_in_memory_connection_syncing():
    """Test basic syncing flow with mock drive service connection.

    Unit test equivalent of integration test_google_drive_connection_syncing.
    """
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()

    grant_job_inbox_access(do_manager, ds_manager.email)

    # DS sends a file change to DO's job folder (where DS has write access)
    ds_manager._send_file_change(
        path_for_job(do_manager.email, ds_manager.email, "my.job"), "Hello, world!"
    )

    # DO should have events in cache after sync
    do_manager.datasite_owner_syncer.sync(peer_emails=[ds_manager.email])
    assert len(do_manager.datasite_owner_syncer.event_cache.get_cached_events()) > 0

    # DS syncs to get any outbox updates from DO
    ds_manager.sync()

    events = (
        ds_manager.datasite_watcher_syncer.datasite_watcher_cache.get_cached_events()
    )
    assert len(events) > 0


def test_incoming_syft_pub_yaml_write_requires_admin():
    """Test that DS cannot write syft.pub.yaml unless they have admin access.

    DS proposes a change to a syft.pub.yaml file in the job folder.
    Even though DS has write access to the job folder, writing syft.pub.yaml
    requires admin, so the change should be rejected.
    """
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
    )

    grant_job_inbox_access(do_manager, ds_manager.email)

    ds_email = ds_manager.email
    do_email = do_manager.email
    datasite_dir_do = do_manager.syftbox_folder / do_email

    # DS has write access to their job folder (granted by approve_peer_request)
    # but NOT admin access. Verify this.
    from syft_perms import SyftPermContext

    ctx = SyftPermContext(datasite=datasite_dir_do)
    job_folder = ctx.open(f"app_data/job/inbox/{ds_email}/")
    assert job_folder.has_write_access(ds_email), "DS should have write access"

    # DS proposes a syft.pub.yaml change (trying to escalate permissions)
    perm_path = f"app_data/job/inbox/{ds_email}/syft.pub.yaml"
    message = ProposedFileChangesMessage(
        sender_email=ds_email,
        proposed_file_changes=[
            ProposedFileChange(
                old_hash=None,
                path_in_datasite=perm_path,
                content="rules:\n- pattern: '**'\n  access:\n    read: ['*']",
                datasite_email=do_email,
            )
        ],
    )

    do_manager.datasite_owner_syncer.handle_proposed_filechange_events_message(
        ds_email, message
    )

    # The syft.pub.yaml change should be rejected (requires admin)
    cached_events = do_manager.datasite_owner_syncer.event_cache.get_cached_events()
    perm_events = [
        e for e in cached_events if "syft.pub.yaml" in str(e.path_in_datasite)
    ]
    # Only the existing perm file from approve_peer_request should exist
    assert not any(
        f"app_data/job/inbox/{ds_email}/syft.pub.yaml" == str(e.path_in_datasite)
        for e in perm_events
    ), "DS should NOT be able to write syft.pub.yaml without admin access"


def test_permission_change_triggers_resend():
    """Test that changing permissions causes existing files to be resent to new readers.

    1. DO creates a file under a path where only peer A has read access
    2. DO syncs → peer A receives file, peer B does not
    3. DO grants peer B read access (writes syft.pub.yaml)
    4. DO syncs → peer B receives the file (resend triggered by perm change)
    """
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
    )

    peer_a_email = ds_manager.email
    peer_b_email = "peer_b@example.com"

    datasite_dir_do = do_manager.syftbox_folder / do_manager.email

    # Set up a second peer connection router for peer B
    peer_b_conn = GDriveConnection.from_service(
        peer_b_email, ds_manager._connection_router.connections[0].drive_service
    )
    peer_b_conn.add_peer(do_manager.email)
    peer_b_router = ConnectionRouter(
        connections=[peer_b_conn],
        peer_store=PeerStore(email=peer_b_email),
    )

    # Grant only peer A read access to project/
    ctx = do_manager.datasite_owner_syncer.perm_context
    ctx.open("project/").grant_read_access(peer_a_email)

    # DO creates a file under project/
    project_file = datasite_dir_do / "project" / "data.txt"
    project_file.parent.mkdir(parents=True, exist_ok=True)
    project_file.write_text("important data")

    # First sync: only peer A should receive the file
    recipients = [peer_a_email, peer_b_email]
    do_manager.datasite_owner_syncer.process_local_changes(recipients)

    messages_for_a = ds_manager._connection_router.watcher_get_events_messages(
        do_manager.email, None
    )
    paths_for_a = [
        str(e.path_in_datasite) for msg in messages_for_a for e in msg.events
    ]

    messages_for_b = peer_b_router.watcher_get_events_messages(do_manager.email, None)
    paths_for_b = [
        str(e.path_in_datasite) for msg in messages_for_b for e in msg.events
    ]

    assert any("data.txt" in p for p in paths_for_a), "Peer A should receive data.txt"
    assert not any("data.txt" in p for p in paths_for_b), (
        "Peer B should NOT receive data.txt yet"
    )

    # Now grant peer B read access by updating syft.pub.yaml
    ctx.open("project/").grant_read_access(peer_b_email)

    # Second sync: peer B should receive data.txt via resend
    do_manager.datasite_owner_syncer.process_local_changes(recipients)

    messages_for_b_after = peer_b_router.watcher_get_events_messages(
        do_manager.email, None
    )
    paths_for_b_after = [
        str(e.path_in_datasite) for msg in messages_for_b_after for e in msg.events
    ]

    assert any("data.txt" in p for p in paths_for_b_after), (
        "Peer B should receive data.txt after permission change"
    )


def test_default_collections_folder_picks_shareable_spec_regardless_of_order():
    """The collections folder is chosen by owner_only, not by position."""
    from syft_client.sync.syftbox_manager import default_collections_folder

    shareable = CollectionSyncSpec.public("syft_pub", Path("public/pub"))
    owner_only = CollectionSyncSpec.private("syft_priv", Path("private/priv"))
    expected = Path("/sb/me@t.com/public/pub")

    for specs in ([shareable, owner_only], [owner_only, shareable]):
        assert default_collections_folder("/sb", "me@t.com", specs) == expected

    assert default_collections_folder("/sb", "me@t.com", [owner_only]) is None
    assert default_collections_folder("/sb", "me@t.com", []) is None
