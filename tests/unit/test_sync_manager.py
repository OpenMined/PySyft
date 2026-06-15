from pathlib import Path
import json
import tempfile
import shutil

import pytest
import yaml

from syft_client.sync.connections.connection_router import ConnectionRouter
from syft_client.sync.connections.drive.gdrive_transport import GDriveConnection
from syft_client.sync.connections.drive import mock_drive_service
from syft_client.sync.peers.peer_store import PeerStore
from syft_client.sync.messages.proposed_filechange import ProposedFileChangesMessage
from syft_client.sync.messages.proposed_filechange import ProposedFileChange
from syft_client.sync.syftbox_manager import SyftboxManager
from syft_client.sync.sync.caches.datasite_owner_cache import (
    ProposedEventFileOutdatedException,
)
from syft_datasets.dataset import Dataset
from tests.unit.utils import (
    create_tmp_dataset_files,
    create_tmp_dataset_files_with_parquet,
    create_test_project_folder,
    get_mock_events_messages,
    get_mock_proposed_events_messages,
)


def path_for_job(do_email: str, ds_email: str, filename: str = "test.job") -> str:
    """Return the correct path for DS to submit a job file to DO."""
    return f"{do_email}/app_data/job/inbox/{ds_email}/{filename}"


def test_sync_to_syftbox_eventlog():
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()
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
    assert (
        n_files_in_cache == 4
    )  # 2 data files + 2 syft.pub.yaml permission files (inbox + review)
    assert (
        hashes_in_cache == 4
    )  # 2 data files + 2 syft.pub.yaml permission files (inbox + review)
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
    assert (
        n_files_in_cache == 5
    )  # 2 data files + 3 syft.pub.yaml permission files (inbox + review + root)
    assert (
        hashes_in_cache == 5
    )  # 2 data files + 3 syft.pub.yaml permission files (inbox + review + root)


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


def test_datasets():
    """Test dataset creation and sync between DO and DS."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )

    mock_dset_path, private_dset_path, readme_path = create_tmp_dataset_files()

    # Create dataset with specific users
    do_manager.create_dataset(
        name="my dataset",
        mock_path=mock_dset_path,
        private_path=private_dset_path,
        summary="This is a summary",
        readme_path=readme_path,
        tags=["tag1", "tag2"],
        users=[ds_manager.email],  # Share with specific user
    )

    # Verify collection created
    collections = do_manager._connection_router.owner_list_dataset_collections()
    assert "my dataset" in collections

    datasets = do_manager.datasets.get_all()
    assert len(datasets) == 1

    # Retrieve dataset by name
    dataset_do = do_manager.datasets["my dataset"]
    assert isinstance(dataset_do, Dataset)
    assert len(dataset_do.private_files) > 0
    assert len(dataset_do.mock_files) > 0

    ds_manager.sync()

    # Verify DS can see collection
    ds_collections = ds_manager._connection_router.watcher_list_dataset_collections()
    assert any(c["tag"] == "my dataset" for c in ds_collections)

    assert len(ds_manager.datasets.get_all()) == 1

    dataset_ds = ds_manager.datasets.get("my dataset", datasite=do_manager.email)

    assert dataset_ds.mock_files[0].exists()

    mock_content_ds = (dataset_ds.mock_dir / "mock.txt").read_text()
    assert len(mock_content_ds) > 0

    # test getting it via resolve path
    from syft_client import resolve_dataset_file_path

    mock_file_path = resolve_dataset_file_path("my dataset", client=ds_manager)
    assert mock_file_path.exists()

    mock_content_ds = mock_file_path.read_text()
    assert len(mock_content_ds) > 0

    def has_file(root_dir, filename):
        return any(p.name == filename for p in Path(root_dir).rglob("*"))

    assert has_file(ds_manager.syftbox_folder, "mock.txt")
    assert not has_file(ds_manager.syftbox_folder, "private.txt")
    # Confirm that "private.txt" does not exist anywhere in the DS syftbox folder
    for path in Path(ds_manager.syftbox_folder).rglob("*"):
        assert path.name != "private.txt"


def test_datasets_with_parquet():
    """Test dataset creation and sync with parquet files (binary format)."""
    import pandas as pd

    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )

    mock_dset_path, private_dset_path, readme_path = (
        create_tmp_dataset_files_with_parquet()
    )

    # This should work without errors even though parquet files are binary
    do_manager.create_dataset(
        name="parquet dataset",
        mock_path=mock_dset_path,
        private_path=private_dset_path,
        summary="This is a dataset with parquet files",
        readme_path=readme_path,
        tags=["parquet", "binary"],
        users=[ds_manager.email],
    )

    datasets = do_manager.datasets.get_all()
    assert len(datasets) == 1

    # Dataset files are synced via collections, not the event log.
    # Only the permission file (syft.pub.yaml) should appear as an event.
    cached_events = do_manager.datasite_owner_syncer.event_cache.get_cached_events()
    assert all(str(e.path_in_datasite).endswith("syft.pub.yaml") for e in cached_events)

    # Retrieve dataset by name
    dataset_do = do_manager.datasets["parquet dataset"]
    assert isinstance(dataset_do, Dataset)
    assert len(dataset_do.private_files) > 0
    assert len(dataset_do.mock_files) > 0

    # Verify parquet files are present
    mock_files = [f.name for f in dataset_do.mock_files]
    assert "mock_data.parquet" in mock_files

    private_files = [f.name for f in dataset_do.private_files]
    assert "private_data.parquet" in private_files

    # Sync to datasite
    ds_manager.sync()

    assert len(ds_manager.datasets.get_all()) == 1

    dataset_ds = ds_manager.datasets.get("parquet dataset", datasite=do_manager.email)

    # Verify the parquet file exists and can be read
    mock_parquet_path = dataset_ds.mock_dir / "mock_data.parquet"
    assert mock_parquet_path.exists()

    # Verify we can read the parquet file back
    df = pd.read_parquet(mock_parquet_path)
    assert len(df) == 5
    assert "name" in df.columns
    assert "age" in df.columns

    def has_file(root_dir, filename):
        return any(p.name == filename for p in Path(root_dir).rglob("*"))

    assert has_file(ds_manager.syftbox_folder, "mock_data.parquet")
    assert not has_file(ds_manager.syftbox_folder, "private_data.parquet")


def test_dataset_empty_permissions_no_access():
    """Test that empty permissions list means no one can access the dataset collection."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )

    mock_dset_path, private_dset_path, readme_path = create_tmp_dataset_files()

    # Create dataset with empty permissions list (share with no one)
    do_manager.create_dataset(
        name="private dataset",
        mock_path=mock_dset_path,
        private_path=private_dset_path,
        summary="This is a private summary",
        readme_path=readme_path,
        tags=["private"],
        users=[],  # Empty list - no one has access
    )

    # Verify collection created
    collections = do_manager._connection_router.owner_list_dataset_collections()
    assert "private dataset" in collections

    # DO should be able to see their own dataset
    datasets = do_manager.datasets.get_all()
    assert len(datasets) == 1

    # DS syncs
    ds_manager.sync()

    # DS should NOT see the collection (no permissions)
    ds_collections = ds_manager._connection_router.watcher_list_dataset_collections()
    assert not any(c["tag"] == "private dataset" for c in ds_collections)

    # DS should not have downloaded any datasets
    assert len(ds_manager.datasets.get_all()) == 0


def test_dataset_only_mock_data_uploaded():
    """Test that only mock data is uploaded to the collection, not private data."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )

    mock_dset_path, private_dset_path, readme_path = create_tmp_dataset_files()

    do_manager.create_dataset(
        name="test dataset",
        mock_path=mock_dset_path,
        private_path=private_dset_path,
        summary="Test summary",
        readme_path=readme_path,
        tags=["test"],
        users=[ds_manager.email],
    )

    # Sync so DS receives the dataset
    ds_manager.sync()

    files = ds_manager._connection_router.connections[
        0
    ].drive_service._backing_store.files
    list(files.keys())
    file_objs = list(files.values())
    file_objs_ex_dataset_yaml = [
        file_obj for file_obj in file_objs if file_obj.name != "dataset.yaml"
    ]

    assert not any("private" in file_obj.name for file_obj in file_objs)
    # dataset.yaml does mention "private", but thats just the path
    assert not any(
        b"private" in file_obj.content for file_obj in file_objs_ex_dataset_yaml
    )

    assert any("mock" in file_obj.name for file_obj in file_objs)
    assert any(b"Hello, world" in file_obj.content for file_obj in file_objs)

    mock_file = next(file_obj for file_obj in file_objs if file_obj.name == "mock.txt")

    # Verify mock content is correct
    mock_content = mock_file.content.decode("utf-8")
    assert len(mock_content) > 0, "Mock file should have content"
    assert "Hello" in mock_content, "Mock file should contain expected data"


def test_jobs():
    """Test basic job submission, approval, execution, and result sync."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
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

    # We want to make sure that we only send one message for the multiple files in the job.
    # this is to reduce the number of messages sent, which increases the speed of sync
    # we do this by not always syncing on a file change, currently this logic is a bit of
    # a short cut, but we could do this based on timing eventually (if there are items in the
    # queue for longer than a certain time we start pushing)
    connection_do = do_manager._connection_router.connections[0]
    inbox_folder_id = connection_do._get_own_datasite_inbox_id(ds_manager.email)
    inbox_file_metadatas = connection_do.get_file_metadatas_from_folder(inbox_folder_id)
    assert len(inbox_file_metadatas) == 1

    do_manager.sync()

    assert len(do_manager.job_client.jobs) == 1
    job = do_manager.job_client.jobs[0]

    job.approve()

    do_manager.job_runner.process_approved_jobs()
    do_manager.job_runner.share_job_results(
        "test.job", share_outputs=True, share_logs=False
    )

    do_manager.sync()

    ds_manager.sync()

    output_path = ds_manager.job_client.jobs[-1].output_paths[0]
    with open(output_path, "r") as f:
        json_content = json.loads(f.read())

    assert json_content["result"] == 1


def test_jobs_with_dataset():
    """Test job execution with dataset access using syft:// protocol."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
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
    do_manager.sync()

    ds_manager.sync()
    assert len(ds_manager.datasets.get_all()) == 1

    dataset_ds = ds_manager.datasets.get("my dataset", datasite=do_manager.email)
    assert dataset_ds.mock_files[0].exists()
    import syft_client as sc

    assert (
        sc.resolve_dataset_file_path("my dataset", client=ds_manager)
        == dataset_ds.mock_files[0]
    )

    test_py_path = "/tmp/test.py"
    with open(test_py_path, "w") as f:
        f.write("""
import syft_client as sc
import json

data_path = sc.resolve_dataset_file_path("my dataset")
with open(data_path, "r") as f:
    data = f.read()
result = {"result": len(data)}
with open("outputs/result.json", "w") as f:
    f.write(json.dumps(result))
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
    do_manager.job_runner.share_job_results(
        "test.job", share_outputs=True, share_logs=False
    )

    do_manager.sync()

    ds_manager.sync()

    output_path = ds_manager.job_client.jobs[-1].output_paths[0]
    with open(output_path, "r") as f:
        json_content = json.loads(f.read())

    with open(private_dset_path, "r") as f:
        private_data_length = len(f.read())

    assert json_content["result"] == private_data_length


def test_single_file_job_submission_without_pyproject():
    """Test that code files are placed in job_dir/code/ subdirectory.

    Verifies code is at:
        job_dir/code/main.py
    """
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )

    # Test with single file submission
    test_py_path = "/tmp/test_direct_copy.py"
    with open(test_py_path, "w") as f:
        f.write('print("hello")')

    ds_manager.submit_python_job(
        user=do_manager.email,
        code_path=test_py_path,
        job_name="test.direct.copy",
    )

    do_manager.sync()

    assert len(do_manager.job_client.jobs) == 1
    job = do_manager.job_client.jobs[0]
    job_dir = job.job_submission_path

    # Verify code is in job_dir/code/
    assert (job_dir / "code" / "test_direct_copy.py").exists(), (
        "Code should be in job_dir/code/"
    )
    assert (job_dir / "run.sh").exists(), "run.sh should exist"
    assert (job_dir / "config.yaml").exists(), "config.yaml should exist"


def test_folder_job_submission_without_pyproject():
    """Test folder submission without pyproject.toml uses uv venv + uv pip install.

    Verifies:
        - Folder without pyproject.toml works
        - Folder is preserved with its name (not dumped at root)
        - Generated run.sh uses 'uv venv' (not 'uv sync')
        - Entrypoint path includes folder name
    """
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )

    # Create a folder without pyproject.toml
    project_dir = tempfile.mkdtemp(prefix="test_no_pyproject_")
    Path(project_dir).name

    try:
        # Create main.py
        main_path = Path(project_dir) / "main.py"
        with open(main_path, "w") as f:
            f.write("""
with open("outputs/result.txt", "w") as f:
    f.write("success")
""")

        # Create a helper module
        helper_path = Path(project_dir) / "helper.py"
        with open(helper_path, "w") as f:
            f.write("VALUE = 42\n")

        # Submit folder (no pyproject.toml)
        ds_manager.submit_python_job(
            user=do_manager.email,
            code_path=project_dir,
            job_name="test.no.pyproject",
            entrypoint="main.py",
        )

        do_manager.sync()

        assert len(do_manager.job_client.jobs) == 1
        job = do_manager.job_client.jobs[0]
        job_dir = job.job_submission_path

        # Verify folder structure - code is in code/ subdirectory
        assert (job_dir / "code").exists(), "code/ should exist in job_dir"
        assert (job_dir / "code" / "main.py").exists(), "main.py should be inside code/"
        assert (job_dir / "code" / "helper.py").exists(), (
            "helper.py should be inside code/"
        )
        assert (job_dir / "run.sh").exists(), "run.sh should be at job_dir root"
        assert (job_dir / "config.yaml").exists(), (
            "config.yaml should be at job_dir root"
        )

        # Verify run.sh uses uv venv (not uv sync) and correct entrypoint path
        run_script = (job_dir / "run.sh").read_text()
        assert "uv venv" in run_script, (
            "Should use 'uv venv' for folders without pyproject.toml"
        )
        assert "uv sync" not in run_script, (
            "Should NOT use 'uv sync' without pyproject.toml"
        )
        assert "cd code" in run_script, "Should cd into code folder"
        assert "python main.py" in run_script, "Should run main.py from code/"

    finally:
        shutil.rmtree(project_dir, ignore_errors=True)


def test_folder_job_submission_with_pyproject():
    """Test folder submission with pyproject.toml uses uv sync.

    Verifies:
        - Folder is preserved with its name inside job_dir
        - pyproject.toml is inside the folder, not at job_dir root
        - run.sh uses 'uv sync' inside the folder
        - Entrypoint path includes folder name
    """
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )

    # Create a folder with pyproject.toml
    project_dir = tempfile.mkdtemp(prefix="test_with_pyproject_")
    Path(project_dir).name

    try:
        # Create pyproject.toml
        pyproject_path = Path(project_dir) / "pyproject.toml"
        with open(pyproject_path, "w") as f:
            f.write("""
[project]
name = "test-project"
version = "0.1.0"
dependencies = []
""")

        # Create main.py
        main_path = Path(project_dir) / "main.py"
        with open(main_path, "w") as f:
            f.write('print("hello from pyproject project")')

        # Submit folder (with pyproject.toml)
        ds_manager.submit_python_job(
            user=do_manager.email,
            code_path=project_dir,
            job_name="test.with.pyproject",
            entrypoint="main.py",
        )

        do_manager.sync()

        assert len(do_manager.job_client.jobs) == 1
        job = do_manager.job_client.jobs[0]
        job_dir = job.job_submission_path

        # Verify folder structure - code is in code/ subdirectory
        assert (job_dir / "code").exists(), "code/ should exist in job_dir"
        assert (job_dir / "code" / "main.py").exists(), "main.py should be inside code/"
        assert (job_dir / "code" / "pyproject.toml").exists(), (
            "pyproject.toml should be inside code/"
        )
        assert (job_dir / "run.sh").exists(), "run.sh should be at job_dir root"
        assert (job_dir / "config.yaml").exists(), (
            "config.yaml should be at job_dir root"
        )

        # Verify run.sh uses uv sync inside code folder and correct entrypoint path
        run_script = (job_dir / "run.sh").read_text()
        assert "uv sync" in run_script, (
            "Should use 'uv sync' for folders with pyproject.toml"
        )
        assert "cd code" in run_script, "Should cd into code folder for uv sync"
        assert "python main.py" in run_script, "Should run main.py from code/"

    finally:
        shutil.rmtree(project_dir, ignore_errors=True)


def test_folder_job_auto_detect_main_py():
    """Test that entrypoint is auto-detected when main.py exists."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )

    project_dir = tempfile.mkdtemp(prefix="test_auto_main_")
    Path(project_dir).name

    try:
        # Create main.py and another file
        (Path(project_dir) / "main.py").write_text('print("main")')
        (Path(project_dir) / "utils.py").write_text('print("utils")')

        # Submit without entrypoint - should auto-detect main.py
        ds_manager.submit_python_job(
            user=do_manager.email,
            code_path=project_dir,
            job_name="test.auto.main",
            # No entrypoint specified
        )

        do_manager.sync()
        job = do_manager.job_client.jobs[0]
        job_dir = job.job_submission_path

        # Verify main.py was auto-detected
        run_script = (job_dir / "run.sh").read_text()
        assert "cd code" in run_script, "Should cd into code folder"
        assert "python main.py" in run_script, (
            "Should auto-detect main.py as entrypoint"
        )

    finally:
        shutil.rmtree(project_dir, ignore_errors=True)


def test_folder_job_auto_detect_single_py():
    """Test that entrypoint is auto-detected when only one .py file exists."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )

    project_dir = tempfile.mkdtemp(prefix="test_auto_single_")
    Path(project_dir).name

    try:
        # Create only one .py file (not named main.py)
        (Path(project_dir) / "script.py").write_text('print("script")')
        (Path(project_dir) / "README.md").write_text("# Readme")

        # Submit without entrypoint - should auto-detect script.py
        ds_manager.submit_python_job(
            user=do_manager.email,
            code_path=project_dir,
            job_name="test.auto.single",
        )

        do_manager.sync()
        job = do_manager.job_client.jobs[0]
        job_dir = job.job_submission_path

        # Verify script.py was auto-detected
        run_script = (job_dir / "run.sh").read_text()
        assert "cd code" in run_script, "Should cd into code folder"
        assert "python script.py" in run_script, (
            "Should auto-detect single .py file as entrypoint"
        )

    finally:
        shutil.rmtree(project_dir, ignore_errors=True)


def test_folder_job_no_auto_detect_multiple_py():
    """Test that auto-detection fails when multiple .py files and no main.py."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )

    project_dir = tempfile.mkdtemp(prefix="test_no_auto_")

    try:
        # Create multiple .py files (no main.py)
        (Path(project_dir) / "script1.py").write_text('print("1")')
        (Path(project_dir) / "script2.py").write_text('print("2")')

        # Submit without entrypoint - should fail
        with pytest.raises(ValueError, match="Could not auto-detect entrypoint"):
            ds_manager.submit_python_job(
                user=do_manager.email,
                code_path=project_dir,
                job_name="test.no.auto",
            )

    finally:
        shutil.rmtree(project_dir, ignore_errors=True)


def test_single_file_job_flow_with_dataset():
    """Test complete job submission flow with dataset access.

    This test verifies the end-to-end flow of:
    1. Data Owner (DO) creates a dataset with mock and private data
    2. Data Scientist (DS) syncs and sees the dataset
    3. DS submits a Python job that accesses the private dataset
    4. DO approves and runs the job
    5. Job reads private data using syft:// protocol
    6. Results sync back to DS

    Test flow:
        DO: create_dataset("my dataset") with private.txt containing "Hello, world!"
                ↓
        DS: sync() → sees dataset
                ↓
        DS: submit_python_job() with code that reads syft://private/...
                ↓
        DO: sync() → receives job
                ↓
        DO: job.approve() + process_approved_jobs()
                ↓
        Job executes: reads private data → writes outputs/result.json
                ↓
        DO: sync() → sends results
                ↓
        DS: sync() → receives results
                ↓
        Assert: result.json contains {"result": "Hello, world!"}

    Verifies:
        - Dataset creation and sync between DO and DS
        - Job submission with syft:// path resolution
        - Job approval and execution workflow
        - Output file sync back to DS
    """
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )

    mock_dset_path, private_dset_path, readme_path = create_tmp_dataset_files()
    do_manager.create_dataset(
        name="my dataset",
        mock_path=mock_dset_path,
        private_path=private_dset_path,
        summary="This is a summary",
        readme_path=readme_path,
        tags=["tag1", "tag2"],
        users=[ds_manager.email],  # Share with DS so they can access the dataset
    )

    datasets = do_manager.datasets.get_all()
    assert len(datasets) == 1

    ds_manager.sync()

    assert len(ds_manager.datasets.get_all()) == 1

    test_py_path = "/tmp/test.py"
    with open(test_py_path, "w") as f:
        f.write("""
import json
import syft_client as sc

data_path = sc.resolve_dataset_file_path("my dataset")

with open(data_path, "r") as data_file:
    data = data_file.read()

result = {"result": data}

with open("outputs/result.json", "w") as f:
    f.write(json.dumps(result))
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

    # After sharing: DS should see outputs
    do_manager.job_runner.share_job_results(
        "test.job", share_outputs=True, share_logs=False
    )
    do_manager.sync()
    ds_manager.sync()

    output_path = ds_manager.job_client.jobs[-1].output_paths[0]
    with open(output_path, "r") as f:
        json_content = json.loads(f.read())

    assert json_content["result"] == "Hello, world private!"


def test_folder_job_flow_with_dataset():
    """Test job submission with a folder containing multiple Python files.

    Tests folder structure:
        project_dir/
        ├── main.py              # entrypoint, imports from helpers.helper
        └── helpers/
            ├── __init__.py      # package marker
            └── helper.py        # helper functions

    Verifies:
        - Folder submission with entrypoint parameter works
        - Nested package imports work (from helpers.helper import ...)
        - Outputs created at job root (not inside code/)
        - End-to-end flow: submit → approve → run → sync → verify output
    """
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )

    mock_dset_path, private_dset_path, readme_path = create_tmp_dataset_files()
    do_manager.create_dataset(
        name="my dataset",
        mock_path=mock_dset_path,
        private_path=private_dset_path,
        summary="This is a summary",
        readme_path=readme_path,
        tags=["tag1", "tag2"],
        users=[ds_manager.email],  # Share with DS so they can access the dataset
    )

    ds_manager.sync()
    assert len(ds_manager.datasets.get_all()) == 1

    # Create test project folder (no pyproject.toml, multiplier=2)
    project_dir = create_test_project_folder(with_pyproject=False, multiplier=2)

    try:
        ds_manager.submit_python_job(
            user=do_manager.email,
            code_path=str(project_dir),
            job_name="test.folder.job",
            entrypoint="main.py",
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

        # After sharing: DS should see outputs
        do_manager.job_runner.share_job_results(
            "test.folder.job", share_outputs=True, share_logs=False
        )
        do_manager.sync()
        ds_manager.sync()

        # Verify the job completed and produced output
        output_path = ds_manager.job_client.jobs[-1].output_paths[0]
        with open(output_path, "r") as f:
            json_content = json.loads(f.read())

        # Verify the helper module was imported and used correctly
        assert json_content["original"] == "Hello, world private!"
        assert json_content["processed"] == "Processed: Hello, world private!"
        assert json_content["multiplier"] == 2

    finally:
        shutil.rmtree(project_dir, ignore_errors=True)


def test_pyproject_folder_job_flow_with_dataset():
    """Test job submission with a folder containing pyproject.toml.

    Tests folder structure:
        project_dir/
        ├── pyproject.toml       # project config with dependencies
        ├── main.py              # entrypoint, imports from helpers.helper
        └── helpers/
            ├── __init__.py      # package marker
            └── helper.py        # helper functions

    Verifies:
        - Folder with pyproject.toml uses 'uv sync' (not 'uv venv')
        - Folder is preserved with its name in job_dir
        - .venv is created inside the code folder by uv sync
        - Nested package imports work
        - End-to-end flow: submit → approve → run → sync → verify output
    """
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )

    mock_dset_path, private_dset_path, readme_path = create_tmp_dataset_files()
    do_manager.create_dataset(
        name="my dataset",
        mock_path=mock_dset_path,
        private_path=private_dset_path,
        summary="This is a summary",
        readme_path=readme_path,
        tags=["tag1", "tag2"],
        users=[ds_manager.email],  # Share with DS so they can access the dataset
    )

    ds_manager.sync()
    assert len(ds_manager.datasets.get_all()) == 1

    # Create test project folder with pyproject.toml, multiplier=3
    project_dir = create_test_project_folder(
        with_pyproject=True, multiplier=3, prefix="test_pyproject_"
    )

    try:
        ds_manager.submit_python_job(
            user=do_manager.email,
            code_path=str(project_dir),
            job_name="test.pyproject.job",
            entrypoint="main.py",
        )

        do_manager.sync()
        assert len(do_manager.job_client.jobs) == 1
        job = do_manager.job_client.jobs[0]
        job_dir = job.job_submission_path

        # Verify folder structure before running - code is in code/ subdirectory
        assert (job_dir / "code").exists(), "code/ should exist in job_dir"
        assert (job_dir / "code" / "pyproject.toml").exists(), (
            "pyproject.toml should be inside code/"
        )
        assert (job_dir / "code" / "main.py").exists(), "main.py should be inside code/"
        assert (job_dir / "run.sh").exists(), "run.sh should be at job_dir root"

        # Verify run.sh uses uv sync (pyproject.toml case)
        run_script = (job_dir / "run.sh").read_text()
        assert "uv sync" in run_script, (
            "Should use 'uv sync' for folders with pyproject.toml"
        )
        assert "cd code" in run_script, "Should cd into code folder for uv sync"
        assert "python main.py" in run_script, "Should run main.py from code/"

        # Run the job
        job.approve()
        do_manager.job_runner.process_approved_jobs()

        # Verify .venv was created inside the code folder (by uv sync)
        assert (job_dir / "code" / ".venv").exists(), (
            ".venv should be created inside code/ folder by uv sync"
        )

        # Before sharing: DS should not see outputs
        do_manager.sync()
        ds_manager.sync()
        assert len(ds_manager.job_client.jobs[-1].output_paths) == 0

        # After sharing: DS should see outputs
        do_manager.job_runner.share_job_results(
            "test.pyproject.job", share_outputs=True, share_logs=False
        )
        do_manager.sync()
        ds_manager.sync()

        # Verify the job completed and produced output
        output_path = ds_manager.job_client.jobs[-1].output_paths[0]
        with open(output_path, "r") as f:
            json_content = json.loads(f.read())

        # Verify the helper module was imported and used correctly
        assert json_content["original"] == "Hello, world private!"
        assert json_content["processed"] == "Processed: Hello, world private!"
        assert json_content["multiplier"] == 3

    finally:
        shutil.rmtree(project_dir, ignore_errors=True)


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


def test_syft_datasets_excluded_from_outbox_sync():
    """Test that files in syft_datasets folder are excluded from outbox sync.

    Datasets have their own dedicated sync channel with proper permissions,
    so they should not be broadcast to all peers via the general outbox.
    """
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )

    datasite_dir_do = do_manager.syftbox_folder / do_manager.email

    # Grant DS read access to public/ so regular_file.txt syncs
    ctx = do_manager.datasite_owner_syncer.perm_context
    ctx.open("public/").grant_read_access(ds_manager.email)

    # Create a regular file (should be synced)
    regular_file = datasite_dir_do / "public" / "regular_file.txt"
    regular_file.parent.mkdir(parents=True, exist_ok=True)
    regular_file.write_text("regular content")

    # Create a dataset file (should NOT be synced via outbox)
    dataset_file = (
        datasite_dir_do / "public" / "syft_datasets" / "my_dataset" / "dataset.yaml"
    )
    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    dataset_file.write_text("name: my_dataset")

    # Sync DO to generate events
    do_manager.sync()

    # Check which files are in the DO's event cache (i.e., what gets sent to outbox)
    do_cache = do_manager.datasite_owner_syncer.event_cache
    cached_paths = [str(p) for p in do_cache.file_hashes.keys()]

    # Regular file should be in cache (will be synced)
    assert any("regular_file.txt" in p for p in cached_paths), (
        "Regular files should be included in outbox sync"
    )

    # Dataset file should NOT be in cache (excluded from outbox)
    assert not any("syft_datasets" in p for p in cached_paths), (
        "Files in syft_datasets should be excluded from outbox sync"
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


def test_ds_dataset_cache_aware_sync():
    """Test that DS loads dataset hashes from disk and skips re-download on restart.

    This test verifies cache-aware dataset syncing:
    1. Create pair 1, DO creates dataset, DS syncs and downloads
    2. Create pair 2 with same directories and backing store (simulates restart)
    3. Verify hash is loaded from disk on startup and matches remote hash
    4. This ensures sync_down_datasets will skip downloading (hash comparison passes)
    """
    from unittest.mock import patch

    # Create first pair
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )

    mock_dset_path, private_dset_path, readme_path = create_tmp_dataset_files()

    do_manager.create_dataset(
        name="cached dataset",
        mock_path=mock_dset_path,
        private_path=private_dset_path,
        summary="This is a cached dataset",
        readme_path=readme_path,
        tags=["cache", "test"],
        users=[ds_manager.email],
    )

    # DS syncs and receives dataset
    ds_manager.sync()

    # Verify dataset was downloaded
    assert len(ds_manager.datasets.get_all()) == 1
    dataset = ds_manager.datasets.get("cached dataset", datasite=do_manager.email)
    assert dataset.mock_files[0].exists()

    # Get the original hash from the collection
    collections = ds_manager._connection_router.watcher_list_dataset_collections()
    remote_hash = None
    for c in collections:
        if c["tag"] == "cached dataset":
            remote_hash = c["content_hash"]
            break
    assert remote_hash is not None

    # Get the mock backing store and directories for creating second pair
    mock_backing_store = ds_manager._connection_router.connections[
        0
    ].drive_service._backing_store
    ds_folder = ds_manager.syftbox_folder
    do_folder = do_manager.syftbox_folder
    ds_email = ds_manager.email
    do_email = do_manager.email

    # Create second pair with same directories (simulates restart)
    ds_manager2, do_manager2 = SyftboxManager.pair_with_mock_drive_service_connection(
        email1=do_email,
        email2=ds_email,
        base_path1=do_folder,
        base_path2=ds_folder,
        use_in_memory_cache=False,
        add_peers=False,
    )

    # Replace mock backing store to share dataset collections
    ds_manager2._connection_router.connections[
        0
    ].drive_service._backing_store = mock_backing_store
    do_manager2._connection_router.connections[
        0
    ].drive_service._backing_store = mock_backing_store

    # Load peers (already approved in shared backing store)
    ds_manager2.load_peers()
    do_manager2.load_peers()

    # Verify hash was loaded from disk on startup
    ds_cache = ds_manager2.datasite_watcher_syncer.datasite_watcher_cache
    # Cache uses full path as key: syftbox_folder / owner_email / collection_subpath / tag
    cache_key = ds_cache.get_collection_path(do_email, "cached dataset")
    assert cache_key in ds_cache.dataset_collection_hashes, (
        "Hash should be loaded from disk on startup"
    )

    # Verify the loaded hash matches the remote hash
    # This ensures sync_down_datasets will skip the download (hash comparison at line 283-284)
    loaded_hash = ds_cache.dataset_collection_hashes[cache_key]
    assert loaded_hash == remote_hash, (
        "Loaded hash should match remote hash, ensuring no re-download is needed"
    )

    # Patch the download method to verify it's NOT called (hash match should skip download)
    syncer = ds_manager2.datasite_watcher_syncer
    original_method = syncer.download_dataset_file_with_new_connection

    with patch(
        "syft_client.sync.sync.datasite_watcher_syncer.DatasiteWatcherSyncer.download_dataset_file_with_new_connection",
        wraps=original_method,
    ) as mock_download:
        # Sync - no download should happen because hash matches
        ds_manager2.sync()

        # Verify download_dataset_file_with_new_connection was NOT called
        assert mock_download.call_count == 0, (
            "Should not download files when local hash matches remote"
        )

    # Verify dataset still accessible
    assert len(ds_manager2.datasets.get_all()) == 1


def test_do_dataset_cache_aware_sync():
    """Test that DO doesn't re-download datasets on restart when hash matches.

    This test verifies cache-aware dataset syncing for DO side:
    1. Create pair 1, DO creates dataset
    2. Create pair 2 with same directories and backing store (simulates restart)
    3. Verify _download_dataset_collections_parallel is NOT called on sync
    """
    from unittest.mock import patch

    # Create first pair
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )

    mock_dset_path, private_dset_path, readme_path = create_tmp_dataset_files()

    do_manager.create_dataset(
        name="do cached dataset",
        mock_path=mock_dset_path,
        private_path=private_dset_path,
        summary="This is a DO cached dataset",
        readme_path=readme_path,
        tags=["cache", "do", "test"],
        users=[ds_manager.email],
    )

    # Verify dataset was created locally
    assert len(do_manager.datasets.get_all()) == 1

    # Get the mock backing store and directories for creating second pair
    mock_backing_store = ds_manager._connection_router.connections[
        0
    ].drive_service._backing_store
    ds_folder = ds_manager.syftbox_folder
    do_folder = do_manager.syftbox_folder
    ds_email = ds_manager.email
    do_email = do_manager.email

    # Create second pair with same directories (simulates restart)
    ds_manager2, do_manager2 = SyftboxManager.pair_with_mock_drive_service_connection(
        email1=do_email,
        email2=ds_email,
        base_path1=do_folder,
        base_path2=ds_folder,
        use_in_memory_cache=False,
        add_peers=False,
    )

    # Replace mock backing store to share dataset collections
    ds_manager2._connection_router.connections[
        0
    ].drive_service._backing_store = mock_backing_store
    do_manager2._connection_router.connections[
        0
    ].drive_service._backing_store = mock_backing_store

    # Load peers (already approved in shared backing store)
    ds_manager2.load_peers()
    do_manager2.load_peers()

    # Patch the download method to verify it's NOT called (hash match should skip download)
    syncer = do_manager2.datasite_owner_syncer
    with patch.object(
        syncer,
        "_download_file_with_new_connection",
        wraps=syncer._download_file_with_new_connection,
    ) as mock_download:
        # Sync - should NOT trigger download since local hash matches remote
        do_manager2.sync()

        # Verify _download_file_with_new_connection was NOT called
        assert mock_download.call_count == 0, (
            "Should not download files when local hash matches remote"
        )

    # Verify dataset still accessible
    assert len(do_manager2.datasets.get_all()) == 1


def test_in_memory_connection_syncing():
    """Test basic syncing flow with mock drive service connection.

    Unit test equivalent of integration test_google_drive_connection_syncing.
    """
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection()

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


def test_in_memory_connection_load_state():
    """Test state persistence and loading with mock drive connection.

    Unit test equivalent of integration test_google_drive_connection_load_state.

    Workflow (matches integration test):
    1. Pair 1: Create peers, make changes, create dataset
    2. Pair 2: Load peers, sync DO → verify events processed
    3. Pair 3: Load peers, sync both → verify state loaded from storage
    """
    from syft_client.sync.syftbox_manager import SyftboxManagerConfig

    # Get shared backing store and directories that will persist across pairs
    ds_manager1, do_manager1 = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        add_peers=True,
    )

    # Get the backing store that will persist across "restarts"
    backing_store = do_manager1._connection_router.connections[
        0
    ].drive_service._backing_store
    ds_folder = ds_manager1.syftbox_folder
    do_folder = do_manager1.syftbox_folder
    ds_email = ds_manager1.email
    do_email = do_manager1.email
    ds_manager1._connection_router.connections[0]
    do_manager1._connection_router.connections[0]

    # Make some changes (submit to DS's job folder where DS has write access)
    ds_manager1._send_file_change(
        path_for_job(do_manager1.email, ds_manager1.email, "my.job"), "Hello, world!"
    )
    ds_manager1._send_file_change(
        path_for_job(do_manager1.email, ds_manager1.email, "my_second.job"),
        "Hello, world!",
    )

    # Create a dataset with "any" permission
    mock_dset_path, private_dset_path, readme_path = create_tmp_dataset_files()
    do_manager1.create_dataset(
        name="load_state_dataset",
        mock_path=mock_dset_path,
        private_path=private_dset_path,
        summary="Dataset for load state test",
        readme_path=readme_path,
        tags=["test"],
        users="any",
    )

    # Verify dataset was created and cache populated
    assert len(do_manager1._connection_router.owner_list_dataset_collections()) == 1
    assert len(do_manager1.datasite_owner_syncer._any_shared_datasets) == 1

    # Create second pair (simulates restart, tests loading peers and processing inbox)
    do_config2 = SyftboxManagerConfig._base_config_for_testing(
        email=do_email,
        syftbox_folder=do_folder,
        has_ds_role=False,
        has_do_role=True,
        use_in_memory_cache=False,
        check_versions=False,
    )
    ds_config2 = SyftboxManagerConfig._base_config_for_testing(
        email=ds_email,
        syftbox_folder=ds_folder,
        has_ds_role=True,
        has_do_role=False,
        use_in_memory_cache=False,
        check_versions=False,
    )

    do_manager2 = SyftboxManager.from_config(do_config2)
    ds_manager2 = SyftboxManager.from_config(ds_config2)

    # Connect to the same backing store
    service_do = mock_drive_service.MockDriveService(backing_store, do_email)
    do_connection2 = GDriveConnection.from_service(do_email, service_do)

    service_ds = mock_drive_service.MockDriveService(backing_store, ds_email)
    ds_connection2 = GDriveConnection.from_service(ds_email, service_ds)

    do_manager2._add_connection(do_connection2)
    ds_manager2._add_connection(ds_connection2)

    # Load peers
    do_manager2.load_peers()
    assert len(do_manager2.peers) == 1

    ds_manager2.load_peers()
    assert len(ds_manager2.peers) == 1

    # Sync DO so we have something in the syftbox and do outbox
    do_manager2.sync()
    ds_manager2.sync()

    # Verify events in DO cache (inbox was processed)
    # 2 data events + 1 permission file event (syft.pub.yaml from approve_peer_request)
    assert len(do_manager2.datasite_owner_syncer.event_cache.get_cached_events()) == 4

    # verify events in DS cache
    loaded_events_ds = (
        ds_manager2.datasite_watcher_syncer.datasite_watcher_cache.get_cached_events()
    )
    assert len(loaded_events_ds) == 2

    # Verify datasets were loaded
    loaded_datasets = do_manager2.datasets.get_all()
    assert len(loaded_datasets) == 1
    assert loaded_datasets[0].name == "load_state_dataset"
    assert len(do_manager2.datasite_owner_syncer._any_shared_datasets) == 1
    assert (
        do_manager2.datasite_owner_syncer._any_shared_datasets[0][0]
        == "load_state_dataset"
    )


def test_datasets_shared_with_any():
    """Test that datasets shared with 'any' become discoverable after peer approval.

    Unit test equivalent of integration test_datasets_shared_with_any.
    """
    # Create managers WITHOUT auto peer setup
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        add_peers=False,
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
    ds_collections = ds_manager._connection_router.watcher_list_dataset_collections()
    assert any(c["tag"] == "any dataset" for c in ds_collections)


def test_datasets_shared_with_any_after_peer_approved():
    """Test that creating a dataset with users='any' after peers are approved
    grants those peers access immediately.

    Workflow:
    1. Create managers without auto peers
    2. DS adds peer, DO approves
    3. DO creates dataset with users='any'
    4. DS can see the dataset (without needing another approve_peer_request)
    """
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        add_peers=False,
    )

    # DS adds peer, DO approves
    ds_manager.add_peer(do_manager.email)
    do_manager.load_peers()
    do_manager.approve_peer_request(ds_manager.email, peer_must_exist=False)

    mock_dset_path, private_dset_path, readme_path = create_tmp_dataset_files()

    # DO creates dataset with users='any' AFTER peer is already approved
    do_manager.create_dataset(
        name="late any dataset",
        mock_path=mock_dset_path,
        private_path=private_dset_path,
        summary="Dataset shared with anyone, created after peer approval",
        readme_path=readme_path,
        tags=["any", "late"],
        users="any",
    )

    # DS should see the dataset immediately (shared at creation time)
    ds_collections = ds_manager._connection_router.watcher_list_dataset_collections()
    assert any(c["tag"] == "late any dataset" for c in ds_collections)


def test_ds_stale_state_cleared_after_do_delete_syftbox():
    """Test that DS state is cleaned up after DO calls delete_syftbox().

    Verifies:
    1. After initial setup, DS has datasets and file_hashes
    2. DO calls delete_syftbox() which broadcasts is_deleted events
    3. DS syncs and its file_hashes and datasets are empty
    """
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
    )

    # Create dataset on DO side
    mock_dset_path, private_dset_path, readme_path = create_tmp_dataset_files()
    do_manager.create_dataset(
        name="my dataset",
        mock_path=mock_dset_path,
        private_path=private_dset_path,
        summary="Test dataset",
        readme_path=readme_path,
        users=[ds_manager.email],
    )

    # DS sends a file change to DO (use correct job path)
    ds_manager._send_file_change(
        path_for_job(do_manager.email, ds_manager.email), "print('hello')"
    )
    do_manager.sync()

    # DS syncs to get dataset and file events
    ds_manager.sync()

    # Verify initial state exists
    ds_cache = ds_manager.datasite_watcher_syncer.datasite_watcher_cache
    assert len(ds_manager.datasets.get_all()) == 1
    assert len(ds_cache.file_hashes) > 0

    # DO deletes syftbox (broadcasts delete events to DS)
    do_manager.delete_syftbox()

    # DS syncs again - should pick up delete events and stale dataset cleanup
    ds_manager.sync()

    # Verify DS state is cleaned up
    assert len(ds_cache.file_hashes) == 0, (
        "DS file_hashes should be empty after DO delete"
    )
    assert len(ds_manager.datasets.get_all()) == 0, (
        "DS datasets should be empty after DO delete"
    )


def test_incoming_syft_pub_yaml_write_requires_admin():
    """Test that DS cannot write syft.pub.yaml unless they have admin access.

    DS proposes a change to a syft.pub.yaml file in the job folder.
    Even though DS has write access to the job folder, writing syft.pub.yaml
    requires admin, so the change should be rejected.
    """
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
    )

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


def test_dataset_delete_propagates_to_ds():
    """Test that deleting a dataset on DO removes it from Drive and DS."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False
    )

    mock_dset_path, private_dset_path, readme_path = create_tmp_dataset_files()

    # DO creates dataset shared with DS
    do_manager.create_dataset(
        name="deleteme",
        mock_path=mock_dset_path,
        private_path=private_dset_path,
        summary="To be deleted",
        readme_path=readme_path,
        users=[ds_manager.email],
    )

    # Verify collection exists on Drive
    collections = do_manager._connection_router.owner_list_dataset_collections()
    assert "deleteme" in collections

    # DS syncs and sees the dataset
    ds_manager.sync()
    assert len(ds_manager.datasets.get_all()) == 1

    # DO deletes the dataset
    do_manager.delete_dataset(name="deleteme", require_confirmation=False, sync=True)

    # Verify collection is gone from Drive
    collections_after = do_manager._connection_router.owner_list_dataset_collections()
    assert "deleteme" not in collections_after

    # DS syncs again — should pick up the deletion
    ds_manager.sync()
    assert len(ds_manager.datasets.get_all()) == 0
