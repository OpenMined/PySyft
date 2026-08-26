"""Dataset and job flows over the mock drive connection."""

from pathlib import Path
import json
import shutil
import tempfile

import pytest

from syft.sync.connections.drive import mock_drive_service
from syft.sync.connections.drive.gdrive_transport import GDriveConnection
from syft_datasets import Dataset
from syft_datasets.dataset_manager import DATASET_COLLECTION_PREFIX
from syft_rds import SyftRDSClient
from syft_rds.config import COLLECTION_SUBPATH, SyftRDSClientConfig
from dataset_test_utils import (
    create_test_project_folder,
    create_tmp_dataset_files,
    create_tmp_dataset_files_with_parquet,
)


def path_for_job(do_email: str, ds_email: str, filename: str = "test.job") -> str:
    """Return the correct path for DS to submit a job file to DO."""
    return f"{do_email}/app_data/job/inbox/{ds_email}/{filename}"


def test_datasets():
    """Test dataset creation and sync between DO and DS."""
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
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
    collections = do_manager.sync_engine._connection_router.owner_list_collections(
        DATASET_COLLECTION_PREFIX
    )
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
    ds_collections = ds_manager.sync_engine._connection_router.watcher_list_collections(
        DATASET_COLLECTION_PREFIX
    )
    assert any(c["tag"] == "my dataset" for c in ds_collections)

    assert len(ds_manager.datasets.get_all()) == 1

    dataset_ds = ds_manager.datasets.get("my dataset", datasite=do_manager.email)

    assert dataset_ds.mock_files[0].exists()

    mock_content_ds = (dataset_ds.mock_dir / "mock.txt").read_text()
    assert len(mock_content_ds) > 0

    # test getting it via resolve path
    from syft import resolve_dataset_file_path

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

    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
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
    cached_events = (
        do_manager.sync_engine.datasite_owner_syncer.event_cache.get_cached_events()
    )
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
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
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
    collections = do_manager.sync_engine._connection_router.owner_list_collections(
        DATASET_COLLECTION_PREFIX
    )
    assert "private dataset" in collections

    # DO should be able to see their own dataset
    datasets = do_manager.datasets.get_all()
    assert len(datasets) == 1

    # DS syncs
    ds_manager.sync()

    # DS should NOT see the collection (no permissions)
    ds_collections = ds_manager.sync_engine._connection_router.watcher_list_collections(
        DATASET_COLLECTION_PREFIX
    )
    assert not any(c["tag"] == "private dataset" for c in ds_collections)

    # DS should not have downloaded any datasets
    assert len(ds_manager.datasets.get_all()) == 0


def test_dataset_only_mock_data_uploaded():
    """Test that only mock data is uploaded to the collection, not private data."""
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
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

    files = ds_manager.sync_engine._connection_router.connections[
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
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
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
    connection_do = do_manager.sync_engine._connection_router.connections[0]
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
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
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
    import syft as sy

    assert (
        sy.resolve_dataset_file_path("my dataset", client=ds_manager)
        == dataset_ds.mock_files[0]
    )

    test_py_path = "/tmp/test.py"
    with open(test_py_path, "w") as f:
        f.write("""
import syft as sy
import json

data_path = sy.resolve_dataset_file_path("my dataset")
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
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
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
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
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
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
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
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
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
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
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
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
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
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
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
import syft as sy

data_path = sy.resolve_dataset_file_path("my dataset")

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
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
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
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
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
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
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
    collections = ds_manager.sync_engine._connection_router.watcher_list_collections(
        DATASET_COLLECTION_PREFIX
    )
    remote_hash = None
    for c in collections:
        if c["tag"] == "cached dataset":
            remote_hash = c["content_hash"]
            break
    assert remote_hash is not None

    # Get the mock backing store and directories for creating second pair
    mock_backing_store = ds_manager.sync_engine._connection_router.connections[
        0
    ].drive_service._backing_store
    ds_folder = ds_manager.syftbox_folder
    do_folder = do_manager.syftbox_folder
    ds_email = ds_manager.email
    do_email = do_manager.email

    # Create second pair with same directories (simulates restart)
    ds_manager2, do_manager2 = SyftRDSClient.pair_with_mock_drive_service_connection(
        email1=do_email,
        email2=ds_email,
        base_path1=do_folder,
        base_path2=ds_folder,
        use_in_memory_cache=False,
        add_peers=False,
    )

    # Replace mock backing store to share dataset collections
    ds_manager2.sync_engine._connection_router.connections[
        0
    ].drive_service._backing_store = mock_backing_store
    do_manager2.sync_engine._connection_router.connections[
        0
    ].drive_service._backing_store = mock_backing_store

    # Load peers (already approved in shared backing store)
    ds_manager2.load_peers()
    do_manager2.load_peers()

    # Verify hash was loaded from disk on startup
    ds_cache = ds_manager2.sync_engine.datasite_watcher_syncer.datasite_watcher_cache
    # Cache uses full path as key: syftbox_folder / owner_email / collection_subpath / tag
    cache_key = ds_cache.get_collection_path(
        do_email, "cached dataset", COLLECTION_SUBPATH
    )
    assert cache_key in ds_cache.collection_hashes, (
        "Hash should be loaded from disk on startup"
    )

    # Verify the loaded hash matches the remote hash
    # This ensures sync_down_datasets will skip the download (hash comparison at line 283-284)
    loaded_hash = ds_cache.collection_hashes[cache_key]
    assert loaded_hash == remote_hash, (
        "Loaded hash should match remote hash, ensuring no re-download is needed"
    )

    # Patch the download method to verify it's NOT called (hash match should skip download)
    syncer = ds_manager2.sync_engine.datasite_watcher_syncer
    original_method = syncer.download_collection_file_with_new_connection

    with patch(
        "syft.sync.sync.datasite_watcher_syncer.DatasiteWatcherSyncer.download_collection_file_with_new_connection",
        wraps=original_method,
    ) as mock_download:
        # Sync - no download should happen because hash matches
        ds_manager2.sync()

        # Verify download_collection_file_with_new_connection was NOT called
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
    3. Verify _download_collections_parallel is NOT called on sync
    """
    from unittest.mock import patch

    # Create first pair
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
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
    mock_backing_store = ds_manager.sync_engine._connection_router.connections[
        0
    ].drive_service._backing_store
    ds_folder = ds_manager.syftbox_folder
    do_folder = do_manager.syftbox_folder
    ds_email = ds_manager.email
    do_email = do_manager.email

    # Create second pair with same directories (simulates restart)
    ds_manager2, do_manager2 = SyftRDSClient.pair_with_mock_drive_service_connection(
        email1=do_email,
        email2=ds_email,
        base_path1=do_folder,
        base_path2=ds_folder,
        use_in_memory_cache=False,
        add_peers=False,
    )

    # Replace mock backing store to share dataset collections
    ds_manager2.sync_engine._connection_router.connections[
        0
    ].drive_service._backing_store = mock_backing_store
    do_manager2.sync_engine._connection_router.connections[
        0
    ].drive_service._backing_store = mock_backing_store

    # Load peers (already approved in shared backing store)
    ds_manager2.load_peers()
    do_manager2.load_peers()

    # Patch the download method to verify it's NOT called (hash match should skip download)
    syncer = do_manager2.sync_engine.datasite_owner_syncer
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


def test_in_memory_connection_load_state():
    """Test state persistence and loading with mock drive connection.

    Unit test equivalent of integration test_google_drive_connection_load_state.

    Workflow (matches integration test):
    1. Pair 1: Create peers, make changes, create dataset
    2. Pair 2: Load peers, sync DO → verify events processed
    3. Pair 3: Load peers, sync both → verify state loaded from storage
    """

    # Get shared backing store and directories that will persist across pairs
    ds_manager1, do_manager1 = SyftRDSClient.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        add_peers=True,
    )

    # Get the backing store that will persist across "restarts"
    backing_store = do_manager1.sync_engine._connection_router.connections[
        0
    ].drive_service._backing_store
    ds_folder = ds_manager1.syftbox_folder
    do_folder = do_manager1.syftbox_folder
    ds_email = ds_manager1.email
    do_email = do_manager1.email
    ds_manager1.sync_engine._connection_router.connections[0]
    do_manager1.sync_engine._connection_router.connections[0]

    # Make some changes (submit to DS's job folder where DS has write access)
    ds_manager1.sync_engine._send_file_change(
        path_for_job(do_manager1.email, ds_manager1.email, "my.job"), "Hello, world!"
    )
    ds_manager1.sync_engine._send_file_change(
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
    assert (
        len(
            do_manager1.sync_engine._connection_router.owner_list_collections(
                DATASET_COLLECTION_PREFIX
            )
        )
        == 1
    )
    assert (
        len(do_manager1.sync_engine.datasite_owner_syncer.any_shared_collections) == 1
    )

    # Create second pair (simulates restart, tests loading peers and processing inbox)
    do_config2 = SyftRDSClientConfig._base_config_for_testing(
        email=do_email,
        syftbox_folder=do_folder,
        has_ds_role=False,
        has_do_role=True,
        use_in_memory_cache=False,
        check_versions=False,
    )
    ds_config2 = SyftRDSClientConfig._base_config_for_testing(
        email=ds_email,
        syftbox_folder=ds_folder,
        has_ds_role=True,
        has_do_role=False,
        use_in_memory_cache=False,
        check_versions=False,
    )

    do_manager2 = SyftRDSClient.from_config(do_config2)
    ds_manager2 = SyftRDSClient.from_config(ds_config2)

    # Connect to the same backing store
    service_do = mock_drive_service.MockDriveService(backing_store, do_email)
    do_connection2 = GDriveConnection.from_service(do_email, service_do)

    service_ds = mock_drive_service.MockDriveService(backing_store, ds_email)
    ds_connection2 = GDriveConnection.from_service(ds_email, service_ds)

    do_manager2.sync_engine._add_connection(do_connection2)
    ds_manager2.sync_engine._add_connection(ds_connection2)

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
    assert (
        len(
            do_manager2.sync_engine.datasite_owner_syncer.event_cache.get_cached_events()
        )
        == 4
    )

    # verify events in DS cache
    loaded_events_ds = ds_manager2.sync_engine.datasite_watcher_syncer.datasite_watcher_cache.get_cached_events()
    assert len(loaded_events_ds) == 2

    # Verify datasets were loaded
    loaded_datasets = do_manager2.datasets.get_all()
    assert len(loaded_datasets) == 1
    assert loaded_datasets[0].name == "load_state_dataset"
    assert (
        len(do_manager2.sync_engine.datasite_owner_syncer.any_shared_collections) == 1
    )
    assert (
        do_manager2.sync_engine.datasite_owner_syncer.any_shared_collections[0][0]
        == "load_state_dataset"
    )


def test_datasets_shared_with_any():
    """Test that datasets shared with 'any' become discoverable after peer approval.

    Unit test equivalent of integration test_datasets_shared_with_any.
    """
    # Create managers WITHOUT auto peer setup
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
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
    ds_collections = ds_manager.sync_engine._connection_router.watcher_list_collections(
        DATASET_COLLECTION_PREFIX
    )
    assert not any(c["tag"] == "any dataset" for c in ds_collections)

    # DS adds peer, DO approves (this should share 'any' datasets)
    ds_manager.add_peer(do_manager.email)
    do_manager.load_peers()
    do_manager.approve_peer_request(ds_manager.email, peer_must_exist=False)

    # DS should now see the dataset
    ds_collections = ds_manager.sync_engine._connection_router.watcher_list_collections(
        DATASET_COLLECTION_PREFIX
    )
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
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
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
    ds_collections = ds_manager.sync_engine._connection_router.watcher_list_collections(
        DATASET_COLLECTION_PREFIX
    )
    assert any(c["tag"] == "late any dataset" for c in ds_collections)


def test_ds_stale_state_cleared_after_do_delete_syftbox():
    """Test that DS state is cleaned up after DO calls delete_syftbox().

    Verifies:
    1. After initial setup, DS has datasets and file_hashes
    2. DO calls delete_syftbox() which broadcasts is_deleted events
    3. DS syncs and its file_hashes and datasets are empty
    """
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
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
    ds_manager.sync_engine._send_file_change(
        path_for_job(do_manager.email, ds_manager.email), "print('hello')"
    )
    do_manager.sync()

    # DS syncs to get dataset and file events
    ds_manager.sync()

    # Verify initial state exists
    ds_cache = ds_manager.sync_engine.datasite_watcher_syncer.datasite_watcher_cache
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


def test_dataset_delete_propagates_to_ds():
    """Test that deleting a dataset on DO removes it from Drive and DS."""
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
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
    collections = do_manager.sync_engine._connection_router.owner_list_collections(
        DATASET_COLLECTION_PREFIX
    )
    assert "deleteme" in collections

    # DS syncs and sees the dataset
    ds_manager.sync()
    assert len(ds_manager.datasets.get_all()) == 1

    # DO deletes the dataset
    do_manager.delete_dataset(name="deleteme", require_confirmation=False, sync=True)

    # Verify collection is gone from Drive
    collections_after = (
        do_manager.sync_engine._connection_router.owner_list_collections(
            DATASET_COLLECTION_PREFIX
        )
    )
    assert "deleteme" not in collections_after

    # DS syncs again — should pick up the deletion
    ds_manager.sync()
    assert len(ds_manager.datasets.get_all()) == 0
