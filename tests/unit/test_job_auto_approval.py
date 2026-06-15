"""Test for job auto-approval utility."""

import json
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from syft_bg.api import auto_approve_job
from syft_bg.approve.config import AutoApproveConfig
from syft_bg.common.config import get_default_paths
from syft_client.job_auto_approval import auto_approve_and_run_jobs
from syft_client.sync.syftbox_manager import SyftboxManager


@contextmanager
def _temp_config_paths():
    """Redirect config and auto_approvals_dir to a temp directory."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        original = get_default_paths()
        patched = replace(
            original,
            config=tmp_path / "config.yaml",
            auto_approvals_dir=tmp_path / "auto_approvals",
        )
        with (
            patch("syft_bg.common.config.get_default_paths", return_value=patched),
            patch("syft_bg.approve.config.get_default_paths", return_value=patched),
        ):
            yield patched


def test_auto_approve_and_run_jobs():
    """
    End-to-end test for auto_approve_and_run_jobs.

    Scenario: A job is submitted with a Python script and a JSON data file.
    The utility should only approve and run jobs that have:
    - The exact Python script content (newline agnostic)
    - Exactly the required files (no more, no less)
    """
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )

    # The expected script content (with \n newlines)
    expected_script = """import json
with open("outputs/result.json", "w") as f:
    f.write(json.dumps({"result": 42}))
"""

    # The same script but with \r\n newlines (Windows style) and extra blank lines
    script_with_crlf = 'import json\r\n\r\nwith open("outputs/result.json", "w") as f:\r\n    f.write(json.dumps({"result": 42}))\r\n'

    # Create a project folder with script and data file
    project_dir = Path(tempfile.mkdtemp(prefix="test_auto_approve_"))

    # Create main.py with CRLF newlines (simulating Windows file)
    main_path = project_dir / "main.py"
    main_path.write_text(script_with_crlf)

    # Create data.json
    data_path = project_dir / "data.json"
    data_path.write_text(json.dumps({"input": "test"}))

    # Submit job from DS to DO (folder submission)
    ds_manager.submit_python_job(
        user=do_manager.email,
        code_path=str(project_dir),
        job_name="test_auto_approve.job",
        entrypoint="main.py",
    )

    # Sync to DO
    do_manager.sync()

    # Verify job is in inbox
    assert len(do_manager.jobs) == 1
    assert do_manager.jobs[0].status == "pending"

    # Call auto_approve_and_run_jobs - must specify ALL files including the .py file
    approved = auto_approve_and_run_jobs(
        do_manager,
        required_file_contents={"main.py": expected_script},
        required_file_paths=["main.py", "data.json"],
        verbose=False,
    )

    # Verify job was approved and run
    assert len(approved) == 1

    # Before sharing: DS should not see outputs
    do_manager.sync()
    ds_manager.sync()
    assert len(ds_manager.jobs[-1].output_paths) == 0

    # After sharing: DS should see outputs
    do_manager.job_runner.share_job_results(
        "test_auto_approve.job", share_outputs=True, share_logs=False
    )
    do_manager.sync()
    ds_manager.sync()

    # Verify output
    output_path = ds_manager.jobs[-1].output_paths[0]
    with open(output_path, "r") as f:
        result = json.loads(f.read())

    assert result["result"] == 42


def _submit_job_and_sync(ds_manager, do_manager, project_dir, job_name="test.job"):
    """Helper to submit a job and sync it to the DO."""
    ds_manager.submit_python_job(
        user=do_manager.email,
        code_path=str(project_dir),
        job_name=job_name,
        entrypoint="main.py",
    )
    do_manager.sync()
    return do_manager.jobs[-1]


def _create_project_dir(script_content="print('hello')\n", data_content='{"k": "v"}'):
    """Helper to create a project directory with a .py and .json file."""
    project_dir = Path(tempfile.mkdtemp(prefix="test_auto_approve_job_"))
    (project_dir / "main.py").write_text(script_content)
    (project_dir / "data.json").write_text(data_content)
    return project_dir


def test_auto_approve_job_default_all_content_matched():
    """Default behavior: all files are content-matched."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )
    project_dir = _create_project_dir()
    job = _submit_job_and_sync(ds_manager, do_manager, project_dir)

    with _temp_config_paths():
        result = auto_approve_job(job)
        assert result.success is True

        config = AutoApproveConfig.load()
        obj = config.auto_approvals.objects[job.name]
        content_names = {e.relative_path for e in obj.file_contents}
        assert content_names == {"main.py", "data.json"}
        assert all(e.hash.startswith("sha256:") for e in obj.file_contents)
        assert obj.file_paths == []
        assert obj.peers == [ds_manager.email]


def test_auto_approve_job_file_paths_only():
    """file_paths specified: those are name-only, rest are content-matched."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )
    project_dir = _create_project_dir()
    job = _submit_job_and_sync(ds_manager, do_manager, project_dir)

    with _temp_config_paths():
        result = auto_approve_job(job, file_paths=["data.json"])
        assert result.success is True

        config = AutoApproveConfig.load()
        obj = config.auto_approvals.objects[job.name]
        assert [e.relative_path for e in obj.file_contents] == ["main.py"]
        assert obj.file_paths == ["data.json"]


def test_auto_approve_job_contents_only():
    """contents specified: only those files are content-matched, rest ignored."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )
    project_dir = _create_project_dir()
    job = _submit_job_and_sync(ds_manager, do_manager, project_dir)

    with _temp_config_paths():
        result = auto_approve_job(job, contents=["main.py"])
        assert result.success is True

        config = AutoApproveConfig.load()
        obj = config.auto_approvals.objects[job.name]
        assert [e.relative_path for e in obj.file_contents] == ["main.py"]
        assert obj.file_paths == []


def test_auto_approve_job_both_contents_and_file_paths():
    """Both specified: contents are content-matched, file_paths are name-only."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )
    project_dir = _create_project_dir()
    job = _submit_job_and_sync(ds_manager, do_manager, project_dir)

    with _temp_config_paths():
        result = auto_approve_job(job, contents=["main.py"], file_paths=["data.json"])
        assert result.success is True

        config = AutoApproveConfig.load()
        obj = config.auto_approvals.objects[job.name]
        assert [e.relative_path for e in obj.file_contents] == ["main.py"]
        assert obj.file_paths == ["data.json"]


def test_auto_approve_job_overlap_error():
    """Overlap between contents and file_paths should fail."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )
    project_dir = _create_project_dir()
    job = _submit_job_and_sync(ds_manager, do_manager, project_dir)

    result = auto_approve_job(job, contents=["main.py"], file_paths=["main.py"])
    assert result.success is False
    assert "Overlap" in result.error


def test_auto_approve_job_file_not_found_error():
    """Referencing a non-existent file should fail."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )
    project_dir = _create_project_dir()
    job = _submit_job_and_sync(ds_manager, do_manager, project_dir)

    result = auto_approve_job(job, contents=["nonexistent.py"])
    assert result.success is False
    assert "not found in job" in result.error


def test_auto_approve_job_nested_directory():
    """Files in subdirectories are stored with relative paths."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )
    project_dir = Path(tempfile.mkdtemp(prefix="test_auto_approve_nested_"))
    (project_dir / "main.py").write_text("print('hello')\n")
    subdir = project_dir / "subdir"
    subdir.mkdir()
    (subdir / "helper.py").write_text("def helper(): pass\n")
    (project_dir / "data.json").write_text('{"k": "v"}')

    job = _submit_job_and_sync(ds_manager, do_manager, project_dir)

    with _temp_config_paths():
        result = auto_approve_job(job)
        assert result.success is True

        config = AutoApproveConfig.load()
        obj = config.auto_approvals.objects[job.name]
        entries = {e.relative_path: e for e in obj.file_contents}
        assert set(entries.keys()) == {"main.py", "subdir/helper.py", "data.json"}

        # Verify stored copies match original content
        for entry in entries.values():
            stored_content = Path(entry.path).read_text(encoding="utf-8")
            original = job.code_dir / entry.relative_path
            assert stored_content == original.read_text(encoding="utf-8")


def test_auto_approve_job_default_no_special_treatment_for_non_params_json():
    """Default behavior with non-params.json files: all are content-matched."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )
    project_dir = Path(tempfile.mkdtemp(prefix="test_auto_approve_job_"))
    (project_dir / "main.py").write_text("print('hello')\n")
    (project_dir / "somefile.json").write_text('{"k": "v"}')
    job = _submit_job_and_sync(ds_manager, do_manager, project_dir)

    with _temp_config_paths():
        result = auto_approve_job(job)
        assert result.success is True

        config = AutoApproveConfig.load()
        obj = config.auto_approvals.objects[job.name]
        content_names = {e.relative_path for e in obj.file_contents}
        assert content_names == {"main.py", "somefile.json"}
        assert obj.file_paths == []


def test_auto_approve_job_default_params_json_is_name_only():
    """Default behavior: params.json is automatically name-only."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )
    project_dir = Path(tempfile.mkdtemp(prefix="test_auto_approve_job_"))
    (project_dir / "main.py").write_text("print('hello')\n")
    (project_dir / "params.json").write_text('{"k": "v"}')
    job = _submit_job_and_sync(ds_manager, do_manager, project_dir)

    with _temp_config_paths():
        result = auto_approve_job(job)
        assert result.success is True

        config = AutoApproveConfig.load()
        obj = config.auto_approvals.objects[job.name]
        content_names = {e.relative_path for e in obj.file_contents}
        assert content_names == {"main.py"}
        assert obj.file_paths == ["params.json"]
