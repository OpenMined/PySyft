"""Test for job auto-approval utility."""

import json
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from syft_bg.api import auto_approve_job
from syft_bg.approve.api_store import ApiStore
from syft_bg.common.config import get_default_paths
from syft_rds import SyftRDSClient
from syft_rds.job_auto_approval import auto_approve_and_run_jobs


@contextmanager
def _temp_config_paths():
    """Redirect config to a temp directory."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        original = get_default_paths()
        patched = replace(
            original,
            config=tmp_path / "config.yaml",
        )
        with (
            patch("syft_bg.common.config.get_default_paths", return_value=patched),
            patch("syft_bg.api.api.get_default_paths", return_value=patched),
            patch("syft_bg.api.utils.get_default_paths", return_value=patched),
            patch("syft_bg.approve.config.get_default_paths", return_value=patched),
            patch(
                "syft_bg.common.syft_bg_config.get_default_paths", return_value=patched
            ),
        ):
            yield patched


def _api_store(do_manager: SyftRDSClient) -> ApiStore:
    return ApiStore(do_manager.syftbox_folder, do_manager.email)


def test_auto_approve_and_run_jobs():
    """
    End-to-end test for auto_approve_and_run_jobs.

    Scenario: A job is submitted with a Python script and a JSON data file.
    The utility should only approve and run jobs that have:
    - The exact Python script content (newline agnostic)
    - Exactly the required files (no more, no less)
    """
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
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

    # Criteria name every file under the submission root, and pin the content of
    # run.sh, which is the only file the runner executes. The owner takes that
    # script from a job it has reviewed.
    reviewed_run_script = do_manager.jobs[0].run_script
    criteria = dict(
        required_file_contents={
            "code/main.py": expected_script,
            "run.sh": reviewed_run_script,
        },
        required_file_paths=["code/main.py", "code/data.json", "run.sh", "config.yaml"],
    )

    # A job whose run.sh is not the reviewed one is not approved.
    assert (
        auto_approve_and_run_jobs(
            do_manager,
            required_file_contents={
                **criteria["required_file_contents"],
                "run.sh": "#!/bin/bash\necho something else\n",
            },
            required_file_paths=criteria["required_file_paths"],
            verbose=False,
        )
        == []
    )

    approved = auto_approve_and_run_jobs(do_manager, verbose=False, **criteria)

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
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )
    project_dir = _create_project_dir()
    job = _submit_job_and_sync(ds_manager, do_manager, project_dir)

    with _temp_config_paths():
        result = auto_approve_job(job)
        assert result.success is True

        obj = _api_store(do_manager).get(job.name)
        content_names = {e.relative_path for e in obj.file_contents}
        # run.sh is pinned: it is the file the runner executes. config.yaml is
        # name-only, because it carries per-job metadata.
        assert content_names == {"code/main.py", "code/data.json", "run.sh"}
        assert all(e.hash.startswith("sha256:") for e in obj.file_contents)
        assert obj.file_paths == ["config.yaml"]
        assert obj.peers == [ds_manager.email]


def test_auto_approve_job_file_paths_only():
    """file_paths specified: those are name-only, rest are content-matched."""
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )
    project_dir = _create_project_dir()
    job = _submit_job_and_sync(ds_manager, do_manager, project_dir)

    with _temp_config_paths():
        result = auto_approve_job(job, file_paths=["data.json"])
        assert result.success is True

        obj = _api_store(do_manager).get(job.name)
        # config.yaml is name-only in every branch: its bytes carry the job
        # name and the submission time, so pinning them matches one job.
        assert sorted(e.relative_path for e in obj.file_contents) == [
            "code/main.py",
            "run.sh",
        ]
        assert sorted(obj.file_paths) == ["code/data.json", "config.yaml"]


def test_auto_approve_job_contents_only_refuses_without_run_sh():
    """An object that pins no run.sh approves nothing, so it is not written."""
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )
    project_dir = _create_project_dir()
    job = _submit_job_and_sync(ds_manager, do_manager, project_dir)

    with _temp_config_paths():
        result = auto_approve_job(job, contents=["main.py"])
        assert result.success is False
        assert "run.sh" in result.error
        assert not _api_store(do_manager).exists(job.name)

        # Naming run.sh is not enough on its own: a file the caller places in
        # neither bucket would leave the object matching nothing.
        result = auto_approve_job(job, contents=["main.py", "run.sh"])
        assert result.success is False
        assert "code/data.json" in result.error
        assert "config.yaml" in result.error


def test_auto_approve_job_both_contents_and_file_paths():
    """Both specified: the caller places every file, or gets told which it missed."""
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )
    project_dir = _create_project_dir()
    job = _submit_job_and_sync(ds_manager, do_manager, project_dir)

    with _temp_config_paths():
        result = auto_approve_job(job, contents=["main.py"], file_paths=["data.json"])
        assert result.success is False
        assert "run.sh" in result.error

        result = auto_approve_job(
            job,
            contents=["main.py", "run.sh"],
            file_paths=["data.json", "config.yaml"],
        )
        assert result.success is True

        obj = _api_store(do_manager).get(job.name)
        assert sorted(e.relative_path for e in obj.file_contents) == [
            "code/main.py",
            "run.sh",
        ]
        assert sorted(obj.file_paths) == ["code/data.json", "config.yaml"]


def test_auto_approve_job_overlap_error():
    """Overlap between contents and file_paths should fail."""
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
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
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
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
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
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

        obj = _api_store(do_manager).get(job.name)
        entries = {e.relative_path: e for e in obj.file_contents}
        assert set(entries.keys()) == {
            "code/main.py",
            "code/subdir/helper.py",
            "code/data.json",
            "run.sh",
        }

        # Verify stored copies match original content
        for entry in entries.values():
            stored_content = Path(entry.path).read_text(encoding="utf-8")
            original = job.job_submission_path / entry.relative_path
            assert stored_content == original.read_text(encoding="utf-8")


def test_auto_approve_job_default_no_special_treatment_for_non_params_json():
    """Default behavior with non-params.json files: all are content-matched."""
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
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

        obj = _api_store(do_manager).get(job.name)
        content_names = {e.relative_path for e in obj.file_contents}
        assert content_names == {"code/main.py", "code/somefile.json", "run.sh"}
        assert obj.file_paths == ["config.yaml"]


def test_auto_approve_job_default_params_json_is_name_only():
    """Default behavior: params.json is automatically name-only."""
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
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

        obj = _api_store(do_manager).get(job.name)
        content_names = {e.relative_path for e in obj.file_contents}
        assert content_names == {"code/main.py", "run.sh"}
        assert sorted(obj.file_paths) == ["code/params.json", "config.yaml"]


def test_auto_approve_job_api_syncs_to_submitter_only():
    """The api lands in the DO's app_data/apis and syncs to its peers only."""
    ds_manager, do_manager = SyftRDSClient.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )
    job = _submit_job_and_sync(ds_manager, do_manager, _create_project_dir())

    with _temp_config_paths():
        assert auto_approve_job(job).success is True
        assert auto_approve_job(job, peers=["other@test.com"], name="other").success

    do_manager.sync()
    ds_manager.sync()

    ds_apis_dir = Path(ds_manager.syftbox_folder) / do_manager.email / "app_data/apis"
    assert (ds_apis_dir / job.name / "api.yaml").exists()
    assert (ds_apis_dir / job.name / "files" / "run.sh").exists()
    assert not (ds_apis_dir / "other").exists()
