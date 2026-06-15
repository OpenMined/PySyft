"""Tests for SyftDatasetManager and JobsList repr and indexing."""

import pytest

from syft_client.sync.syftbox_manager import SyftboxManager
from syft_job.job import JobInfo, JobsList
from tests.unit.utils import create_tmp_dataset_files


def _create_manager_with_dataset():
    """Create a pair of managers and a dataset, return (ds_manager, do_manager, dataset)."""
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )
    mock_path, private_path, readme_path = create_tmp_dataset_files()
    dataset = do_manager.create_dataset(
        name="test-dataset",
        mock_path=mock_path,
        private_path=private_path,
        summary="A test dataset",
    )
    return ds_manager, do_manager, dataset


# --- SyftDatasetManager tests ---


def test_dataset_manager_getitem_str():
    _, do_manager, dataset = _create_manager_with_dataset()
    result = do_manager.datasets["test-dataset"]
    assert result.name == "test-dataset"


def test_dataset_manager_getitem_int():
    _, do_manager, dataset = _create_manager_with_dataset()
    result = do_manager.datasets[0]
    assert result.name == "test-dataset"


def test_dataset_manager_getitem_int_out_of_range():
    _, do_manager, _ = _create_manager_with_dataset()
    with pytest.raises(IndexError):
        do_manager.datasets[99]


def test_dataset_manager_len():
    _, do_manager, _ = _create_manager_with_dataset()
    assert len(do_manager.datasets) == 1


def test_dataset_manager_iter():
    _, do_manager, _ = _create_manager_with_dataset()
    datasets = list(do_manager.datasets)
    assert len(datasets) == 1
    assert datasets[0].name == "test-dataset"


def test_dataset_manager_repr():
    _, do_manager, _ = _create_manager_with_dataset()
    r = repr(do_manager.datasets)
    assert "SyftDatasetManager" in r
    assert "1 datasets" in r


def test_dataset_manager_repr_html():
    _, do_manager, _ = _create_manager_with_dataset()
    html = do_manager.datasets._repr_html_()
    assert html is not None
    assert "📦 Available datasets (1)" in html
    assert "test-dataset" in html
    assert f"from: {do_manager.email}" in html
    assert "1 mock file" in html
    assert 'client.datasets.get("test-dataset"' in html
    assert "dataset.mock_files[0].read_text()" in html


def test_dataset_manager_repr_html_with_tags():
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )
    mock_path, private_path, _ = create_tmp_dataset_files()
    do_manager.create_dataset(
        name="tagged-dataset",
        mock_path=mock_path,
        private_path=private_path,
        tags=["ocean", "water-quality"],
    )
    html = do_manager.datasets._repr_html_()
    assert "[ocean, water-quality]" in html


def test_dataset_manager_repr_html_empty():
    _, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )
    html = do_manager.datasets._repr_html_()
    assert "📦 No datasets available yet." in html
    assert "client.sync()" in html
    assert "haven't created any datasets yet" in html
    assert "not connected to any peers yet" in html


def test_dataset_manager_get_missing_lists_available():
    _, do_manager, _ = _create_manager_with_dataset()
    with pytest.raises(FileNotFoundError) as excinfo:
        do_manager.datasets.get("nope")
    msg = str(excinfo.value)
    assert "❌" in msg
    assert "'nope'" in msg
    assert "client.sync()" in msg
    assert "Available datasets:" in msg
    assert "test-dataset" in msg


def test_dataset_manager_get_missing_no_datasets():
    _, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
        sync_automatically=False,
    )
    with pytest.raises(FileNotFoundError) as excinfo:
        do_manager.datasets.get("nope")
    assert "(none found — check your peer connections)" in str(excinfo.value)


def test_dataset_repr_html_mentions_mock_files():
    _, _, dataset = _create_manager_with_dataset()
    html = dataset._repr_html_()
    assert ".mock_files" in html


# --- JobsList tests ---


def _make_job_info(name: str, status: str = "pending") -> JobInfo:
    """Create a minimal JobInfo for testing."""
    from datetime import datetime, timezone
    from pathlib import Path

    from syft_job.client import JobClient
    from syft_job.config import SyftJobConfig
    from syft_job.models.config import JobSubmissionMetadata
    from syft_job.models.state import JobState, JobStatus

    config = SyftJobConfig(
        syftbox_folder=Path("/tmp/fake"), current_user_email="test@test.com"
    )
    client = JobClient(config=config)
    submission_config = JobSubmissionMetadata(
        name=name,
        type="python",
        submitted_by="ds@test.com",
        datasite_email="ds@test.com",
        submitted_at=datetime.now(timezone.utc),
    )
    state = JobState(status=JobStatus(status))
    return JobInfo(
        job_metadata=submission_config,
        state=state,
        datasite_owner_email="test@test.com",
        client=client,
        current_user_email="test@test.com",
    )


def test_jobs_list_getitem_int():
    jobs = JobsList(
        [_make_job_info("job-a"), _make_job_info("job-b")],
        root_email="test@test.com",
    )
    assert jobs[0].name == "job-a"
    assert jobs[1].name == "job-b"


def test_jobs_list_getitem_str():
    jobs = JobsList(
        [_make_job_info("job-a"), _make_job_info("job-b")],
        root_email="test@test.com",
    )
    assert jobs["job-b"].name == "job-b"


def test_jobs_list_getitem_str_not_found():
    jobs = JobsList(
        [_make_job_info("job-a")],
        root_email="test@test.com",
    )
    with pytest.raises(ValueError, match="not found"):
        jobs["nonexistent"]


def test_jobs_list_getitem_invalid_type():
    jobs = JobsList(
        [_make_job_info("job-a")],
        root_email="test@test.com",
    )
    with pytest.raises(TypeError):
        jobs[3.14]
