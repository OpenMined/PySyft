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


def _make_job_info(
    name: str,
    status: str = "pending",
    datasite_owner_email: str = "do@test.com",
    submitted_by: str = "ds@test.com",
    current_user_email: str = "do@test.com",
) -> JobInfo:
    """Create a minimal JobInfo for testing."""
    from datetime import datetime, timezone
    from pathlib import Path

    from syft_job.client import JobClient
    from syft_job.config import SyftJobConfig
    from syft_job.job_ref import JobRef
    from syft_job.models import JobState, JobStatus, JobSubmissionMetadata

    config = SyftJobConfig(
        syftbox_folder=Path("/tmp/fake"), current_user_email=current_user_email
    )
    client = JobClient(config=config)
    submission_config = JobSubmissionMetadata(
        name=name,
        type="python",
        submitted_by=submitted_by,
        datasite_email=datasite_owner_email,
        submitted_at=datetime.now(timezone.utc),
    )
    state = JobState(status=JobStatus(status))
    # Identity (owner, submitter, name) comes from the path-derived ref.
    ref = JobRef(
        datasite_email=datasite_owner_email,
        ds_email=submitted_by,
        job_name=name,
        protocol_version="1",
    )
    return JobInfo(
        job_metadata=submission_config,
        state=state,
        client=client,
        current_user_email=current_user_email,
        ref=ref,
    )


def test_jobs_list_getitem_int():
    jobs = JobsList([_make_job_info("job-a"), _make_job_info("job-b")])
    assert jobs[0].name == "job-a"
    assert jobs[1].name == "job-b"


def test_jobs_list_getitem_str():
    jobs = JobsList([_make_job_info("job-a"), _make_job_info("job-b")])
    assert jobs["job-b"].name == "job-b"


def test_jobs_list_getitem_str_not_found():
    jobs = JobsList([_make_job_info("job-a")])
    with pytest.raises(ValueError, match="not found"):
        jobs["nonexistent"]


def test_jobs_list_getitem_invalid_type():
    jobs = JobsList([_make_job_info("job-a")])
    with pytest.raises(TypeError):
        jobs[3.14]  # type: ignore[arg-type]


# --- JobsList summary rendering (__str__ / _repr_html_ / __repr__) ---


def _ds_jobs(*statuses: str) -> JobsList:
    """Outgoing jobs submitted by us (ds@test.com) to do@org.com, one per status."""
    jobs = [
        _make_job_info(
            f"Job {i}",
            status=status,
            datasite_owner_email="do@org.com",
            submitted_by="ds@test.com",
            current_user_email="ds@test.com",
        )
        for i, status in enumerate(statuses)
    ]
    return JobsList(jobs)


def test_jobs_list_summary_header_and_rows():
    text = str(_ds_jobs("pending", "done", "running"))
    assert text.startswith("📋 Your jobs (3):")
    assert "Job 0" in text and "Job 1" in text and "Job 2" in text
    assert "⏳ inbox" in text
    assert "✅ done" in text
    assert "🔄 running" in text
    assert "📤" in text
    assert "submitted to: do@org.com" in text


def test_jobs_list_summary_done_tip_uses_first_done_index():
    text = str(_ds_jobs("pending", "done", "done"))
    assert "💡 Job 1 is done" in text
    assert "for path in client.jobs[1].output_paths:" in text
    assert "print(open(path).read())" in text
    assert "waiting for the data owner" not in text


def test_jobs_list_summary_all_inbox_tip():
    text = str(_ds_jobs("received", "pending"))
    assert "waiting for the data owner" in text
    assert "client.sync()" in text
    assert "is done" not in text


def test_jobs_list_summary_no_tip_when_mixed_without_done():
    text = str(_ds_jobs("pending", "running"))
    assert "is done" not in text
    assert "waiting for the data owner" not in text


def test_jobs_list_summary_empty():
    assert str(JobsList([])) == "📭 No jobs found.\n"


def test_jobs_list_summary_incoming_do_view():
    # We are the DO; job came from someone else.
    job = _make_job_info(
        "incoming-job",
        status="pending",
        datasite_owner_email="do@test.com",
        submitted_by="alice@ds.com",
        current_user_email="do@test.com",
    )
    text = str(JobsList([job]))
    assert "📥" in text
    assert "from: alice@ds.com" in text
    assert "submitted to:" not in text


def test_jobs_list_repr_html_summary():
    html = _ds_jobs("pending", "done")._repr_html_()
    assert "syftjob-overview" in html
    assert "Your jobs (2)" in html
    assert "⏳ inbox" in html
    assert "✅ done" in html
    assert "client.jobs[1].output_paths" in html


def test_jobs_list_repr_preserves_technical_string():
    assert repr(_ds_jobs("pending", "done", "running")) == "JobsList(3 jobs)"
