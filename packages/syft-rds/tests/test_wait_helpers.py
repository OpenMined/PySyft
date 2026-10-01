"""wait_until_* helpers: job and dataset matching, and the waits over mock Drive."""

import time
from types import SimpleNamespace

import pytest
from dataset_test_utils import create_tmp_dataset_files
from syft_rds import SyftRDSClient
from syft_rds.waiting import JobEndedError, describe_job, find_dataset, find_job


def _job(
    name="j",
    submitted_by="ds@x",
    datasite="do@x",
    status="pending",
    submitted_at="2026-09-30T10:00:00+00:00",
):
    return SimpleNamespace(
        name=name,
        submitted_by=submitted_by,
        datasite_owner_email=datasite,
        status=status,
        submitted_at=submitted_at,
    )


def _dataset(name="d", owner="do@x"):
    return SimpleNamespace(name=name, owner=owner)


@pytest.fixture
def clock(monkeypatch):
    """Fake monotonic clock; each time.sleep() runs the queued callbacks first."""
    state = {"now": 0.0, "sleeps": [], "on_sleep": []}

    def sleep(seconds):
        state["sleeps"].append(seconds)
        state["now"] += seconds
        if state["on_sleep"]:
            state["on_sleep"].pop(0)()

    monkeypatch.setattr(time, "sleep", sleep)
    monkeypatch.setattr(time, "monotonic", lambda: state["now"])
    return state


# ---------------------------------------------------------------- find_job ---


def test_find_job_returns_only_job_with_name():
    job = _job()
    assert find_job([job, _job(name="other")], "j", None, None) is job


def test_find_job_returns_none_when_no_job_has_name():
    assert find_job([_job(name="other")], "j", None, None) is None


def test_find_job_raises_when_name_has_two_submitters():
    jobs = [_job(submitted_by="a@x"), _job(submitted_by="b@x")]

    with pytest.raises(ValueError, match="2 jobs are named 'j'.*a@x.*b@x.*user_name"):
        find_job(jobs, "j", None, None)


def test_find_job_raises_when_one_name_has_two_datasites():
    jobs = [_job(datasite="do1@x"), _job(datasite="do2@x")]

    with pytest.raises(ValueError, match="2 jobs are named 'j'.*do1@x.*do2@x"):
        find_job(jobs, "j", None, None)


def test_find_job_takes_newest_run_of_same_job():
    # An earlier run with the same submitter and datasite is still on disk.
    old = _job(status="done", submitted_at="2026-09-30T09:00:00+00:00")
    new = _job(status="pending", submitted_at="2026-09-30T11:00:00+00:00")

    assert find_job([old, new], "j", None, None) is new
    assert find_job([new, old], "j", None, {"pending"}) is new
    assert describe_job([old, new], "j", None) == "job 'j' is 'pending'"


def test_find_job_user_name_matches_submitter_or_datasite():
    by_a = _job(submitted_by="a@x", datasite="do@x")
    at_enclave = _job(submitted_by="b@x", datasite="enclave@x")

    assert find_job([by_a, at_enclave], "j", "a@x", None) is by_a
    assert find_job([by_a, at_enclave], "j", "enclave@x", None) is at_enclave


def test_find_job_returns_none_while_status_is_not_final():
    assert find_job([_job(status="running")], "j", None, {"done"}) is None


def test_find_job_accepts_any_of_several_statuses():
    job = _job(status="approved")
    assert find_job([job], "j", None, {"approved", "running"}) is job


def test_find_job_raises_when_job_ended_in_other_status():
    with pytest.raises(JobEndedError, match="'j' ended as 'failed'.*done"):
        find_job([_job(status="failed")], "j", None, {"done"})


def test_find_job_counts_done_as_past_any_earlier_status():
    # Between two polls the job can pass the status waited for and end done.
    job = _job(status="done")

    assert find_job([job], "j", None, {"approved"}) is job
    assert find_job([job], "j", None, {"running", "approved"}) is job


def test_find_job_raises_when_job_failed_after_status():
    with pytest.raises(JobEndedError, match="ended as 'failed'"):
        find_job([_job(status="failed")], "j", None, {"approved"})


def test_find_job_returns_none_while_where_check_is_false():
    job = _job(status="done")

    assert find_job([job], "j", None, {"done"}, where=lambda j: False) is None
    assert find_job([job], "j", None, {"done"}, where=lambda j: True) is job


def test_find_job_raises_on_other_final_status_with_where():
    with pytest.raises(JobEndedError, match="ended as 'rejected'"):
        find_job([_job(status="rejected")], "j", None, {"done"}, where=lambda j: True)


def test_describe_job_names_failed_where_check():
    text = describe_job([_job(status="done")], "j", None, {"done"}, lambda j: False)
    assert text == "job 'j' is 'done', and the where= check is False"


# ------------------------------------------------------------ find_dataset ---


def test_find_dataset_matches_name_and_optional_datasite():
    mine = _dataset(owner="a@x")
    theirs = _dataset(owner="b@x")

    assert find_dataset([mine, _dataset(name="other")], "d", None) is mine
    assert find_dataset([mine, theirs], "d", "b@x") is theirs
    assert find_dataset([mine], "missing", None) is None


def test_find_dataset_raises_when_name_has_two_owners():
    with pytest.raises(ValueError, match="2 datasets are named 'd'.*datasite"):
        find_dataset([_dataset(owner="a@x"), _dataset(owner="b@x")], "d", None)


# ------------------------------------------------------- waits, mock Drive ---


def _pair():
    return SyftRDSClient.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False, sync_automatically=False
    )


def _submit_job(ds, do, tmp_path, name="wait.job"):
    code = tmp_path / "main.py"
    code.write_text('with open("outputs/result.json", "w") as f:\n    f.write("{}")\n')
    ds.submit_python_job(user=do.email, code_path=str(code), job_name=name)


def _run_job(do, name="wait.job"):
    do.sync()
    next(j for j in do.job_client.jobs if j.name == name).approve()
    do.job_runner.process_approved_jobs()
    do.job_runner.share_job_results(name, share_outputs=True, share_logs=False)
    do.sync()


def test_do_waits_until_submitted_job_arrives(tmp_path, clock):
    ds, do = _pair()
    _submit_job(ds, do, tmp_path)

    job = do.wait_until_has_job("wait.job", status="pending")
    assert job.submitted_by == ds.email
    assert clock["sleeps"] == []


def test_ds_waits_until_job_is_done(tmp_path, clock):
    ds, do = _pair()
    _submit_job(ds, do, tmp_path)
    clock["on_sleep"].append(lambda: _run_job(do))

    job = ds.wait_until_has_job("wait.job", user_name=do.email, status="done")
    assert job.status == "done"
    assert clock["sleeps"] == [15]


def test_ds_waits_until_dataset_is_shared(clock):
    ds, do = _pair()
    mock_path, private_path, readme_path = create_tmp_dataset_files()

    def share():
        do.create_dataset(
            name="waited",
            mock_path=mock_path,
            private_path=private_path,
            readme_path=readme_path,
            summary="s",
            users=[ds.email],
        )

    clock["on_sleep"].append(share)

    dataset = ds.wait_until_has_dataset("waited", datasite=do.email)
    assert dataset.owner == do.email
    assert clock["sleeps"] == [15]


def test_ds_waits_until_job_is_done_with_outputs(tmp_path, clock):
    ds, do = _pair()
    _submit_job(ds, do, tmp_path)
    clock["on_sleep"].append(lambda: _run_job(do))

    job = ds.wait_until_has_job(
        "wait.job", status="done", where=lambda j: bool(j.output_paths)
    )
    assert job.output_paths


def test_wait_until_has_job_timeout_names_state(clock):
    ds, _ = _pair()

    with pytest.raises(TimeoutError, match="no job named 'missing'"):
        ds.wait_until_has_job("missing", timeout=30)
    assert clock["sleeps"] == [15, 15]


def test_wait_until_peered_passes_for_live_peer(clock):
    ds, do = _pair()

    assert ds.wait_until_peered(do.email).email == do.email
    assert clock["sleeps"] == []
