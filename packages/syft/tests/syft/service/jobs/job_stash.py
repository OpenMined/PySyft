import pytest
from datetime import datetime, timedelta
from syft.service.job.job_stash import Job, JobStatus
from syft.types.uid import UID

def test_job():
    return Job(
        id=UID(),
        node_uid=UID(),
        n_iters=200000,
        current_iter=0,
        creation_time=(datetime.now() - timedelta(hours=2)).isoformat(),
        status=JobStatus.CREATED
    )

@pytest.mark.parametrize("current_iter, n_iters, status, creation_time_delta, expected", [
    (0, 10, JobStatus.CREATED, timedelta(hours=2), None),
    (1, None, JobStatus.CREATED, timedelta(hours=2), None),
    (5, 10, JobStatus.PROCESSING, timedelta(hours=2), "24:00s/it"),
    (200000, 200000, JobStatus.COMPLETED, timedelta(hours=2), None),
    (156000, 200000, JobStatus.PROCESSING, timedelta(hours=2), "00:00s/it"),
    (1, 3, JobStatus.PROCESSING, timedelta(hours=2), "2:00:00s/it"),
    (10, 10, JobStatus.PROCESSING, timedelta(minutes=5), "00:30s/it"),
    (0, 10, JobStatus.CREATED, timedelta(days=1), None),
    (10, 100, JobStatus.PROCESSING, timedelta(seconds=3600), "06:00s/it"),
    (100000, 200000, JobStatus.PROCESSING, timedelta(minutes=1), "00:00s/it"),
])
def test_eta_string(current_iter, n_iters, status, creation_time_delta, expected):
    job = test_job()
    job.current_iter = current_iter
    job.n_iters = n_iters
    job.status = status
    job.creation_time = (datetime.now() - creation_time_delta).isoformat()

    if expected is None:
        assert job.eta_string is None
    else:
        print(f"job ({current_iter}, {n_iters}, {status}, {expected}) {job.eta_string}")
        assert job.eta_string is not None
        assert isinstance(job.eta_string, str)
        assert expected in job.eta_string
