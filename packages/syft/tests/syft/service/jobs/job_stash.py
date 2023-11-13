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

def test_no_iterations():
    job = test_job()
    job.current_iter = 0
    job.n_iters = 10

    assert job.eta_string is None

def test_no_total_iterations():
    job = test_job()
    job.current_iter = 1
    job.n_iters = None

    assert job.eta_string is None

def test_completed_job():
    job = test_job()
    job.status = JobStatus.COMPLETED

    assert job.eta_string is None

def test_valid_iterations():
    job = test_job()
    job.current_iter = 5
    job.n_iters = 10

    assert job.eta_string is not None
    assert isinstance(job.eta_string, str)
    assert "24:00s/it" in job.eta_string

def test_rounding_accuracy():
    job = test_job()
    job.current_iter = 156000
    job.n_iters = 200000

    assert job.eta_string is not None
    assert isinstance(job.eta_string, str)
    assert "00:00s/it" in job.eta_string
    assert "2:00:00<33:50" in job.eta_string

def test_rounding_accuracy_2():
    job = test_job()
    job.current_iter = 1
    job.n_iters = 3

    assert job.eta_string is not None
    assert isinstance(job.eta_string, str)
    assert "2:00:00s/it" in job.eta_string
    assert "2:00:00<4:00:00" in job.eta_string
