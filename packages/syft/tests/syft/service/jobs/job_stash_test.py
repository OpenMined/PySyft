# stdlib
from datetime import datetime, timedelta, timezone

# third party
import pytest

# syft absolute
import syft as sy
from syft.service.job.job_stash import Job, JobStatus
from syft.types.uid import UID


@pytest.mark.parametrize(
    "current_iter, n_iters, status, creation_time_delta, expected",
    [
        (0, 10, JobStatus.CREATED, timedelta(hours=2), None),
        (1, None, JobStatus.CREATED, timedelta(hours=2), None),
        (5, 10, JobStatus.PROCESSING, timedelta(hours=2), "24:00s/it"),
        (200000, 200000, JobStatus.COMPLETED, timedelta(hours=2), "<00:00"),
        (156000, 200000, JobStatus.PROCESSING, timedelta(hours=2), "22it/s"),
        (1, 3, JobStatus.PROCESSING, timedelta(hours=2), "2:00:00s/it"),
        (10, 10, JobStatus.PROCESSING, timedelta(minutes=5), "00:30s/it"),
        (0, 10, JobStatus.CREATED, timedelta(days=1), None),
        (10, 100, JobStatus.PROCESSING, timedelta(seconds=3600), "06:00s/it"),
        (100000, 200000, JobStatus.PROCESSING, timedelta(minutes=1), "1667it/s"),
        (2, 10, JobStatus.PROCESSING, timedelta(seconds=119.6), "00:59s/it"),
    ],
)
def test_eta_string(current_iter, n_iters, status, creation_time_delta, expected):
    job = Job(
        id=UID(),
        server_uid=UID(),
        n_iters=n_iters,
        current_iter=current_iter,
        creation_time=(datetime.now(tz=timezone.utc) - creation_time_delta).isoformat(),
        status=status,
    )

    if expected is None:
        assert job.eta_string is None
    else:
        assert job.eta_string is not None
        assert isinstance(job.eta_string, str)
        assert expected in job.eta_string


def test_job_no_consumer(worker):
    client = worker.root_client
    ds_client = worker.guest_client

    @sy.syft_function_single_use()
    def process_all(): ...

    _ = ds_client.code.request_code_execution(process_all)
    job = client.code.process_all(blocking=False)
    res = job.wait()
    assert not res, "Should return error when no consumers are available"
