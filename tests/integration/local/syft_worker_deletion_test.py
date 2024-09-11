# stdlib
import operator
import time
from typing import Any

# third party
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft.orchestra import ServerHandle
from syft.service.job.job_stash import JobStatus
from syft.service.response import SyftError
from syft.types.errors import SyftException

# relative
from .conftest import matrix

# equivalent to adding this mark to every test in this file
pytestmark = pytest.mark.local_server


SERVER_ARGS_TEST_CASES = {
    "n_consumers": [1],
    "dev_mode": [True, False],
    "thread_workers": [True, False],
    "create_producer": [True],
}


class FlakyMark(RuntimeError):
    """To mark a flaky part of a test to use with @pytest.mark.flaky"""

    pass


@pytest.mark.flaky(reruns=3, rerun_delay=1, only_rerun=["FlakyMark"])
@pytest.mark.parametrize(
    "server_args",
    matrix(
        **{**SERVER_ARGS_TEST_CASES, "n_consumers": [3]},
    ),
)
@pytest.mark.parametrize("force", [True, False])
def test_delete_idle_worker(
    server: ServerHandle, force: bool, server_args: dict[str, Any]
) -> None:
    client = server.login(email="info@openmined.org", password="changethis")
    original_workers = client.worker.get_all()
    worker_to_delete = max(original_workers, key=operator.attrgetter("name"))

    client.worker.delete(worker_to_delete.id, force=force)

    if force:
        assert (
            len(workers := client.worker.get_all()) == len(original_workers) - 1
            and all(w.id != worker_to_delete.id for w in workers)
        ), f"{workers.message=} {server_args=} {[(w.id, w.name) for w in original_workers]}"
        return

    start = time.time()
    while True:
        workers = client.worker.get_all()
        if isinstance(workers, SyftError):
            raise FlakyMark(
                f"`workers = client.worker.get_all()` failed.\n"
                f"{workers.message=} {server_args=} {[(w.id, w.name) for w in original_workers]}"
            )

        if len(workers) == len(original_workers) - 1 and all(
            w.id != worker_to_delete.id for w in workers
        ):
            break
        if time.time() - start > 3:
            raise TimeoutError("Worker did not get removed from stash.")


@pytest.mark.parametrize("server_args", matrix(**SERVER_ARGS_TEST_CASES))
@pytest.mark.parametrize("force", [True, False])
def test_delete_worker(server: ServerHandle, force: bool) -> None:
    client = server.login(email="info@openmined.org", password="changethis")
    data = np.array([1, 2, 3])
    data_action_obj = sy.ActionObject.from_obj(data)
    data_pointer = data_action_obj.send(client)

    @sy.syft_function_single_use(data=data_pointer)
    def compute_mean(data):
        # stdlib
        import time

        time.sleep(1.5)
        return data.mean()

    client.code.request_code_execution(compute_mean)
    client.requests[-1].approve()

    job = client.code.compute_mean(data=data_pointer, blocking=False)

    start = time.time()
    while True:
        if (syft_worker_id := client.jobs.get_all()[0].job_worker_id) is not None:
            break
        if time.time() - start > 5:
            raise TimeoutError("Job did not get picked up by any worker.")

    client.worker.delete(syft_worker_id, force=force)

    if not force and len(client.worker.get_all()) > 0:
        assert client.worker.get(syft_worker_id).to_be_deleted
        job.wait(timeout=30)

    job = client.jobs[0]
    if force:
        assert job.status in (JobStatus.COMPLETED, JobStatus.INTERRUPTED)
    else:
        assert job.status == JobStatus.COMPLETED

    start = time.time()
    while True:
        try:
            client.worker.get(syft_worker_id)
        except SyftException:
            break
        if time.time() - start > 5:
            raise TimeoutError("Worker did not get removed from stash.")

    assert len(client.worker.get_all()) == 0
