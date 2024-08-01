# stdlib
from collections.abc import Generator
from collections.abc import Iterable
from itertools import product
import operator
from secrets import token_hex
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

# equivalent to adding this mark to every test in this file
pytestmark = pytest.mark.local_server


@pytest.fixture()
def server_args() -> dict[str, Any]:
    return {}


@pytest.fixture
def server(server_args: dict[str, Any]) -> Generator[ServerHandle, None, None]:
    _server = sy.orchestra.launch(
        **{
            "name": token_hex(8),
            "dev_mode": True,
            "reset": True,
            "n_consumers": 3,
            "create_producer": True,
            "queue_port": None,
            "local_db": False,
            **server_args,
        },
    )
    # startup code here
    yield _server
    # Cleanup code
    _server.python_server.cleanup()
    _server.land()


def matrix(
    *,
    excludes_: Iterable[dict[str, Any]] | None = None,
    **kwargs: Iterable,
) -> list[dict[str, Any]]:
    args = ([(k, v) for v in vs] for k, vs in kwargs.items())
    args = product(*args)

    if excludes_ is None:
        excludes_ = []
    excludes_ = [kv.items() for kv in excludes_]

    args = (
        arg
        for arg in args
        if not any(all(kv in arg for kv in kvs) for kvs in excludes_)
    )

    return [dict(kvs) for kvs in args]


SERVER_ARGS_TEST_CASES = {
    "n_consumers": [1],
    "dev_mode": [True, False],
    "thread_workers": [True, False],
}


class FlakyMark(RuntimeError):
    """To mark a flaky part of a test to use with @pytest.mark.flaky"""



@pytest.mark.flaky(reruns=3, rerun_delay=1, only_rerun=["FlakyMark"])
@pytest.mark.parametrize(
    "server_args",
    matrix(
        **{**SERVER_ARGS_TEST_CASES, "n_consumers": [3]},
    ),
)
@pytest.mark.parametrize("force", [True, False])
def test_delete_idle_worker(
    server: ServerHandle, force: bool, server_args: dict[str, Any],
) -> None:
    client = server.login(email="info@openmined.org", password="changethis")

    original_workers = client.worker.get_all()
    worker_to_delete = max(original_workers, key=operator.attrgetter("name"))

    res = client.worker.delete(worker_to_delete.id, force=force)
    assert not isinstance(res, SyftError)

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
            msg = (
                f"`workers = client.worker.get_all()` failed.\n"
                f"{workers.message=} {server_args=} {[(w.id, w.name) for w in original_workers]}"
            )
            raise FlakyMark(
                msg,
            )

        if len(workers) == len(original_workers) - 1 and all(
            w.id != worker_to_delete.id for w in workers
        ):
            break
        if time.time() - start > 3:
            msg = "Worker did not get removed from stash."
            raise TimeoutError(msg)


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
            msg = "Job did not get picked up by any worker."
            raise TimeoutError(msg)

    res = client.worker.delete(syft_worker_id, force=force)
    assert not isinstance(res, SyftError)

    if not force and len(client.worker.get_all()) > 0:
        assert client.worker.get(syft_worker_id).to_be_deleted
        job.wait()

    job = client.jobs[0]
    if force:
        assert job.status in (JobStatus.COMPLETED, JobStatus.INTERRUPTED)
    else:
        assert job.status == JobStatus.COMPLETED

    start = time.time()
    while True:
        res = client.worker.get(syft_worker_id)
        if isinstance(res, SyftError):
            break
        if time.time() - start > 5:
            msg = "Worker did not get removed from stash."
            raise TimeoutError(msg)

    assert len(client.worker.get_all()) == 0
