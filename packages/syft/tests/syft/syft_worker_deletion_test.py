# stdlib
from collections.abc import Generator
from secrets import token_hex
import time
from typing import Any

# third party
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft.orchestra import NodeHandle
from syft.service.job.job_stash import JobStatus
from syft.service.response import SyftError


@pytest.fixture()
def node_args() -> dict[str, Any]:
    return {}


@pytest.fixture
def node(node_args: dict[str, Any]) -> Generator[NodeHandle, None, None]:
    _node = sy.orchestra.launch(
        **{
            "name": token_hex(8),
            "dev_mode": True,
            "reset": True,
            "n_consumers": 3,
            "create_producer": True,
            "queue_port": None,
            "local_db": False,
            **node_args,
        }
    )
    # startup code here
    yield _node
    # Cleanup code
    _node.python_node.cleanup()
    _node.land()


@pytest.mark.parametrize("node_args", [{"n_consumers": 1}])
@pytest.mark.parametrize("force", [True, False])
def test_delete_idle_worker(node: NodeHandle, force: bool) -> None:
    client = node.login(email="info@openmined.org", password="changethis")
    worker = client.worker.get_all()[0]

    res = client.worker.delete(worker.id, force=force)
    assert not isinstance(res, SyftError)

    if force:
        assert len(client.worker.get_all()) == 0

    start = time.time()
    while True:
        if len(client.worker.get_all()) == 0:
            break
        if time.time() - start > 3:
            raise TimeoutError("Worker did not get removed from stash.")


@pytest.mark.parametrize("node_args", [{"n_consumers": 1}])
@pytest.mark.parametrize("force", [True, False])
def test_delete_worker(node: NodeHandle, force: bool) -> None:
    client = node.login(email="info@openmined.org", password="changethis")

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
            raise TimeoutError("Worker did not get removed from stash.")

    assert len(client.worker.get_all()) == 0
