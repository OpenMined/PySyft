# stdlib

# stdlib
from secrets import token_hex

# third party
import pytest

# syft absolute
import syft as sy
from syft import syft_function
from syft import syft_function_single_use
from syft.service.job.job_service import wait_until
from syft.service.job.job_stash import JobStatus
from syft.service.response import SyftError
from syft.service.response import SyftSuccess


@pytest.mark.local_node
def test_job_restart(job) -> None:
    job.wait(timeout=2)

    assert wait_until(
        lambda: job.fetched_status == JobStatus.PROCESSING
    ), "Job not started"
    assert wait_until(
        lambda: all(
            subjob.fetched_status == JobStatus.PROCESSING for subjob in job.subjobs
        )
    ), "Subjobs not started"

    result = job.subjobs[0].restart()
    assert isinstance(result, SyftError), "Should not restart subjob"

    result = job.restart()
    assert isinstance(result, SyftError), "Should not restart running job"

    result = job.kill()
    assert isinstance(result, SyftSuccess), "Should kill job"
    assert job.fetched_status == JobStatus.INTERRUPTED

    result = job.restart()
    assert isinstance(result, SyftSuccess), "Should restart idle job"

    job.wait(timeout=10)

    assert wait_until(
        lambda: job.fetched_status == JobStatus.PROCESSING
    ), "Job not restarted"
    assert wait_until(
        lambda: len(
            [
                subjob.fetched_status == JobStatus.PROCESSING
                for subjob in job.subjobs
                if subjob.fetched_status != JobStatus.INTERRUPTED
            ]
        )
        == 2
    ), "Subjobs not restarted"


@pytest.fixture
def node():
    node = sy.orchestra.launch(
        name=token_hex(8),
        dev_mode=False,
        thread_workers=False,
        reset=True,
        n_consumers=4,
        create_producer=True,
        node_side_type=sy.NodeSideType.LOW_SIDE,
    )
    try:
        yield node
    finally:
        node.python_node.cleanup()
        node.land()


@pytest.fixture
def job(node):
    client = node.login(email="info@openmined.org", password="changethis")
    _ = client.register(name="a", email="aa@b.org", password="c", password_verify="c")
    ds_client = node.login(email="aa@b.org", password="c")

    @syft_function()
    def process_batch():
        import time  # noqa

        while time.sleep(1) is None:
            ...

    ds_client.code.submit(process_batch)

    @syft_function_single_use()
    def process_all(domain):
        _ = domain.launch_job(process_batch)
        _ = domain.launch_job(process_batch)
        import time  # noqa

        while time.sleep(1) is None:
            ...

    _ = ds_client.code.request_code_execution(process_all)
    client.requests[-1].approve(approve_nested=True)
    client = node.login(email="info@openmined.org", password="changethis")
    job = client.code.process_all(blocking=False)
    try:
        yield job
    finally:
        job.kill()


@pytest.mark.local_node
def test_job_kill(job) -> None:
    job.wait(timeout=2)
    assert wait_until(
        lambda: job.fetched_status == JobStatus.PROCESSING
    ), "Job not started"
    assert wait_until(
        lambda: all(
            subjob.fetched_status == JobStatus.PROCESSING for subjob in job.subjobs
        )
    ), "Subjobs not started"

    result = job.subjobs[0].kill()
    assert isinstance(result, SyftError), "Should not kill subjob"

    result = job.kill()
    assert isinstance(result, SyftSuccess), "Should kill job"
    assert job.fetched_status == JobStatus.INTERRUPTED
