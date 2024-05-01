# stdlib
from secrets import token_hex
import sys
from time import sleep

# third party
import pytest
from result import Err

# syft absolute
import syft
import syft as sy
from syft import ActionObject
from syft import syft_function
from syft import syft_function_single_use
from syft.abstract_node import NodeSideType
from syft.client.domain_client import DomainClient
from syft.client.syncing import compare_clients
from syft.client.syncing import resolve_single
from syft.node.worker import Worker
from syft.service.job.job_stash import JobStash
from syft.service.job.job_stash import JobStatus
from syft.service.response import SyftError
from syft.service.response import SyftSuccess


@pytest.mark.local_node
def test_job_kill_restart(full_low_worker) -> None:
    node = full_low_worker

    client = node.login(email="info@openmined.org", password="changethis")
    res = client.register(name="a", email="aa@b.org", password="c", password_verify="c")
    ds_client = node.login(email="aa@b.org", password="c")

    @syft_function()
    def process_batch():
        # stdlib
        import time

        while time.sleep(1) is None:
            ...

    ds_client.code.submit(process_batch)

    @syft_function_single_use()
    def process_all(domain):
        _ = domain.launch_job(process_batch)
        _ = domain.launch_job(process_batch)
        # stdlib
        import time

        while time.sleep(1) is None:
            ...

    r = ds_client.code.request_code_execution(process_all)
    client.requests[-1].approve(approve_nested=True)
    client = node.login(email="info@openmined.org", password="changethis")
    job = client.code.process_all(blocking=False)
    while not job.subjobs:
        sleep(0.5)

    result = job.subjobs[0].kill()
    assert isinstance(result, SyftError), "Should not kill subjob"
    result = job.subjobs[0].restart()
    assert isinstance(result, SyftError), "Should not restart subjob"
    # result = job.restart()
    # assert isinstance(result, SyftSuccess), "Should restart job"
    result = job.kill()
    assert isinstance(result, SyftSuccess), "Should kill job"

    node.python_node.cleanup()
    node.land()
