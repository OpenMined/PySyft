# stdlib
from secrets import token_hex
import sys

# third party
import pytest
from result import Err

# syft absolute
import syft
import syft as sy
from syft.abstract_node import NodeSideType
from syft.client.domain_client import DomainClient
from syft.client.syncing import compare_clients
from syft.client.syncing import resolve_single
from syft.node.worker import Worker
from syft.service.job.job_stash import JobStatus
from syft.service.response import SyftError
from syft.service.response import SyftSuccess


def compare_and_resolve(*, from_client: DomainClient, to_client: DomainClient):
    diff_state_before = compare_clients(from_client, to_client)
    for obj_diff_batch in diff_state_before.batches:
        widget = resolve_single(obj_diff_batch)
        widget.click_share_all_private_data()
        res = widget.click_sync()
        assert isinstance(res, SyftSuccess)
    from_client.refresh()
    to_client.refresh()
    diff_state_after = compare_clients(from_client, to_client)
    return diff_state_before, diff_state_after


def run_and_accept_result(client):
    job_high = client.code.compute(blocking=True)
    client.requests[0].accept_by_depositing_result(job_high)
    return job_high


def get_ds_client(client: DomainClient) -> DomainClient:
    client.register(
        name="a",
        email="a@a.com",
        password="asdf",
        password_verify="asdf",
    )
    return client.login(email="a@a.com", password="asdf")


@pytest.fixture(scope="function")
def full_high_worker(n_consumers: int = 3, create_producer: bool = True) -> Worker:
    _node = sy.orchestra.launch(
        node_side_type=NodeSideType.HIGH_SIDE,
        name=token_hex(8),
        # dev_mode=True,
        reset=True,
        n_consumers=n_consumers,
        create_producer=create_producer,
        queue_port=None,
        in_memory_workers=True,
        local_db=False,
        thread_workers=False,
    )
    # startup code here
    yield _node
    # Cleanup code
    _node.python_node.cleanup()
    _node.land()


@pytest.fixture(scope="function")
def full_low_worker(n_consumers: int = 3, create_producer: bool = True) -> Worker:
    _node = sy.orchestra.launch(
        node_side_type=NodeSideType.LOW_SIDE,
        name=token_hex(8),
        # dev_mode=True,
        reset=True,
        n_consumers=n_consumers,
        create_producer=create_producer,
        queue_port=None,
        in_memory_workers=True,
        local_db=False,
        thread_workers=False,
    )
    # startup code here
    yield _node
    # Cleanup code
    _node.python_node.cleanup()
    _node.land()


@sy.api_endpoint_method()
def mock_function(context) -> str:
    return -42


@sy.api_endpoint_method()
def private_function(context) -> str:
    return 42


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@pytest.mark.local_node
def test_twin_api_integration(full_high_worker, full_low_worker):
    low_client = full_low_worker.login(
        email="info@openmined.org", password="changethis"
    )
    high_client = full_high_worker.login(
        email="info@openmined.org", password="changethis"
    )
    client_low_ds = get_ds_client(low_client)

    new_endpoint = sy.TwinAPIEndpoint(
        path="testapi.query",
        private_function=private_function,
        mock_function=mock_function,
        description="",
    )
    high_client.api.services.api.add(endpoint=new_endpoint)
    high_client.refresh()
    high_private_result = high_client.api.services.testapi.query.private()

    job = high_client.api.services.job.get_all()[0]
    private_job_id = job.id

    diff_before, diff_after = compare_and_resolve(
        from_client=high_client, to_client=low_client
    )
    assert not diff_before.is_same
    assert diff_after.is_same

    client_low_ds.refresh()

    @syft.syft_function_single_use(
        query=client_low_ds.api.services.testapi.query,
    )
    def compute(query):
        return query()

    _ = client_low_ds.code.request_code_execution(compute)

    diff_before, diff_after = compare_and_resolve(
        from_client=low_client, to_client=high_client
    )

    job_high = high_client.code.compute(query=high_client.api.services.testapi.query)
    high_client.requests[0].accept_by_depositing_result(job_high)
    diff_before, diff_after = compare_and_resolve(
        from_client=high_client, to_client=low_client
    )
    client_low_ds.refresh()
    res = client_low_ds.code.compute(query=client_low_ds.api.services.testapi.query)
    assert res.syft_action_data == high_private_result
    assert diff_after.is_same

    # verify that ds cannot access private job
    assert client_low_ds.api.services.job.get(private_job_id) is None
    assert low_client.api.services.job.get(private_job_id) is not None

    # we only sync the mock function, we never sync the private function to the low side
    mock_res = low_client.api.services.testapi.query.mock()
    private_res = low_client.api.services.testapi.query.private()
    assert mock_res == -42
    assert isinstance(
        private_res, SyftError
    ), "Should not be able to access private function on low side."


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@pytest.mark.local_node
def test_function_error(full_low_worker) -> None:
    root_domain_client = full_low_worker.login(
        email="info@openmined.org", password="changethis"
    )
    root_domain_client.register(
        name="data-scientist",
        email="test_user@openmined.org",
        password="0000",
        password_verify="0000",
    )
    ds_client = root_domain_client.login(
        email="test_user@openmined.org",
        password="0000",
    )

    users = root_domain_client.users.get_all()

    @sy.syft_function_single_use()
    def compute_sum():
        raise RuntimeError

    ds_client.api.services.code.request_code_execution(compute_sum)

    users[-1].allow_mock_execution()
    result = ds_client.api.services.code.compute_sum(blocking=True)
    assert isinstance(result.get(), Err)

    job_info = ds_client.api.services.code.compute_sum(blocking=False)
    result = job_info.wait(timeout=10)
    assert isinstance(result.get(), Err)
    assert job_info.status == JobStatus.ERRORED
