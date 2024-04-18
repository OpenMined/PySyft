# stdlib
from textwrap import dedent

# syft absolute
import syft
import syft as sy
from syft.client.domain_client import DomainClient
from syft.client.syncing import compare_clients
from syft.client.syncing import resolve_single
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


@syft.syft_function_single_use()
def compute() -> int:
    return 42


def get_ds_client(client: DomainClient) -> DomainClient:
    client.register(
        name="a",
        email="a@a.com",
        password="asdf",
        password_verify="asdf",
    )
    return client.login(email="a@a.com", password="asdf")


def test_diff_state(low_worker, high_worker):
    low_client: DomainClient = low_worker.root_client
    client_low_ds = get_ds_client(low_client)
    high_client: DomainClient = high_worker.root_client

    @sy.syft_function_single_use()
    def compute() -> int:
        return 42

    compute.code = dedent(compute.code)

    _ = client_low_ds.code.request_code_execution(compute)

    diff_state_before, diff_state_after = compare_and_resolve(
        from_client=low_client, to_client=high_client
    )

    assert not diff_state_before.is_same

    assert diff_state_after.is_same

    run_and_accept_result(high_client)
    diff_state_before, diff_state_after = compare_and_resolve(
        from_client=high_client, to_client=low_client
    )

    high_state = high_client.get_sync_state()
    low_state = high_client.get_sync_state()
    assert high_state.get_previous_state_diff().is_same
    assert low_state.get_previous_state_diff().is_same
    assert diff_state_after.is_same

    client_low_ds.refresh()
    res = client_low_ds.code.compute(blocking=True)
    assert res == compute(blocking=True).get()


@sy.api_endpoint_method()
def mock_function(context) -> str:
    return -42


@sy.api_endpoint_method()
def private_function(context) -> str:
    return 42


def test_twin_api_integration(low_worker, high_worker):
    low_client = low_worker.root_client
    high_client = high_worker.root_client
    client_low_ds = get_ds_client(low_client)

    new_endpoint = sy.TwinAPIEndpoint(
        path="testapi.query",
        private_function=private_function,
        mock_function=mock_function,
        description="",
    )
    high_client.api.services.api.add(endpoint=new_endpoint)
    high_client.refresh()
    high_private_res = high_client.api.services.testapi.query.private()
    assert high_private_res == 42

    diff_before, diff_after = compare_and_resolve(
        from_client=high_client, to_client=low_client
    )

    client_low_ds.refresh()
    low_private_res = client_low_ds.api.services.testapi.query.private()
    assert isinstance(
        low_private_res, SyftError
    ), "Should not have access to private on low side"
    low_mock_res = client_low_ds.api.services.testapi.query.mock()
    high_mock_res = high_client.api.services.testapi.query.mock()
    assert low_mock_res == high_mock_res == -42

    @syft.syft_function_single_use(
        query=client_low_ds.api.services.testapi.query,
    )
    def compute(query):
        return query()

    compute.code = dedent(compute.code)
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
    assert res.syft_action_data == high_client.api.services.testapi.query.private()
    assert diff_after.is_same
