# third party
from result import Err

# syft absolute
import syft
import syft as sy
from syft.client.domain_client import DomainClient
from syft.client.sync_decision import SyncDecision
from syft.client.syncing import compare_clients
from syft.client.syncing import resolve_single
from syft.service.code.user_code import UserCode
from syft.service.response import SyftSuccess
from syft.service.sync.resolve_widget import ResolveWidget


def handle_decision(widget: ResolveWidget, decision: SyncDecision):
    if decision == SyncDecision.SKIP:
        return widget.click_skip()
    elif decision == SyncDecision.IGNORE:
        return widget.click_ignore()
    elif decision in [SyncDecision.LOW, SyncDecision.HIGH]:
        return widget.click_sync()
    else:
        raise ValueError(f"Unknown decision {decision}")


def compare_and_resolve(
    *,
    from_client: DomainClient,
    to_client: DomainClient,
    decision: SyncDecision = SyncDecision.LOW,
    decision_callback: callable = None,
):
    diff_state_before = compare_clients(from_client, to_client)
    for obj_diff_batch in diff_state_before.active_batches:
        widget = resolve_single(
            obj_diff_batch=obj_diff_batch,
        )
        if decision_callback:
            decision = decision_callback(obj_diff_batch)
        widget.click_share_all_private_data()
        res = handle_decision(widget, decision)
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


def test_sync_with_error(low_worker, high_worker):
    """Check syncing with an error in a syft function"""
    low_client: DomainClient = low_worker.root_client
    client_low_ds = get_ds_client(low_client)
    high_client: DomainClient = high_worker.root_client

    @sy.syft_function_single_use()
    def compute() -> int:
        raise RuntimeError
        return 42

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

    assert not diff_state_before.is_same
    assert diff_state_after.is_same

    client_low_ds.refresh()
    res = client_low_ds.code.compute(blocking=True)
    assert isinstance(res.get(), Err)


def test_ignore_unignore_single(low_worker, high_worker):
    low_client: DomainClient = low_worker.root_client
    client_low_ds = get_ds_client(low_client)
    high_client: DomainClient = high_worker.root_client

    @sy.syft_function_single_use()
    def compute() -> int:
        return 42

    _ = client_low_ds.code.request_code_execution(compute)

    diff = compare_clients(low_client, high_client)

    assert len(diff.batches) == 2  # Request + UserCode
    assert len(diff.ignored_batches) == 0

    # Ignore usercode, request also gets ignored
    res = diff[0].ignore()
    assert isinstance(res, SyftSuccess)

    diff = compare_clients(low_client, high_client)
    assert len(diff.batches) == 0
    assert len(diff.ignored_batches) == 2
    assert len(diff.all_batches) == 2

    # Unignore usercode
    res = diff.ignored_batches[0].unignore()
    assert isinstance(res, SyftSuccess)

    diff = compare_clients(low_client, high_client)
    assert len(diff.batches) == 1
    assert len(diff.ignored_batches) == 1
    assert len(diff.all_batches) == 2


def test_forget_usercode(low_worker, high_worker):
    low_client = low_worker.root_client
    client_low_ds = low_worker.guest_client
    high_client = high_worker.root_client

    @sy.syft_function_single_use()
    def compute() -> int:
        print("computing...")
        return 42

    _ = client_low_ds.code.request_code_execution(compute)

    diff_before, diff_after = compare_and_resolve(
        from_client=low_client, to_client=high_client
    )

    run_and_accept_result(high_client)

    def skip_if_user_code(diff):
        if diff.root_type is UserCode:
            return SyncDecision.IGNORE

        raise ValueError(
            f"Should not reach here after ignoring user code, got {diff.root_type}"
        )

    diff_before, diff_after = compare_and_resolve(
        from_client=low_client,
        to_client=high_client,
        decision_callback=skip_if_user_code,
    )
    assert not diff_after.is_same
    assert not diff_after.is_same


def test_request_code_execution_multiple(low_worker, high_worker):
    low_client = low_worker.root_client
    client_low_ds = low_worker.guest_client
    high_client = high_worker.root_client

    @sy.syft_function_single_use()
    def compute() -> int:
        return 42

    @sy.syft_function_single_use()
    def compute_twice() -> int:
        return 42 * 2

    @sy.syft_function_single_use()
    def compute_thrice() -> int:
        return 42 * 3

    _ = client_low_ds.code.request_code_execution(compute)
    _ = client_low_ds.code.request_code_execution(compute_twice)

    diff_before, diff_after = compare_and_resolve(
        from_client=low_client, to_client=high_client
    )

    assert not diff_before.is_same
    assert diff_after.is_same

    _ = client_low_ds.code.request_code_execution(compute_thrice)

    diff_before, diff_after = compare_and_resolve(
        from_client=low_client, to_client=high_client
    )

    assert not diff_before.is_same
    assert diff_after.is_same
