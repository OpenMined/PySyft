# stdlib
import sys

# third party
import numpy as np
import pytest

# syft absolute
import syft
import syft as sy
from syft.abstract_node import NodeSideType
from syft.client.domain_client import DomainClient
from syft.client.sync_decision import SyncDecision
from syft.client.syncing import compare_clients
from syft.client.syncing import compare_states
from syft.client.syncing import resolve
from syft.client.syncing import resolve_single
from syft.service.action.action_object import ActionObject
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


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
# @pytest.mark.flaky(reruns=3, reruns_delay=3)
def test_sync_flow():
    # somehow skipif does not work
    if sys.platform == "win32":
        return
    low_worker = sy.Worker(
        name="low-test",
        local_db=True,
        n_consumers=1,
        create_producer=True,
        node_side_type=NodeSideType.LOW_SIDE,
        queue_port=None,
        in_memory_workers=True,
    )
    high_worker = sy.Worker(
        name="high-test",
        local_db=True,
        n_consumers=1,
        create_producer=True,
        node_side_type=NodeSideType.HIGH_SIDE,
        queue_port=None,
        in_memory_workers=True,
    )

    low_client = low_worker.root_client
    high_client = high_worker.root_client

    low_client.register(
        email="newuser@openmined.org",
        name="John Doe",
        password="pw",
        password_verify="pw",
    )
    client_low_ds = low_worker.guest_client

    mock_high = np.array([10, 11, 12, 13, 14])
    private_high = np.array([15, 16, 17, 18, 19])

    dataset_high = sy.Dataset(
        name="my-dataset",
        description="abc",
        asset_list=[
            sy.Asset(
                name="numpy-data",
                mock=mock_high,
                data=private_high,
                shape=private_high.shape,
                mock_is_real=True,
            )
        ],
    )

    high_client.upload_dataset(dataset_high)
    mock_low = np.array([0, 1, 2, 3, 4])  # do_high.mock

    dataset_low = sy.Dataset(
        id=dataset_high.id,
        name="my-dataset",
        description="abc",
        asset_list=[
            sy.Asset(
                name="numpy-data",
                mock=mock_low,
                data=ActionObject.empty(data_node_id=high_client.id),
                shape=mock_low.shape,
                mock_is_real=True,
            )
        ],
    )

    res = low_client.upload_dataset(dataset_low)

    data_low = client_low_ds.datasets[0].assets[0]

    @sy.syft_function_single_use(data=data_low)
    def compute_mean(data) -> float:
        return data.mean()

    res = client_low_ds.code.request_code_execution(compute_mean)
    res = client_low_ds.code.request_code_execution(compute_mean)
    print(res)
    print("LOW CODE:", low_client.code.get_all())

    low_state = low_client.get_sync_state()
    high_state = high_client.get_sync_state()

    print(low_state.objects, high_state.objects)

    diff_state = compare_states(low_state, high_state)
    low_items_to_sync, high_items_to_sync = resolve(
        diff_state, decision="low", share_private_objects=True
    )

    print(low_items_to_sync, high_items_to_sync)

    low_client.apply_state(low_items_to_sync)

    high_client.apply_state(high_items_to_sync)

    low_state = low_client.get_sync_state()
    high_state = high_client.get_sync_state()

    diff_state = compare_states(low_state, high_state)

    high_client._fetch_api(high_client.credentials)

    data_high = high_client.datasets[0].assets[0]

    print(high_client.code.get_all())
    job_high = high_client.code.compute_mean(data=data_high, blocking=False)
    print("Waiting for job...")
    job_high.wait(timeout=60)
    job_high.result.get()

    # syft absolute
    from syft.service.request.request import Request

    request: Request = high_client.requests[0]
    job_info = job_high.info(public_metadata=True, result=True)

    print(request.syft_client_verify_key, request.syft_node_location)
    print(request.code.syft_client_verify_key, request.code.syft_node_location)
    request.accept_by_depositing_result(job_info)

    request = high_client.requests[0]
    code = request.code
    job_high._get_log_objs()

    action_store_high = high_worker.get_service("actionservice").store
    blob_store_high = high_worker.get_service("blobstorageservice").stash.partition
    assert (
        f"{client_low_ds.verify_key}_READ"
        in action_store_high.permissions[job_high.result.id.id]
    )
    assert (
        f"{client_low_ds.verify_key}_READ"
        in blob_store_high.permissions[job_high.result.syft_blob_storage_entry_id]
    )

    low_state = low_client.get_sync_state()
    high_state = high_client.get_sync_state()

    diff_state_2 = compare_states(low_state, high_state)

    low_items_to_sync, high_items_to_sync = resolve(
        diff_state_2, decision="high", share_private_objects=True
    )
    for diff in diff_state_2.diffs:
        print(diff.status, diff.object_type)
    low_client.apply_state(low_items_to_sync)

    action_store_low = low_worker.get_service("actionservice").store
    blob_store_low = low_worker.get_service("blobstorageservice").stash.partition
    assert (
        f"{client_low_ds.verify_key}_READ"
        in action_store_low.permissions[job_high.result.id.id]
    )
    assert (
        f"{client_low_ds.verify_key}_READ"
        in blob_store_low.permissions[job_high.result.syft_blob_storage_entry_id]
    )

    low_state = low_client.get_sync_state()
    high_state = high_client.get_sync_state()
    res_low = client_low_ds.code.compute_mean(data=data_low)
    print("Res Low", res_low)

    assert res_low.get() == private_high.mean()

    assert (
        res_low.id.id
        == job_high.result.id.id
        == code.output_history[-1].outputs[0].id.id
    )
    assert (
        job_high.result.syft_blob_storage_entry_id == res_low.syft_blob_storage_entry_id
    )

    job_low = client_low_ds.code.compute_mean(data=data_low, blocking=False)

    assert job_low.id == job_high.id
    assert job_low.result.id == job_high.result.id
    assert (
        job_low.result.syft_blob_storage_entry_id
        == job_high.result.syft_blob_storage_entry_id
    )
    low_worker.cleanup()
    high_worker.cleanup()


def test_forget_usercode(low_worker, high_worker):
    low_client = low_worker.root_client
    client_low_ds = low_worker.guest_client
    high_client = high_worker.root_client

    @sy.syft_function_single_use()
    def compute() -> int:
        print("computing...")
        return 42

    _ = client_low_ds.code.request_code_execution(compute)

    diff_state = compare_clients(low_client, high_client)
    low_items_to_sync, high_items_to_sync = resolve(
        diff_state, decision="low", share_private_objects=True
    )
    low_client.apply_state(low_items_to_sync)
    high_client.apply_state(high_items_to_sync)

    high_client.code.get_all()
    job_high = high_client.code.compute().get()
    # job_info = job_high.info(public_metadata=True, result=True)

    request = high_client.requests[0]
    request.accept_by_depositing_result(job_high)

    # job_high._get_log_objs()

    low_state = low_client.get_sync_state()
    high_state = high_client.get_sync_state()

    diff_state_2 = compare_states(low_state, high_state)

    def skip_if_user_code(diff):
        if diff.root.object_type == "UserCode":
            return SyncDecision.IGNORE
        raise Exception(f"Should not reach here, but got {diff.root.object_type}")

    low_items_to_sync, high_items_to_sync = resolve(
        diff_state_2,
        share_private_objects=True,
        decision_callback=skip_if_user_code,
    )


@sy.api_endpoint_method()
def mock_function(context) -> str:
    return -42


@sy.api_endpoint_method()
def private_function(context) -> str:
    return 42


def test_skip_user_code(low_worker, high_worker):
    low_client = low_worker.root_client
    client_low_ds = low_worker.guest_client
    high_client = high_worker.root_client

    @sy.syft_function_single_use()
    def compute() -> int:
        return 42

    _ = client_low_ds.code.request_code_execution(compute)

    def skip_if_user_code(diff):
        if diff.root.object_type == "UserCode":
            return SyncDecision.SKIP
        raise Exception(f"Should not reach here, but got {diff.root.object_type}")

    diff_state = compare_clients(low_client, high_client)
    low_items_to_sync, high_items_to_sync = resolve(
        diff_state,
        share_private_objects=True,
        decision_callback=skip_if_user_code,
    )
    low_client.apply_state(low_items_to_sync)
    high_client.apply_state(high_items_to_sync)

    assert low_items_to_sync.is_empty
    assert high_items_to_sync.is_empty


def test_unignore(low_worker, high_worker):
    low_client = low_worker.root_client
    client_low_ds = low_worker.guest_client
    high_client = high_worker.root_client

    @sy.syft_function_single_use()
    def compute() -> int:
        return 42

    _ = client_low_ds.code.request_code_execution(compute)

    diff_state = compare_clients(low_client, high_client)
    low_items_to_sync, high_items_to_sync = resolve(
        diff_state,
        share_private_objects=True,
        decision="ignore",
    )
    low_client.apply_state(low_items_to_sync)
    high_client.apply_state(high_items_to_sync)

    assert low_items_to_sync.is_empty
    assert high_items_to_sync.is_empty

    diff_state = compare_clients(low_client, high_client)

    for ignored in diff_state.ignored_changes:
        deps = ignored.batch.get_dependencies()
        if "Request" in [dep.object_type for dep in deps]:
            ignored.stage_change()

    low_items_to_sync, high_items_to_sync = resolve(
        diff_state,
        share_private_objects=True,
        decision="low",
    )

    assert not low_items_to_sync.is_empty
    assert not high_items_to_sync.is_empty

    low_client.apply_state(low_items_to_sync)
    high_client.apply_state(high_items_to_sync)

    diff_state = compare_clients(low_client, high_client)
    low_items_to_sync, high_items_to_sync = resolve(
        diff_state,
        share_private_objects=True,
        decision="low",
    )

    assert diff_state.is_same


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

    diff_state = compare_clients(low_client, high_client)
    low_items_to_sync, high_items_to_sync = resolve(
        diff_state, decision="low", share_private_objects=True
    )

    assert not diff_state.is_same
    assert len(diff_state.diffs) % 2 == 0
    assert not low_items_to_sync.is_empty
    assert not high_items_to_sync.is_empty

    low_client.apply_state(low_items_to_sync)
    high_client.apply_state(high_items_to_sync)

    _ = client_low_ds.code.request_code_execution(compute_thrice)

    diff_state = compare_clients(low_client, high_client)
    low_items_to_sync, high_items_to_sync = resolve(
        diff_state, decision="low", share_private_objects=True
    )

    assert not diff_state.is_same
    assert len(diff_state.diffs) % 3 == 0
    assert not low_items_to_sync.is_empty
    assert not high_items_to_sync.is_empty


def test_sync_high(low_worker, high_worker):
    low_client = low_worker.root_client
    client_low_ds = low_worker.guest_client
    high_client = high_worker.root_client

    @sy.syft_function_single_use()
    def compute() -> int:
        return 42

    _ = client_low_ds.code.request_code_execution(compute)

    diff_state = compare_clients(low_client, high_client)
    low_items_to_sync, high_items_to_sync = resolve(
        diff_state,
        decision="high",
    )

    assert not diff_state.is_same
    assert not low_items_to_sync.is_empty
    assert high_items_to_sync.is_empty


@pytest.mark.parametrize(
    "decision",
    ["skip", "ignore"],
)
def test_sync_skip_ignore(low_worker, high_worker, decision):
    low_client = low_worker.root_client
    client_low_ds = low_worker.guest_client
    high_client = high_worker.root_client

    @sy.syft_function_single_use()
    def compute() -> int:
        return 42

    _ = client_low_ds.code.request_code_execution(compute)

    diff_state = compare_clients(low_client, high_client)
    low_items_to_sync, high_items_to_sync = resolve(
        diff_state,
        decision=decision,
    )

    assert not diff_state.is_same
    assert low_items_to_sync.is_empty
    assert high_items_to_sync.is_empty

    low_client.apply_state(low_items_to_sync)
    high_client.apply_state(high_items_to_sync)

    def should_not_be_called(diff):
        # should not be called when decision is ignore before
        if decision == "ignore":
            raise Exception("Should not reach here")
        return SyncDecision.SKIP

    diff_state = compare_clients(low_client, high_client)
    low_items_to_sync, high_items_to_sync = resolve(
        diff_state,
        decision_callback=should_not_be_called,
    )


def test_update_after_ignore(low_worker, high_worker):
    low_client = low_worker.root_client
    client_low_ds = low_worker.guest_client
    high_client = high_worker.root_client

    @sy.syft_function_single_use()
    def compute() -> int:
        return 42

    _ = client_low_ds.code.request_code_execution(compute)

    diff_state = compare_clients(low_client, high_client)
    low_items_to_sync, high_items_to_sync = resolve(
        diff_state,
        decision="ignore",
    )

    assert not diff_state.is_same
    assert low_items_to_sync.is_empty
    assert high_items_to_sync.is_empty

    low_client.apply_state(low_items_to_sync)
    high_client.apply_state(high_items_to_sync)

    @sy.syft_function_single_use()
    def compute() -> int:
        return 43

    # _ = client_low_ds.code.request_code_execution(compute)
    low_client.requests[-1].approve()

    diff_state = compare_clients(low_client, high_client)
    low_items_to_sync, high_items_to_sync = resolve(
        diff_state,
        decision="low",
    )

    assert not high_items_to_sync.is_empty


@pytest.mark.parametrize(
    "decision",
    ["skip", "ignore", "low", "high"],
)
def test_sync_empty(low_worker, high_worker, decision):
    low_client = low_worker.root_client
    high_client = high_worker.root_client

    diff_state = compare_clients(low_client, high_client)
    low_items_to_sync, high_items_to_sync = resolve(
        diff_state,
        decision=decision,
    )

    assert diff_state.is_same
    assert low_items_to_sync.is_empty
    assert high_items_to_sync.is_empty


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@pytest.mark.flaky(reruns=3, reruns_delay=3)
def test_sync_flow_no_sharing():
    # somehow skipif does not work
    if sys.platform == "win32":
        return
    low_worker = sy.Worker(
        name="low-test-2",
        local_db=True,
        n_consumers=1,
        create_producer=True,
        node_side_type=NodeSideType.LOW_SIDE,
        queue_port=None,
        in_memory_workers=True,
    )
    high_worker = sy.Worker(
        name="high-test-2",
        local_db=True,
        n_consumers=1,
        create_producer=True,
        node_side_type=NodeSideType.HIGH_SIDE,
        queue_port=None,
        in_memory_workers=True,
    )

    low_client = low_worker.root_client
    high_client = high_worker.root_client

    low_client.register(
        email="newuser@openmined.org",
        name="John Doe",
        password="pw",
        password_verify="pw",
    )
    client_low_ds = low_worker.guest_client

    mock_high = np.array([10, 11, 12, 13, 14])
    private_high = np.array([15, 16, 17, 18, 19])

    dataset_high = sy.Dataset(
        name="my-dataset",
        description="abc",
        asset_list=[
            sy.Asset(
                name="numpy-data",
                mock=mock_high,
                data=private_high,
                shape=private_high.shape,
                mock_is_real=True,
            )
        ],
    )

    high_client.upload_dataset(dataset_high)
    mock_low = np.array([0, 1, 2, 3, 4])  # do_high.mock

    dataset_low = sy.Dataset(
        id=dataset_high.id,
        name="my-dataset",
        description="abc",
        asset_list=[
            sy.Asset(
                name="numpy-data",
                mock=mock_low,
                data=ActionObject.empty(data_node_id=high_client.id),
                shape=mock_low.shape,
                mock_is_real=True,
            )
        ],
    )

    res = low_client.upload_dataset(dataset_low)

    data_low = client_low_ds.datasets[0].assets[0]

    @sy.syft_function_single_use(data=data_low)
    def compute_mean(data) -> float:
        return data.mean()

    res = client_low_ds.code.request_code_execution(compute_mean)
    print(res)
    print("LOW CODE:", low_client.code.get_all())

    low_state = low_client.get_sync_state()
    high_state = high_client.get_sync_state()

    print(low_state.objects, high_state.objects)

    diff_state = compare_states(low_state, high_state)
    low_items_to_sync, high_items_to_sync = resolve(
        diff_state, decision="low", share_private_objects=True
    )

    print(low_items_to_sync, high_items_to_sync)

    low_client.apply_state(low_items_to_sync)

    high_client.apply_state(high_items_to_sync)

    low_state = low_client.get_sync_state()
    high_state = high_client.get_sync_state()

    diff_state = compare_states(low_state, high_state)

    high_client._fetch_api(high_client.credentials)

    data_high = high_client.datasets[0].assets[0]

    print(high_client.code.get_all())
    job_high = high_client.code.compute_mean(data=data_high, blocking=False)
    print("Waiting for job...")
    job_high.wait(timeout=60)
    job_high.result.get()

    # syft absolute
    from syft.service.request.request import Request

    request: Request = high_client.requests[0]
    job_info = job_high.info(public_metadata=True, result=True)

    print(request.syft_client_verify_key, request.syft_node_location)
    print(request.code.syft_client_verify_key, request.code.syft_node_location)
    request.accept_by_depositing_result(job_info)

    request = high_client.requests[0]
    job_high._get_log_objs()

    action_store_high = high_worker.get_service("actionservice").store
    blob_store_high = high_worker.get_service("blobstorageservice").stash.partition
    assert (
        f"{client_low_ds.verify_key}_READ"
        in action_store_high.permissions[job_high.result.id.id]
    )
    assert (
        f"{client_low_ds.verify_key}_READ"
        in blob_store_high.permissions[job_high.result.syft_blob_storage_entry_id]
    )

    low_state = low_client.get_sync_state()
    high_state = high_client.get_sync_state()

    diff_state_2 = compare_states(low_state, high_state)

    low_items_to_sync, high_items_to_sync = resolve(
        diff_state_2, decision="high", share_private_objects=False, ask_for_input=False
    )
    for diff in diff_state_2.diffs:
        print(diff.status, diff.object_type)
    low_client.apply_state(low_items_to_sync)

    low_state = low_client.get_sync_state()
    high_state = high_client.get_sync_state()
    res_low = client_low_ds.code.compute_mean(data=data_low)
    assert isinstance(res_low, SyftError)
    assert (
        res_low.message
        == f"Permission: [READ: {job_high.result.id.id} as {client_low_ds.verify_key}] denied"
    )

    job_low = client_low_ds.code.compute_mean(data=data_low, blocking=False)

    assert job_low.id == job_high.id
    assert job_low.result.id == job_high.result.id
    result = job_low.result.get()
    assert isinstance(result, SyftError)
    assert (
        result.message
        == f"Permission: [READ: {job_high.result.id.id} as {client_low_ds.verify_key}] denied"
    )

    low_worker.cleanup()
    high_worker.cleanup()
