# stdlib
from textwrap import dedent

# third party
import numpy as np

# syft absolute
import syft as sy
from syft.abstract_node import NodeSideType
from syft.client.syncing import compare_states
from syft.client.syncing import resolve
from syft.service.action.action_object import ActionObject


def test_sync_flow():
    low_worker = sy.Worker(
        name="low-test",
        local_db=True,
        n_consumers=1,
        create_producer=True,
        node_side_type=NodeSideType.LOW_SIDE,
    )
    high_worker = sy.Worker(
        name="high-test",
        local_db=True,
        n_consumers=1,
        create_producer=True,
        node_side_type=NodeSideType.HIGH_SIDE,
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

    compute_mean.code = dedent(compute_mean.code)

    res = client_low_ds.code.request_code_execution(compute_mean)
    print(res)
    print("LOW CODE:", low_client.code.get_all())

    low_state = low_client.sync.get_state()
    high_state = high_client.sync.get_state()

    print(low_state.objects, high_state.objects)

    diff_state = compare_states(low_state, high_state)
    low_items_to_sync, high_items_to_sync = resolve(diff_state, decision="low")

    print(low_items_to_sync, high_items_to_sync)

    low_client.apply_state(low_items_to_sync)

    high_client.apply_state(high_items_to_sync)

    low_state = low_client.sync.get_state()
    high_state = high_client.sync.get_state()

    diff_state = compare_states(low_state, high_state)

    high_client._fetch_api(high_client.credentials)

    data_high = high_client.datasets[0].assets[0]

    print(high_client.code.get_all())
    job_high = high_client.code.compute_mean(data=data_high, blocking=False)
    print("Waiting for job...")
    job_high.wait()
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

    low_state = low_client.sync.get_state()
    high_state = high_client.sync.get_state()

    diff_state_2 = compare_states(low_state, high_state)

    low_items_to_sync, high_items_to_sync = resolve(diff_state_2, decision="high")
    for diff in diff_state_2.diffs:
        print(diff.merge_state, diff.object_type)
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

    low_state = low_client.sync.get_state()
    high_state = high_client.sync.get_state()
    res_low = client_low_ds.code.compute_mean(data=data_low)
    print("Res Low", res_low)

    assert res_low.get() == private_high.mean()
    assert res_low.id == job_high.result.id.id == code.output_history[-1][0].id.id
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
    low_worker.close()
    high_worker.close()
