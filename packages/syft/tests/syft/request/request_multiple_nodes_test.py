# stdlib
import secrets
from textwrap import dedent

# third party
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft.service.job.job_stash import Job
from syft.service.job.job_stash import JobStatus
@pytest.fixture(scope="function")
def node_1():
    name = secrets.token_hex(4)
    node = sy.Worker(
        name=name,
        local_db=True,
        n_consumers=1,
        in_memory_workers=True,
        create_producer=True,
        node_side_type="low",
        dev_mode=True,
    )
    yield node
    node.close()

@pytest.fixture(scope="function")
def node_2():
    name = secrets.token_hex(4)
    node = sy.Worker(
        name=name,
        local_db=True,
        n_consumers=1,
        in_memory_workers=True,
        create_producer=True,
        dev_mode=True,
        node_side_type="high",
    )
    yield node
    node.close()



@pytest.fixture(scope="function")
def client_do_1(node_1):
    guest_client = node_1.get_guest_client()
    client_do_1 = guest_client.login(email="info@openmined.org", password="changethis")
    return client_do_1


@pytest.fixture(scope="function")
def client_do_2(node_2):
    guest_client = node_2.get_guest_client()
    client_do_2 = guest_client.login(email="info@openmined.org", password="changethis")
    return client_do_2


@pytest.fixture(scope="function")
def client_ds_1(node_1, client_do_1):
    client_do_1.register(
        name="test_user", email="test@us.er", password="1234", password_verify="1234"
    )
    return client_do_1.login(email="test@us.er", password="1234")


@pytest.fixture(scope="function")
def dataset_1(client_do_1):
    mock = np.array([0, 1, 2, 3, 4])
    private = np.array([5, 6, 7, 8, 9])

    dataset = sy.Dataset(
        name="my-dataset",
        description="abc",
        asset_list=[
            sy.Asset(
                name="numpy-data",
                mock=mock,
                data=private,
                shape=private.shape,
                mock_is_real=True,
            )
        ],
    )

    client_do_1.upload_dataset(dataset)
    return client_do_1.datasets[0].assets[0]


@pytest.fixture(scope="function")
def dataset_2(client_do_2):
    mock = np.array([0, 1, 2, 3, 4]) + 10
    private = np.array([5, 6, 7, 8, 9]) + 10

    dataset = sy.Dataset(
        name="my-dataset",
        description="abc",
        asset_list=[
            sy.Asset(
                name="numpy-data",
                mock=mock,
                data=private,
                shape=private.shape,
                mock_is_real=True,
            )
        ],
    )

    client_do_2.upload_dataset(dataset)
    return client_do_2.datasets[0].assets[0]

@pytest.mark.flaky(reruns=2, reruns_delay=1)
def test_transfer_request_blocking(
    client_ds_1, client_do_1, client_do_2, dataset_1, dataset_2
):
    @sy.syft_function_single_use(data=dataset_1)
    def compute_sum(data) -> float:
        return data.mean()

    compute_sum.code = dedent(compute_sum.code)

    client_ds_1.code.request_code_execution(compute_sum)

    # Submit + execute on second node
    request_1_do = client_do_1.requests[0]
    client_do_2.sync_code_from_request(request_1_do)

    # DO executes + syncs
    client_do_2._fetch_api(client_do_2.credentials)
    result_2 = client_do_2.code.compute_sum(data=dataset_2).get()
    assert result_2 == dataset_2.data.mean()
    res = request_1_do.accept_by_depositing_result(result_2)
    assert isinstance(res, sy.SyftSuccess)

    # DS gets result blocking + nonblocking
    result_ds_blocking = client_ds_1.code.compute_sum(
        data=dataset_1, blocking=True
    ).get()

    job_1_ds = client_ds_1.code.compute_sum(data=dataset_1, blocking=False)
    assert isinstance(job_1_ds, Job)
    assert job_1_ds == client_ds_1.code.compute_sum.jobs[-1]
    assert job_1_ds.status == JobStatus.COMPLETED

    result_ds_nonblocking = job_1_ds.wait().get()

    assert result_ds_blocking == result_ds_nonblocking == dataset_2.data.mean()


@pytest.mark.flaky(reruns=2, reruns_delay=1)
def test_transfer_request_nonblocking(
    client_ds_1, client_do_1, client_do_2, dataset_1, dataset_2
):
    @sy.syft_function_single_use(data=dataset_1)
    def compute_mean(data) -> float:
        return data.mean()

    compute_mean.code = dedent(compute_mean.code)

    client_ds_1.code.request_code_execution(compute_mean)

    # Submit + execute on second node
    request_1_do = client_do_1.requests[0]
    client_do_2.sync_code_from_request(request_1_do)

    client_do_2._fetch_api(client_do_2.credentials)
    job_2 = client_do_2.code.compute_mean(data=dataset_2, blocking=False)
    assert isinstance(job_2, Job)

    # Transfer back Job Info
    job_2_info = job_2.info()
    assert job_2_info.result is None
    assert job_2_info.status is not None
    res = request_1_do.sync_job(job_2_info)
    assert isinstance(res, sy.SyftSuccess)

    # DS checks job info
    job_1_ds = client_ds_1.code.compute_mean.jobs[-1]
    assert job_1_ds.status == job_2.status

    # DO finishes + syncs job result
    result = job_2.wait().get()
    assert result == dataset_2.data.mean()
    assert job_2.status == JobStatus.COMPLETED

    job_2_info_with_result = job_2.info(result=True)
    res = request_1_do.accept_by_depositing_result(job_2_info_with_result)
    assert isinstance(res, sy.SyftSuccess)

    # DS gets result blocking + nonblocking
    result_ds_blocking = client_ds_1.code.compute_mean(
        data=dataset_1, blocking=True
    ).get()

    job_1_ds = client_ds_1.code.compute_mean(data=dataset_1, blocking=False)
    assert isinstance(job_1_ds, Job)
    assert job_1_ds == client_ds_1.code.compute_mean.jobs[-1]
    assert job_1_ds.status == JobStatus.COMPLETED

    result_ds_nonblocking = job_1_ds.wait().get()

    assert result_ds_blocking == result_ds_nonblocking == dataset_2.data.mean()
