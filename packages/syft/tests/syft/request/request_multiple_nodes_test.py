# stdlib
from textwrap import dedent

# third party
import numpy as np
import pytest
from tests.syft.users.user_test import get_mock_client

# syft absolute
import syft as sy
from syft.client.domain_client import DomainClient
from syft.service.job.job_stash import Job
from syft.service.job.job_stash import JobStatus
from syft.service.user.user_roles import ServiceRole


@pytest.fixture
def worker_2(faker, stage_protocol):
    return sy.Worker.named(name=faker.name())


@pytest.fixture
def client_do_1(worker) -> DomainClient:
    return get_mock_client(worker.root_client, ServiceRole.DATA_OWNER)


@pytest.fixture
def client_do_2(worker):
    return get_mock_client(worker.root_client, ServiceRole.DATA_OWNER)


@pytest.fixture
def client_ds_1(worker):
    return get_mock_client(worker.root_client, ServiceRole.DATA_SCIENTIST)


@pytest.fixture
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


@pytest.fixture
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


def test_transfer_request_blocking(
    client_ds_1, client_do_1, client_do_2, dataset_1, dataset_2
):
    @sy.syft_function_single_use(data=dataset_1)
    def compute_sum(data) -> float:
        return data.mean()

    compute_sum.code = dedent(compute_sum.code)

    request_ds = client_ds_1.code.request_code_execution(compute_sum)
    print(request_ds)
    print(client_ds_1.requests[0])

    # Submit + execute on second node
    request_1_do = client_do_1.requests[0]
    client_do_2.code.submit(request_1_do.code)

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
    client_do_2.code.submit(request_1_do.code)

    client_do_2._fetch_api(client_do_2.credentials)
    job_2 = client_do_2.code.compute_mean(data=dataset_2, blocking=False)
    assert isinstance(job_2, Job)

    # Transfer back Job Info
    res = request_1_do.submit_job_info(job_2.info)
    assert isinstance(res, sy.SyftSuccess)

    # DS checks job info
    job_1_ds = client_ds_1.code.compute_mean.jobs[-1]
    assert job_1_ds.info == job_2.info

    # DO finishes + syncs job result
    result = job_2.wait().get()
    assert result == dataset_2.data.mean()
    assert job_2.status == JobStatus.COMPLETED
    res = request_1_do.accept_by_depositing_result(job_2)
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
