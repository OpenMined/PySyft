@pytest.mark.flaky(reruns=3, reruns_delay=3)
@pytest.mark.local_node
def test_transfer_request_blocking(
    client_ds_1, client_do_1, client_do_2, dataset_1, dataset_2
):
    @sy.syft_function_single_use(data=dataset_1)
    def compute_sum(data) -> float:
        return data.mean()

    client_ds_1.code.request_code_execution(compute_sum)

    # Submit + execute on second node
    request_1_do = client_do_1.requests[0]
    client_do_2.sync_code_from_request(request_1_do)

    # DO executes + syncs
    client_do_2._fetch_api(client_do_2.credentials)
    result_2 = client_do_2.code.compute_sum(data=dataset_2).get()
    assert result_2 == dataset_2.data.mean()

    res = request_1_do.accept_by_depositing_result(
        result_2
    )
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

    assert (
        result_ds_blocking
        == result_ds_nonblocking
        == dataset_2.data.mean()
    )


@pytest.mark.flaky(reruns=3, reruns_delay=3)
@pytest.mark.local_node
def test_transfer_request_nonblocking(
    client_ds_1, client_do_1, client_do_2, dataset_1, dataset_2
):
    @sy.syft_function_single_use(data=dataset_1)
    def compute_mean(data) -> float:
        return data.mean()

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
    res = request_1_do.accept_by_depositing_result(
        job_2_info_with_result
    )
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

    assert (
        result_ds_blocking
        == result_ds_nonblocking
        == dataset_2.data.mean()
    )
