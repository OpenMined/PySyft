# stdlib
from secrets import token_hex
import sys

# third party
import pytest

# syft absolute
import syft as sy
from syft import ActionObject
from syft import syft_function
from syft import syft_function_single_use
from syft.service.job.job_stash import Job
from syft.service.response import SyftSuccess


@pytest.fixture
def server():
    _server = sy.orchestra.launch(
        name=token_hex(8),
        dev_mode=True,
        reset=True,
        n_consumers=3,
        create_producer=True,
        queue_port=None,
    )
    # startup code here
    yield _server
    # Cleanup code
    _server.python_server.cleanup()
    _server.land()


# @pytest.mark.flaky(reruns=3, reruns_delay=3)
@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@pytest.mark.local_server
def test_nested_jobs(server):
    client = server.login(email="info@openmined.org", password="changethis")

    new_user_email = "aa@b.org"
    res = client.register(
        name="a", email=new_user_email, password="c", password_verify="c"
    )
    assert isinstance(res, SyftSuccess)

    ## Dataset

    x = ActionObject.from_obj([1, 2])
    x_ptr = x.send(client)

    search_result = [u for u in client.users.get_all() if u.email == new_user_email]
    assert len(search_result) == 1

    new_ds_user = search_result[0]
    new_ds_user.allow_mock_execution()

    # Login as data scientist
    ds_client = server.login(email=new_user_email, password="c")

    ## aggregate function
    @sy.syft_function()
    def aggregate_job(job_results):
        return sum(job_results)

    res = ds_client.code.submit(aggregate_job)

    ## Batch function
    @syft_function()
    def process_batch(batch):
        print(f"starting batch {batch}")
        return batch + 1

    res = ds_client.code.submit(process_batch)
    print(res)

    ## Main function

    @syft_function_single_use(x=x_ptr)
    def process_all(datasite, x):
        job_results = []
        for elem in x:
            batch_job = datasite.launch_job(process_batch, batch=elem)
            job_results += [batch_job.result]

        job = datasite.launch_job(aggregate_job, job_results=job_results)
        return job.result

    assert process_all.worker_pool_name is None

    # Approve & run
    res = ds_client.code.request_code_execution(process_all)
    print(res)

    assert ds_client.code[-1].worker_pool_name is not None
    client.requests[-1].approve(approve_nested=True)

    job = ds_client.code.process_all(x=x_ptr, blocking=False)

    assert isinstance(job, Job)

    job.wait(timeout=5)

    assert len(job.subjobs) == 3

    assert job.wait(timeout=60).get() == 5
    sub_results = [j.wait(timeout=60).get() for j in job.subjobs]
    assert set(sub_results) == {2, 3, 5}

    job = client.jobs[-1]
    assert job.job_worker_id is not None
