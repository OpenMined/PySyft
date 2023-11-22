# stdlib
from textwrap import dedent

# third party
import pytest

# syft absolute
import syft as sy
from syft import ActionObject
from syft import syft_function
from syft import syft_function_single_use
from syft.service.response import SyftError
from syft.service.response import SyftSuccess


@pytest.fixture
def node():
    _node = sy.orchestra.launch(
        name="nested_job_test_domain",
        dev_mode=True,
        reset=True,
        n_consumers=3,
        create_producer=True,
        queue_port=3322,
    )
    # startup code here
    yield _node
    # Cleanup code
    _node.land()


def test_nested_jobs(node):
    client = node.login(email="info@openmined.org", password="changethis")

    res = client.register(name="a", email="aa@b.org", password="c", password_verify="c")
    assert isinstance(res, SyftSuccess)
    ds_client = node.login(email="aa@b.org", password="c")
    ## Dataset

    x = ActionObject.from_obj([1, 2])
    x_ptr = x.send(ds_client)

    ## Batch function

    @syft_function()
    def process_batch(batch):
        print(f"starting batch {batch}")
        return batch + 1

    process_batch.code = dedent(process_batch.code)

    res = ds_client.code.submit(process_batch)
    print(res)

    ## Main function

    @syft_function_single_use(x=x_ptr)
    def process_all(domain, x):
        jobs = []
        for elem in x:
            batch_job = domain.launch_job(process_batch, batch=elem)
            jobs += [batch_job]
        return None

    process_all.code = dedent(process_all.code)

    # Approve & run
    res = ds_client.code.request_code_execution(process_all)
    print(res)
    assert not isinstance(res, SyftError)
    client.requests[-1].approve()

    job = ds_client.code.process_all(x=x_ptr, blocking=False)
    job.wait()
    assert len(job.subjobs) == 2
    assert sum([j.wait().get() for j in job.subjobs]) == 5
