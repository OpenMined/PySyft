# stdlib
from collections import OrderedDict
from textwrap import dedent

# third party
import numpy as np

# syft absolute
import syft as sy
from syft.client.api import APIRegistry


def test_local_execution(worker):
    root_domain_client = worker.root_client
    dataset = sy.Dataset(
        name="local_test",
        asset_list=[
            sy.Asset(
                name="local_test",
                data=np.array([1, 2, 3]),
                mock=np.array([1, 1, 1]),
            )
        ],
    )
    root_domain_client.upload_dataset(dataset)
    asset = root_domain_client.datasets[0].assets[0]

    APIRegistry.__api_registry__ = OrderedDict()

    APIRegistry.set_api_for(
        node_uid=worker.id,
        user_verify_key=root_domain_client.verify_key,
        api=root_domain_client.api,
    )

    @sy.syft_function(
        input_policy=sy.ExactMatch(x=asset),
        output_policy=sy.SingleExecutionExactOutput(),
    )
    def my_func(x):
        return x + 1

    my_func.code = dedent(my_func.code)

    # time.sleep(10)
    local_res = my_func(x=asset, time_alive=1)
    assert (local_res == np.array([2, 2, 2])).all()
