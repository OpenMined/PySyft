# stdlib
from collections import OrderedDict
import sys

# third party
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft.client.api import APIRegistry


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_local_execution(worker):
    root_datasite_client = worker.root_client
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
    root_datasite_client.upload_dataset(dataset)
    asset = root_datasite_client.datasets[0].assets[0]

    APIRegistry.__api_registry__ = OrderedDict()

    APIRegistry.set_api_for(
        server_uid=worker.id,
        user_verify_key=root_datasite_client.verify_key,
        api=root_datasite_client.api,
    )

    @sy.syft_function(
        input_policy=sy.ExactMatch(x=asset),
        output_policy=sy.SingleExecutionExactOutput(),
    )
    def my_func(x):
        return x + 1

    # time.sleep(10)
    local_res = my_func(
        x=asset,
        time_alive=1,
    )
    assert (local_res == np.array([2, 2, 2])).all()
