# stdlib
import time

# third party
import numpy as np
import pytest

# syft absolute
import syft as sy
from tests.integration.conftest import TestNodeData

DOMAIN1_PORT = 9082


# This fails on Windows CI even though the same code works in a Jupyter Notebook
# on the same Windows CI machine.
@pytest.mark.xfail
@pytest.mark.network
def test_large_message_size(test_domain_1: TestNodeData) -> None:
    domain_client = sy.login(
        email="info@openmined.org", password="changethis", url=test_domain_1.grid_api_url
    )

    ndim = 11300  # 510.76 MB

    # rabbitmq recommends max is 512 MB
    # rabbitmq.conf has max_message_size = 536870912

    x = np.zeros((ndim, ndim), dtype=np.int32)
    size = str(x.shape)
    mb_size = x.nbytes / 1000 / 1000
    mb = f"{mb_size} MB"

    try:
        start_time = time.time()
        print(f"Sending {mb} sized message")
        x_ptr = x.send(domain_client, tags=[size, mb])
        x_ptr.block_with_timeout(180)
        total_time = time.time() - start_time
        print(f"Took {total_time}")
        data_rate = mb_size / total_time
        print(f"Send transfer rate: {data_rate}")
    except Exception as e:
        total_time = time.time() - start_time
        print(f"Failed to send {size} in {total_time}. {e}")
        raise e

    try:
        start_time = time.time()
        back = x_ptr.get()
        assert (back == x).all()
        total_time = time.time() - start_time
        data_rate = mb_size / total_time
        print(f"Return transfer rate: {data_rate}")
    except Exception as e:
        print(f"Failed to get data back. {e}")
        raise e
