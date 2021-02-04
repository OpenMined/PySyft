# stdlib
from multiprocessing.synchronize import Barrier
import sys
import time
from typing import List


def do_test(barriers: List[Barrier], port: int) -> None:
    # third party
    import torch

    # syft absolute
    import syft as sy

    sy.logger.add(sys.stderr, "ERROR")

    duet = sy.launch_duet(loopback=True, network_url=f"http://127.0.0.1:{port}/")
    duet.requests.add_handler(action="accept")

    t = torch.randn(20).reshape(4, 5)
    _ = t.send(duet, searchable=True)

    sy.core.common.event_loop.loop.run_forever()


def ds_test(barriers: List[Barrier], port: int) -> None:
    # third party
    import torch

    # syft absolute
    import syft as sy

    sy.logger.add(sys.stderr, "ERROR")

    duet = sy.join_duet(loopback=True, network_url=f"http://127.0.0.1:{port}/")

    time.sleep(1)

    data = duet.store[0].get(request_block=True, delete_obj=False)

    assert data.shape == torch.Size([4, 5]), data.shape


test_scenario_torch_tensor_sanity = (
    "test_scenario_torch_tensor_sanity",
    do_test,
    ds_test,
)
