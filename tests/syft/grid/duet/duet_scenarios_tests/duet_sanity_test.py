# stdlib
from typing import List
from multiprocessing.synchronize import Barrier
import time

# syft absolute
import syft as sy


def do_test(barriers: List[Barrier], port: int) -> None:
    duet = sy.launch_duet(loopback=True, network_url=f"http://0.0.0.0:{port}/")
    _ = sy.lib.python.List([1, 2, 3]).send(duet, searchable=True, tags=["data"])
    barriers[0].wait()
    time.sleep(1)
    barriers[1].wait()


def ds_test(barriers: List[Barrier], port: int) -> None:
    duet = sy.join_duet(loopback=True, network_url=f"http://0.0.0.0:{port}/")
    barriers[0].wait()
    data_ptr = duet.store["data"]
    data = data_ptr.get(request_block=True)

    assert data == [1, 2, 3]

    barriers[1].wait()


test_scenario_sanity = (do_test, ds_test)
