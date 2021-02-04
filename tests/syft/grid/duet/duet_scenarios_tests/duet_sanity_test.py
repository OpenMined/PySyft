# stdlib
from multiprocessing.synchronize import Barrier
import sys
import time
from typing import List


def do_test(barriers: List[Barrier], port: int) -> None:
    # syft absolute
    import syft as sy

    sy.logger.add(sys.stderr, "ERROR")

    duet = sy.launch_duet(loopback=True, network_url=f"http://127.0.0.1:{port}/")
    duet.requests.add_handler(action="accept")

    _ = sy.lib.python.List([1, 2, 3]).send(duet, searchable=True)

    sy.core.common.event_loop.loop.run_forever()


def ds_test(barriers: List[Barrier], port: int) -> None:
    # syft absolute
    import syft as sy

    sy.logger.add(sys.stderr, "ERROR")

    duet = sy.join_duet(loopback=True, network_url=f"http://127.0.0.1:{port}/")

    time.sleep(1)

    data_ptr = duet.store[0]
    data = data_ptr.get(request_block=True, delete_obj=False)

    assert data == [1, 2, 3]


test_scenario_sanity = ("test_scenario_sanity", do_test, ds_test)
