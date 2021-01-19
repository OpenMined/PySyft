# stdlib
from multiprocessing.synchronize import Barrier
from typing import List

# syft absolute
import syft as sy


def do_test(barriers: List[Barrier], port: int) -> None:
    duet = sy.launch_duet(loopback=True, network_url=f"http://127.0.0.1:{port}/")
    _ = sy.lib.python.List([1, 2, 3]).send(duet)
    barriers[0].wait()


def ds_test(barriers: List[Barrier], port: int) -> None:
    _ = sy.join_duet(loopback=True, network_url=f"http://127.0.0.1:{port}/")
    barriers[0].wait()


test_scenario_init = (do_test, ds_test)
