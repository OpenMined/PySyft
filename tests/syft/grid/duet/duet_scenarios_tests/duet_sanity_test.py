# stdlib
from multiprocessing.synchronize import Barrier

# syft absolute
import syft as sy


def do_test(barrier: Barrier, port: int) -> None:
    duet = sy.launch_duet(loopback=True, network_url=f"http://0.0.0.0:{port}/")
    _ = sy.lib.python.List([1, 2, 3]).send(duet)
    barrier.wait()


def ds_test(barrier: Barrier, port: int) -> None:
    _ = sy.join_duet(loopback=True, network_url=f"http://0.0.0.0:{port}/")
    barrier.wait()


test_scenario_sanity = (do_test, ds_test)
