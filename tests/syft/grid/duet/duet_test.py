# stdlib
import atexit
from multiprocessing import set_start_method, Process

# from pathos.multiprocessing import ProcessPool

# import socket
from time import sleep
from typing import Callable
from typing import List
from typing import Tuple

# syft relative
from .duet_scenarios_tests import register_duet_scenarios

from .signaling_server_test import run

set_start_method("spawn")

port = 21000
grid_proc = Process(target=run, args=(port,))
grid_proc.start()


def grid_cleanup() -> None:
    global grid_proc
    grid_proc.terminate()
    grid_proc.join()


atexit.register(grid_cleanup)

registered_tests: List[Tuple[Callable, Callable]] = []
register_duet_scenarios(registered_tests)


def test_duet() -> None:
    # let the flask server init:
    sleep(5)

    # pool = ProcessPool(nodes=2)

    # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #    assert s.connect_ex(("localhost", port)) == 0

    # for do, ds in registered_tests:
    #    mgr = Manager()
    #    barrier = mgr.Barrier(2, timeout=20)  # type: ignore

    #   do_proc = pool.apipe(do, barrier, port)
    #    ds_proc = pool.apipe(ds, barrier, port)

    #    do_proc.get()
    #    ds_proc.get()

    # pool.close()
    # pool.terminate()
    # pool.join()
