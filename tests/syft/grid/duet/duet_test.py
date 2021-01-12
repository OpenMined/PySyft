# stdlib
import atexit
from multiprocessing import Manager
from pathos.multiprocessing import ProcessPool
import socket
from time import sleep
from typing import Callable
from typing import List
from typing import Tuple

# syft relative
from .duet_scenarios_tests import register_duet_scenarios

from .signaling_server_test import run

registered_tests: List[Tuple[Callable, Callable]] = []
register_duet_scenarios(registered_tests)

pool = ProcessPool(nodes=3)


def cleanup() -> None:
    pool.close()
    pool.terminate()
    pool.join()


atexit.register(cleanup)

port = 21000
pool.amap(run, [port])


def test_duet() -> None:
    # let the flask server init:
    sleep(5)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        assert s.connect_ex(("localhost", port)) == 0

    for do, ds in registered_tests:
        mgr = Manager()
        barrier = mgr.Barrier(2, timeout=20)  # type: ignore

        do_proc = pool.apipe(do, barrier, port)
        ds_proc = pool.apipe(ds, barrier, port)

        do_proc.get()
        ds_proc.get()
