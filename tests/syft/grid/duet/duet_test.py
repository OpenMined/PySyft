# stdlib
from multiprocessing import Barrier
from multiprocessing import Process
import socket
from time import sleep
from typing import Callable
from typing import List
from typing import Tuple

# syft relative
from .duet_scenarios_tests import register_duet_scenarios
from .signaling_server_test import run
from .syft_process_test import SyftTestProcess

port = 21000
grid_proc = Process(target=run, args=(port,))
grid_proc.start()

registered_tests: List[Tuple[Callable, Callable]] = []
register_duet_scenarios(registered_tests)


def test_duet() -> None:
    # let the flask server init:
    sleep(1)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        assert s.connect_ex(("localhost", port)) == 0

    for do, ds in registered_tests:
        barrier = Barrier(2, action=lambda: barrier.reset())

        do_proc = SyftTestProcess(target=do, args=(barrier, port))
        do_proc.start()

        ds_proc = SyftTestProcess(target=ds, args=(barrier, port))
        ds_proc.start()

        do_proc.join()
        ds_proc.join()

        if do_proc.exception:
            exception, tb = do_proc.exception
            raise Exception(tb) from exception

        if ds_proc.exception:
            exception, tb = ds_proc.exception
            raise Exception(tb) from exception

    grid_proc.terminate()
    grid_proc.join()
