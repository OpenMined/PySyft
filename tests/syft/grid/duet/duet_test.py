# stdlib
import atexit
from multiprocessing import Manager, log_to_stderr, Process

import socket
from time import sleep
from typing import Callable
from typing import List
from typing import Tuple

# syft relative
from .duet_scenarios_tests import register_duet_scenarios
import pytest
from .signaling_server_test import run
from .process_test import SyftTestProcess

log_to_stderr()

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

@pytest.mark.skip
def test_duet() -> None:
    # let the flask server init:
    sleep(5)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        assert s.connect_ex(("localhost", port)) == 0

    for do, ds in registered_tests:
        mgr = Manager()
        # the testcases can use barriers at the same index to sync their operations
        barriers = [mgr.Barrier(2, timeout=10)] * 64  # type: ignore

        do_proc = SyftTestProcess(target=do, args=(barriers, port))
        do_proc.start()

        ds_proc = SyftTestProcess(target=ds, args=(barriers, port))
        ds_proc.start()

        do_proc.join(30)
        ds_proc.join(30)

        if do_proc.is_alive():
            do_proc.terminate()
            raise Exception("do_proc is hanged")
        if ds_proc.is_alive():
            ds_proc.terminate()
            raise Exception("ds_proc is hanged")

        if do_proc.exception:
            exception, tb = do_proc.exception
            raise Exception(tb) from exception

        if ds_proc.exception:
            exception, tb = ds_proc.exception
            raise Exception(tb) from exception
