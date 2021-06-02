# stdlib
from multiprocessing import set_start_method
import socket
import time
from typing import Callable
from typing import List
from typing import Tuple

# third party
import pytest

# syft relative
from .duet_scenarios_tests import register_duet_scenarios
from .process_test import SyftTestProcess

set_start_method("spawn", force=True)


registered_tests: List[Tuple[str, Callable, Callable]] = []
register_duet_scenarios(registered_tests)


@pytest.mark.slow
def test_duet(signaling_server: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        assert s.connect_ex(("localhost", signaling_server)) == 0

    for testcase, do, ds in registered_tests:
        start = time.time()

        do_proc = SyftTestProcess(target=do, args=(signaling_server,))
        do_proc.start()

        ds_proc = SyftTestProcess(target=ds, args=(signaling_server,))
        ds_proc.start()

        ds_proc.join(30)

        do_proc.terminate()

        if do_proc.exception:
            exception, tb = do_proc.exception
            raise Exception(tb) from exception

        if ds_proc.exception:
            exception, tb = ds_proc.exception
            raise Exception(tb) from exception

        if ds_proc.is_alive():
            ds_proc.terminate()
            raise Exception(f"ds_proc is hanged in {testcase}")

        print(f"test {testcase} passed in {time.time() - start} seconds")
