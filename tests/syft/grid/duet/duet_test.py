# stdlib
from multiprocessing import Barrier
from multiprocessing import Pipe
from multiprocessing import Process
import socket
from time import sleep
import traceback
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

# syft relative
from .duet_scenarios_tests import register_duet_scenarios
from .signaling_server_test import run

port = 21000
grid_proc = Process(target=run, args=(port,))
grid_proc.start()

registered_tests: List[Tuple[Callable, Callable]] = []
register_duet_scenarios(registered_tests)


class SyftTestProcess(Process):
    def __init__(self, *args: Any, **kwargs: Any):
        Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = Pipe()
        self._exception = None

    def run(self) -> None:
        try:
            Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))

    @property
    def exception(self) -> Optional[tuple]:
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


def test_duet() -> None:
    # let the flask server init:
    sleep(5)

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
