# stdlib
from multiprocessing import Pipe
from multiprocessing import Process
import os
import sys
import traceback
from typing import Any
from typing import Optional


class SyftTestProcess(Process):
    def __init__(self, *args: Any, **kwargs: Any):
        Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = Pipe()
        self._exception = None

    def run(self) -> None:
        try:
            sys.stdout = open(os.devnull, "w")
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
