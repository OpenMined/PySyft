# stdlib
import contextlib
import os
import signal
import subprocess  # nosec
import tempfile
import time
from typing import Callable, Iterator


@contextlib.contextmanager
def pyspy() -> Iterator[subprocess.Popen]:
    """Profile a block of code using py-spy. Intended for development purposes only.

    Example:
    ```
    with pyspy():
        # do some work
        a = [i for i in range(1000000)]
    ```
    """
    fd, fname = tempfile.mkstemp(".svg")
    os.close(fd)

    command = [
        "sudo",
        "-S",
        "py-spy",
        "record",
        "-r",
        "1000",
        "-o",
        fname,
        "--pid",
        str(os.getpid()),
    ]
    process = subprocess.Popen(command, preexec_fn=os.setsid)  # nosec

    start_time = time.time()
    yield process
    end_time = time.time()

    print(f"Execution time: {end_time - start_time}")
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
        os.chmod(fname, 0o444)
    except Exception as e:
        print(f"Error: {e}")


class FakeThread:
    """Convenience class for profiling code that should be run in a thread.
    Easy to swap with Thread when profiling is not needed and we want to run in the main thread.
    """

    def __init__(self, target: Callable, args: tuple = (), daemon: bool = True) -> None:
        self.target = target
        self.args = args
        self.daemon = daemon
        self.is_alive_flag = False

    def start(self) -> None:
        self.is_alive_flag = True
        self.target(*self.args)

    def is_alive(self) -> bool:
        return self.is_alive_flag

    def join(self) -> None:
        pass
