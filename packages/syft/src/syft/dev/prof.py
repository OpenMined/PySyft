# stdlib
import contextlib
import os
import signal
import subprocess
import tempfile
import time


@contextlib.contextmanager
def pyspy():
    fd, fname = tempfile.mkstemp(".svg")
    os.close(fd)

    command = [
        "sudo",
        "-S",
        "py-spy",
        "record",
        "-r",
        "100",
        "-o",
        fname,
        "--pid",
        str(os.getpid()),
    ]
    process = subprocess.Popen(command, preexec_fn=os.setsid)

    start_time = time.time()
    yield process
    end_time = time.time()

    print(f"Execution time: {end_time - start_time}")
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
        os.chmod(fname, 0o444)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    with pyspy():
        print("Goodbye")
