# stdlib
import subprocess
import sys
import threading
from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import TypeAlias

CmdResult: TypeAlias = Tuple[str, str, int]


def NOOP(x: Any) -> None:
    return None


def run_command(
    command: str,
    working_dir: Optional[str] = None,
    stdout: int = subprocess.PIPE,
    stderr: int = subprocess.PIPE,
    stream_output: Optional[dict] = None,
) -> CmdResult:
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=working_dir,
            stdout=stdout,
            stderr=stderr,
            text=True,
            universal_newlines=True,
        )

        # subprocess.DEVNULL will not work with stream_output=True
        if stream_output:
            _out = ""
            _err = ""

            cb_stdout = stream_output.get("cb_stdout", NOOP)
            cb_stderr = stream_output.get("cb_stderr", NOOP)

            def process_output(stream: str, callback: Callable) -> None:
                if not stream:
                    return

                for line in stream:
                    callback(line)

            # Start capturing and processing stdout and stderr in real-time
            stdout_thread = threading.Thread(
                target=process_output,
                args=(process.stdout, cb_stdout),
            )
            stderr_thread = threading.Thread(
                target=process_output,
                args=(process.stderr, cb_stderr),
            )

            stdout_thread.start()
            stderr_thread.start()

            process.wait()

            stdout_thread.join()
            stderr_thread.join()

            return _out, _err, process.returncode
        else:
            _out, _err = process.communicate()
            return (_out, _err, process.returncode)
    except subprocess.CalledProcessError as e:
        return (e.stdout, e.stderr, e.returncode)


def handle_error(result: CmdResult, exit_on_error: bool = False) -> None:
    stdout, stderr, code = result
    if code != 0:
        print("Error:", stderr)
        exit_on_error and sys.exit(-1)
