# stdlib
from collections.abc import Callable
from functools import wraps
from subprocess import CalledProcessError
from subprocess import CompletedProcess
from subprocess import PIPE
from subprocess import Popen
import threading
from typing import Any

__all__ = ["run_command", "check_returncode", "CalledProcessError", "CompletedProcess"]


def NOOP(x: Any) -> None:
    return None


def run_command(
    command: str,
    working_dir: str | None = None,
    stdout: int = PIPE,
    stderr: int = PIPE,
    stream_output: dict | None = None,
    dryrun: bool = False,
) -> CompletedProcess:
    """
    Run a command in a subprocess.

    Args:
        command       (str): The command to run.
        working_dir   (str): The working directory to run the command in.
        stdout        (int): The stdout file descriptor. Defaults to subprocess.PIPE.
        stderr        (int): The stderr file descriptor. Defaults to subprocess.PIPE.
        stream_output (dict): A dict containing callbacks to process stdout and stderr in real-time.
        dryrun        (bool): If True, the command will not be executed.

    Returns:
        A CompletedProcess object.

    Example:
        >>> from syftcli.core.proc import run_command
        >>> result = run_command("echo 'hello world'")
        >>> result.check_returncode()
        >>> result.stdout
    """

    if dryrun:
        return CompletedProcess(command, 0, stdout="", stderr="")

    try:
        process = Popen(
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

        else:
            _out, _err = process.communicate()

        return CompletedProcess(command, process.returncode, _out, _err)
    except CalledProcessError as e:
        return CompletedProcess(command, e.returncode, e.stdout, e.stderr)


def check_returncode(
    func: Callable[..., CompletedProcess],
) -> Callable[..., CompletedProcess]:
    """A decorator to wrap run_command and check the return code."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> CompletedProcess:
        result = func(*args, **kwargs)
        result.check_returncode()
        return result

    return wrapper
