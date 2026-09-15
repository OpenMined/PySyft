"""Run a job entrypoint and record an uncaught exception as plain data.

This file runs inside the job process. It imports the standard library only,
because the job virtual environment does not hold ``syft_job``.

The record is untrusted. The job controls this process, so the job can write any
string into the record or replace the file. ``syft_job.traceback_capture``
checks every field before a party reads it.
"""

import json
import os
import runpy
import sys
import traceback

RAW_TRACE_PATH_ENV = "SYFT_JOB_RAW_TRACE_PATH"
MAX_CHAIN = 5
MAX_FRAMES = 64


def raw_record(exc: BaseException) -> dict:
    """Return the exception chain as plain data.

    The record holds the class names and the frame positions. It never holds the
    exception message, the source text, or the local variables.
    """
    chain = []
    seen = set()
    while exc is not None and id(exc) not in seen and len(chain) < MAX_CHAIN:
        seen.add(id(exc))
        frames = traceback.extract_tb(exc.__traceback__)[:MAX_FRAMES]
        chain.append(
            {
                "mro": [cls.__name__ for cls in type(exc).__mro__],
                "frames": [
                    {"filename": f.filename, "lineno": f.lineno} for f in frames
                ],
            }
        )
        exc = exc.__cause__ or exc.__context__
    return {"chain": chain}


def main(argv: list) -> None:
    entrypoint = argv[1]
    sys.argv = argv[1:]
    try:
        runpy.run_path(entrypoint, run_name="__main__")
    except SystemExit:
        # A deliberate exit carries no failure position.
        raise
    except BaseException as exc:
        # Observe the exception, then let it propagate unchanged. The process
        # must keep the exit code and the stderr traceback it would have had.
        path = os.environ.get(RAW_TRACE_PATH_ENV)
        if path:
            try:
                with open(path, "w") as f:
                    json.dump(raw_record(exc), f)
            except OSError:
                pass
        raise


if __name__ == "__main__":
    main(sys.argv)
