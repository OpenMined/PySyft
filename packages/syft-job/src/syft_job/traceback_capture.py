"""Record where a job failed, from the traceback the job printed to stderr.

The record holds the file and the line of each frame, and the exception type.
It never holds the exception message, because ``traceback_frames`` discloses
where a job failed and ``logs`` discloses what the job wrote. A party that
needs the message asks for ``logs`` as well.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

FRAMES_FILENAME = "traceback_frames.json"
FALLBACK_TYPE = "Exception"

# Size limits, not disclosure limits. Deep recursion prints thousands of
# frames, and the record is a summary of where the job stopped.
MAX_CHAIN = 5
MAX_FRAMES = 50

TRACEBACK_HEADER = "Traceback (most recent call last):"
# Python 3.13 and later colour the traceback when the environment asks for it.
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_FRAME = re.compile(r'^  File "(?P<file>.+)", line (?P<line>\d+)', re.M)
_TYPE = re.compile(r"^(?P<type>[A-Za-z_][A-Za-z0-9_.]*)(?::|$)", re.M)


def _frame_file(raw: str, code_root: Path) -> str:
    """The frame path, relative to ``code_root`` when it points inside it.

    The job runs with ``code/`` as its working directory, so a frame in the
    submitted code is already relative. A frame in a library is not, and it
    keeps the path the interpreter printed.
    """
    path = Path(raw)
    if not path.is_absolute():
        path = code_root / raw
    try:
        return path.resolve().relative_to(code_root).as_posix()
    except (OSError, ValueError):
        return raw


def _frames(block: str, code_root: Path) -> list[dict]:
    return [
        {"file": _frame_file(m["file"], code_root), "line": int(m["line"])}
        for m in list(_FRAME.finditer(block))[:MAX_FRAMES]
    ]


def _exception_type(block: str) -> str:
    """The type name on the line that closes a traceback block."""
    tail = block[max((m.end() for m in _FRAME.finditer(block)), default=0) :]
    match = _TYPE.search(tail)
    return match["type"] if match else FALLBACK_TYPE


def frames_from_stderr(stderr: str, code_root: Path) -> Optional[dict]:
    """Return the failure positions in ``stderr``, or None when it holds none.

    Python prints a chained exception oldest first, therefore the chain is
    reversed: ``chain[0]`` is the exception that stopped the job.
    """
    blocks = _ANSI.sub("", stderr).split(TRACEBACK_HEADER)[1:]
    if not blocks:
        return None
    try:
        root = code_root.resolve()
    except OSError:
        root = code_root
    return {
        "chain": [
            {"type": _exception_type(b), "frames": _frames(b, root)}
            for b in reversed(blocks[-MAX_CHAIN:])
        ]
    }


def write_frames_record(
    stderr_path: Path, out_dir: Path, code_root: Path
) -> Optional[Path]:
    """Read the staged stderr and write ``traceback_frames.json`` beside it.

    ``out_dir`` is the staging directory, therefore the record waits there
    until a party releases it.

    Returns the path written, or None when the job printed no traceback.
    """
    try:
        stderr = stderr_path.read_text(errors="replace")
    except OSError:
        return None

    record = frames_from_stderr(stderr, code_root)
    if record is None:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / FRAMES_FILENAME
    out_path.write_text(json.dumps(record, indent=2))
    return out_path
