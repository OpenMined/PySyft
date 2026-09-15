"""Turn an untrusted traceback record into a bounded one.

The job process writes the raw record, so a job chooses every string in it. A
job can put private data in a file name, in a function name, or in an exception
message. Therefore only these fields reach a party:

- a path that resolves to a file in the approved code bundle, or a fixed label;
- a line number inside that file;
- an exception type that is a builtin.

The function names, the exception messages, and the local variables never reach
a party.
"""

from __future__ import annotations

import builtins
import json
import os
from pathlib import Path
from typing import Optional

from syft_job._trace_runner import RAW_TRACE_PATH_ENV  # noqa: F401  (re-export)

RAW_TRACE_FILENAME = "_raw_traceback.json"
FRAMES_FILENAME = "traceback_frames.json"
TRACE_RUNNER_ENV = "SYFT_JOB_TRACE_RUNNER"

EXTERNAL_LABEL = "<external>"
FALLBACK_TYPE = "Exception"
MAX_CHAIN = 5
MAX_FRAMES = 64
BUNDLE_FILENAME = "code_bundle.json"
RUNNER_DIRS = frozenset({".venv", "__pycache__"})
MAX_BUNDLE_FILES = 2000
MAX_BUNDLE_FILE_BYTES = 5 * 1024 * 1024


def trace_runner_path() -> Path:
    """Return the path of the wrapper script that the job process runs."""
    return Path(__file__).with_name("_trace_runner.py")


def _builtin_exception_name(mro: object) -> str:
    """Return the first name in ``mro`` that names a builtin exception.

    The class names come from the job, so each name must match a builtin before
    it reaches a party. A custom class therefore reports its nearest builtin
    base, such as ``ValueError`` for ``class Secret(ValueError)``.
    """
    if not isinstance(mro, list):
        return FALLBACK_TYPE
    for name in mro:
        if not isinstance(name, str):
            continue
        cls = getattr(builtins, name, None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            return name
    return FALLBACK_TYPE


def _line_count(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def snapshot_code_bundle(code_root: Path) -> dict[str, int]:
    """Record the approved files and their lengths, before the job runs.

    Returns a map of a path relative to ``code_root`` to a line count.

    The job writes into its own code directory: ``run.sh`` builds a virtual
    environment there. A job can therefore plant a file whose name carries
    private data, then raise from it. The record must come from the tree the
    parties approved, so this runs before execution.
    """
    bundle: dict[str, int] = {}
    if not code_root.is_dir():
        return bundle
    try:
        root = code_root.resolve()
    except OSError:
        return bundle

    for path in sorted(root.rglob("*")):
        if len(bundle) >= MAX_BUNDLE_FILES:
            break
        if not path.is_file() or path.is_symlink():
            continue
        name = path.relative_to(root).as_posix()
        # run.sh builds the virtual environment here, so these paths belong to
        # the runner and never to the approved bundle.
        if any(part in RUNNER_DIRS for part in name.split("/")[:-1]):
            continue
        try:
            if path.stat().st_size > MAX_BUNDLE_FILE_BYTES:
                continue
            bundle[name] = _line_count(path)
        except OSError:
            continue
    return bundle


def _safe_frame(frame: object, code_root: Path, bundle: dict) -> dict:
    """Return one frame, or ``<external>`` when ``bundle`` does not hold it."""
    external = {"file": EXTERNAL_LABEL, "line": None}
    if not isinstance(frame, dict):
        return external

    filename = frame.get("filename")
    lineno = frame.get("lineno")
    if not isinstance(filename, str) or not filename:
        return external

    try:
        if os.path.isabs(filename):
            candidate = Path(filename).resolve()
        else:
            candidate = (code_root / filename).resolve()
    except OSError:
        return external

    if not candidate.is_relative_to(code_root):
        return external

    name = candidate.relative_to(code_root).as_posix()
    total = bundle.get(name)
    if total is None:
        # The job planted this file after the parties approved the bundle.
        return external
    if not isinstance(lineno, int) or isinstance(lineno, bool):
        return external
    if lineno < 1 or lineno > total:
        # A line outside the file means the job forged the position.
        return external

    return {"file": name, "line": lineno}


def sanitize_raw_trace(
    raw: object, code_root: Path, bundle: dict[str, int]
) -> Optional[dict]:
    """Return the bounded record, or None when ``raw`` holds no usable chain.

    ``code_root`` is the directory of the approved code bundle, and ``bundle``
    is the snapshot that ``snapshot_code_bundle`` took before the job ran. A
    frame counts as safe only when the snapshot holds its file and its line.
    """
    if not isinstance(raw, dict):
        return None
    chain = raw.get("chain")
    if not isinstance(chain, list) or not chain:
        return None

    try:
        root = code_root.resolve()
    except OSError:
        return None

    safe_chain = []
    for entry in chain[:MAX_CHAIN]:
        if not isinstance(entry, dict):
            continue
        frames = entry.get("frames")
        frames = frames[:MAX_FRAMES] if isinstance(frames, list) else []
        safe_chain.append(
            {
                "type": _builtin_exception_name(entry.get("mro")),
                "frames": [_safe_frame(f, root, bundle) for f in frames],
            }
        )

    if not safe_chain:
        return None
    return {"chain": safe_chain}


def write_frames_record(
    raw_path: Path, out_dir: Path, code_root: Path, bundle: dict[str, int]
) -> Optional[Path]:
    """Read the raw record, check it, and write ``traceback_frames.json``.

    ``out_dir`` is the staging directory, therefore the record waits there until
    a party releases it.

    Returns the path of the written file, or None when the job wrote no usable
    record. A job that writes nothing, or writes invalid JSON, produces no file.
    """
    if not raw_path.is_file():
        return None
    try:
        with open(raw_path, "r") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    record = sanitize_raw_trace(raw, code_root, bundle)
    if record is None:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / FRAMES_FILENAME
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    return out_path


def load_or_create_bundle(bundle_path: Path, code_root: Path) -> dict[str, int]:
    """Return the approved tree, recorded once before the first run.

    A rerun executes the same bundle, and the first run leaves files behind.
    Reading a stored record therefore keeps a later run from approving a file
    that the parties never saw.
    """
    if bundle_path.is_file():
        try:
            stored = json.loads(bundle_path.read_text())
        except (OSError, json.JSONDecodeError):
            stored = None
        if isinstance(stored, dict):
            return {
                name: count
                for name, count in stored.items()
                if isinstance(name, str)
                and isinstance(count, int)
                and not isinstance(count, bool)
            }

    bundle = snapshot_code_bundle(code_root)
    try:
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        bundle_path.write_text(json.dumps(bundle))
    except OSError:
        pass
    return bundle


def write_no_failure_record(out_dir: Path) -> Path:
    """Write a record that says the run did not fail.

    An empty chain never comes from a crash, because a record with no chain is
    dropped. It therefore marks a run that produced no failure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / FRAMES_FILENAME
    with open(out_path, "w") as f:
        json.dump({"chain": []}, f, indent=2)
    return out_path
