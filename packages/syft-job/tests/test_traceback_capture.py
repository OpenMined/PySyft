"""``traceback_frames`` reports where a job failed, and never the message."""

import json
import subprocess
import sys

import pytest

from syft_job.traceback_capture import (
    FALLBACK_TYPE,
    FRAMES_FILENAME,
    MAX_CHAIN,
    MAX_FRAMES,
    frames_from_stderr,
    write_frames_record,
)

CRASH = """\
def inner():
    raise ValueError("account 88213 holds 4120550")


def outer():
    inner()


outer()
"""

EXCEPTION_GROUP = """\
def boom():
    raise ValueError("account 88213 holds 4120550")


try:
    boom()
except ValueError as exc:
    raise ExceptionGroup("eg", [exc])
"""

CHAINED = """\
try:
    raise KeyError("account 88213")
except KeyError as exc:
    raise RuntimeError("wrapped") from exc
"""


@pytest.fixture
def code_root(tmp_path):
    root = tmp_path / "code"
    root.mkdir()
    return root


def run(code_root, source):
    """Run a script the way run.sh does, from inside code/, and return stderr."""
    (code_root / "main.py").write_text(source)
    proc = subprocess.run(
        [sys.executable, "main.py"],
        cwd=code_root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    return proc.stderr


# -- what the record holds ------------------------------------------------------


def test_record_holds_every_crash_frame(code_root):
    record = frames_from_stderr(run(code_root, CRASH), code_root)
    assert record["chain"][0]["type"] == "ValueError"
    assert record["chain"][0]["frames"] == [
        {"file": "main.py", "line": 9},
        {"file": "main.py", "line": 6},
        {"file": "main.py", "line": 2},
    ]


def test_record_never_holds_exception_message(code_root):
    record = frames_from_stderr(run(code_root, CRASH), code_root)
    assert "4120550" not in json.dumps(record)
    assert "88213" not in json.dumps(record)


def test_frame_holds_only_file_and_line(code_root):
    record = frames_from_stderr(run(code_root, CRASH), code_root)
    assert set(record["chain"][0]) == {"type", "frames"}
    assert set(record["chain"][0]["frames"][0]) == {"file", "line"}


def test_chained_exception_puts_last_failure_first(code_root):
    record = frames_from_stderr(run(code_root, CHAINED), code_root)
    assert [entry["type"] for entry in record["chain"]] == ["RuntimeError", "KeyError"]
    assert "account" not in json.dumps(record)


def test_library_frame_keeps_printed_path(code_root):
    source = "import json\njson.loads('{')\n"
    record = frames_from_stderr(run(code_root, source), code_root)
    files = [f["file"] for f in record["chain"][0]["frames"]]
    assert files[0] == "main.py"
    assert any(f.endswith("json/decoder.py") for f in files)


def test_header_without_frames_yields_no_record(code_root):
    """A job that prints the header itself must not reach the record.

    The type was previously read from anywhere in the block, so a line of
    ordinary stderr became the exception type.
    """
    stderr = "Traceback (most recent call last):\naccount_88213_balance: 4120550\n"
    assert frames_from_stderr(stderr, code_root) is None


def test_output_after_traceback_not_read_as_type(code_root):
    stderr = run(code_root, CRASH) + "account_88213_balance: 4120550\n"
    record = frames_from_stderr(stderr, code_root)
    assert record["chain"][0]["type"] == "ValueError"
    assert "account_88213_balance" not in json.dumps(record)


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="ExceptionGroup needs Python 3.11"
)
def test_exception_group_keeps_its_frames_and_leaks_no_output(code_root):
    """An ExceptionGroup prints each frame behind a '|' gutter.

    No frame matched before, so the whole block was scanned for a type and a
    later line of job output supplied it.
    """
    stderr = run(code_root, EXCEPTION_GROUP) + "account_88213_balance: 4120550\n"
    record = frames_from_stderr(stderr, code_root)

    blob = json.dumps(record)
    assert "account_88213_balance" not in blob
    assert "88213" not in blob
    assert any(
        frame["file"] == "main.py"
        for entry in record["chain"]
        for frame in entry["frames"]
    )
    assert {entry["type"] for entry in record["chain"]} <= {
        "ExceptionGroup",
        "ValueError",
    }


# -- input that carries no failure ----------------------------------------------


def test_clean_run_leaves_no_record(code_root):
    (code_root / "main.py").write_text("print('ok')\n")
    proc = subprocess.run(
        [sys.executable, "main.py"], cwd=code_root, capture_output=True, text=True
    )
    assert proc.returncode == 0
    assert frames_from_stderr(proc.stderr, code_root) is None


def test_deliberate_exit_leaves_no_record(code_root):
    source = "import sys\nsys.exit(3)\n"
    (code_root / "main.py").write_text(source)
    proc = subprocess.run(
        [sys.executable, "main.py"], cwd=code_root, capture_output=True, text=True
    )
    assert proc.returncode == 3
    assert frames_from_stderr(proc.stderr, code_root) is None


@pytest.mark.parametrize("stderr", ["", "some log output\n", "Traceback: not really\n"])
def test_stderr_without_traceback_yields_no_record(stderr, code_root):
    assert frames_from_stderr(stderr, code_root) is None


def test_install_output_before_traceback_ignored(code_root):
    noisy = "+ pandas==2.0.0\n+ numpy==1.26\n" + run(code_root, CRASH)
    record = frames_from_stderr(noisy, code_root)
    assert record["chain"][0]["type"] == "ValueError"


# -- size limits ----------------------------------------------------------------


def test_chain_and_frame_count_are_capped(code_root):
    block = (
        "Traceback (most recent call last):\n"
        + "".join(f'  File "main.py", line {i}, in f\n' for i in range(1, 200))
        + "ValueError\n"
    )
    record = frames_from_stderr(block * 20, code_root)
    assert len(record["chain"]) == MAX_CHAIN
    assert len(record["chain"][0]["frames"]) == MAX_FRAMES


def test_coloured_traceback_parses(code_root):
    """Python 3.13 and later colour the traceback when the environment asks."""
    coloured = (
        "Traceback (most recent call last):\n"
        '  File \x1b[35m"main.py"\x1b[0m, line \x1b[35m2\x1b[0m, in \x1b[35mf\x1b[0m\n'
        "\x1b[1;35mValueError\x1b[0m: \x1b[35maccount 88213\x1b[0m\n"
    )
    record = frames_from_stderr(coloured, code_root)
    assert record["chain"][0]["type"] == "ValueError"
    assert record["chain"][0]["frames"] == [{"file": "main.py", "line": 2}]
    assert "88213" not in json.dumps(record)


def test_frame_cap_keeps_innermost_frames(code_root):
    """Python prints the outermost frame first, so the job stopped at the last.

    The cap previously kept the first frames, which drops the failure site on
    exactly the deep-recursion case that motivates the cap.
    """
    block = (
        "Traceback (most recent call last):\n"
        + "".join(f'  File "main.py", line {i}, in f\n' for i in range(1, 200))
        + "ValueError\n"
    )
    frames = frames_from_stderr(block, code_root)["chain"][0]["frames"]
    assert len(frames) == MAX_FRAMES
    assert frames[-1]["line"] == 199
    assert frames[0]["line"] == 199 - MAX_FRAMES + 1


def test_block_without_type_line_falls_back(code_root):
    block = 'Traceback (most recent call last):\n  File "main.py", line 1, in f\n'
    record = frames_from_stderr(block, code_root)
    assert record["chain"][0]["type"] == FALLBACK_TYPE


# -- the file the runner writes -------------------------------------------------


def test_write_frames_record_writes_staged_file(tmp_path, code_root):
    stderr_path = tmp_path / "stderr.txt"
    stderr_path.write_text(run(code_root, CRASH))
    staging = tmp_path / "staging"

    out = write_frames_record(stderr_path, staging, code_root)
    assert out == staging / FRAMES_FILENAME
    assert json.loads(out.read_text())["chain"][0]["type"] == "ValueError"


def test_no_stderr_file_means_no_record(tmp_path, code_root):
    staging = tmp_path / "staging"
    assert write_frames_record(tmp_path / "absent.txt", staging, code_root) is None
    assert not (staging / FRAMES_FILENAME).exists()
