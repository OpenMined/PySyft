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
    raise ValueError("patient 4171 is positive")


def outer():
    inner()


outer()
"""

CHAINED = """\
try:
    raise KeyError("patient 4171")
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


def test_record_holds_every_frame_of_the_crash(code_root):
    record = frames_from_stderr(run(code_root, CRASH), code_root)
    assert record["chain"][0]["type"] == "ValueError"
    assert record["chain"][0]["frames"] == [
        {"file": "main.py", "line": 9},
        {"file": "main.py", "line": 6},
        {"file": "main.py", "line": 2},
    ]


def test_record_never_holds_the_exception_message(code_root):
    record = frames_from_stderr(run(code_root, CRASH), code_root)
    assert "positive" not in json.dumps(record)
    assert "4171" not in json.dumps(record)


def test_frame_holds_only_a_file_and_a_line(code_root):
    record = frames_from_stderr(run(code_root, CRASH), code_root)
    assert set(record["chain"][0]) == {"type", "frames"}
    assert set(record["chain"][0]["frames"][0]) == {"file", "line"}


def test_chained_exception_puts_the_last_failure_first(code_root):
    record = frames_from_stderr(run(code_root, CHAINED), code_root)
    assert [entry["type"] for entry in record["chain"]] == ["RuntimeError", "KeyError"]
    assert "patient" not in json.dumps(record)


def test_library_frame_keeps_the_path_the_interpreter_printed(code_root):
    source = "import json\njson.loads('{')\n"
    record = frames_from_stderr(run(code_root, source), code_root)
    files = [f["file"] for f in record["chain"][0]["frames"]]
    assert files[0] == "main.py"
    assert any(f.endswith("json/decoder.py") for f in files)


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
def test_stderr_without_a_traceback_yields_no_record(stderr, code_root):
    assert frames_from_stderr(stderr, code_root) is None


def test_install_output_before_the_traceback_is_ignored(code_root):
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


def test_a_coloured_traceback_parses(code_root):
    """Python 3.13 and later colour the traceback when the environment asks."""
    coloured = (
        "Traceback (most recent call last):\n"
        '  File \x1b[35m"main.py"\x1b[0m, line \x1b[35m2\x1b[0m, in \x1b[35mf\x1b[0m\n'
        "\x1b[1;35mValueError\x1b[0m: \x1b[35mpatient 4171\x1b[0m\n"
    )
    record = frames_from_stderr(coloured, code_root)
    assert record["chain"][0]["type"] == "ValueError"
    assert record["chain"][0]["frames"] == [{"file": "main.py", "line": 2}]
    assert "4171" not in json.dumps(record)


def test_a_block_without_a_type_line_falls_back(code_root):
    block = 'Traceback (most recent call last):\n  File "main.py", line 1, in f\n'
    record = frames_from_stderr(block, code_root)
    assert record["chain"][0]["type"] == FALLBACK_TYPE


# -- the file the runner writes -------------------------------------------------


def test_write_frames_record_writes_the_staged_file(tmp_path, code_root):
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
