"""The sanitizer treats the raw record as attacker-controlled input."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from syft_job._trace_runner import raw_record
from syft_job.traceback_capture import (
    EXTERNAL_LABEL,
    FALLBACK_TYPE,
    FRAMES_FILENAME,
    MAX_CHAIN,
    MAX_FRAMES,
    sanitize_raw_trace,
    snapshot_code_bundle,
    trace_runner_path,
    write_frames_record,
)


def sanitize(raw, code_root):
    """Sanitize against a snapshot taken from the current tree."""
    return sanitize_raw_trace(raw, code_root, snapshot_code_bundle(code_root))


@pytest.fixture
def code_root(tmp_path):
    root = tmp_path / "code"
    root.mkdir()
    (root / "main.py").write_text("\n".join(f"line {i}" for i in range(1, 21)))
    return root


def frame(filename, lineno):
    return {"filename": filename, "lineno": lineno}


def chain(mro, frames):
    return {"chain": [{"mro": mro, "frames": frames}]}


# -- what a reader is allowed to see ------------------------------------------


def test_frame_in_the_bundle_keeps_its_file_and_line(code_root):
    out = sanitize(chain(["ValueError"], [frame("main.py", 12)]), code_root)
    assert out["chain"][0]["frames"] == [{"file": "main.py", "line": 12}]
    assert out["chain"][0]["type"] == "ValueError"


def test_absolute_path_in_the_bundle_is_relative_in_the_record(code_root):
    raw = chain(["KeyError"], [frame(str(code_root / "main.py"), 3)])
    out = sanitize(raw, code_root)
    assert out["chain"][0]["frames"] == [{"file": "main.py", "line": 3}]


def test_the_record_never_holds_a_message_or_a_function_name(code_root):
    raw = chain(["ValueError"], [frame("main.py", 1)])
    raw["chain"][0]["frames"][0]["name"] = "leak_do2_row_4171"
    raw["chain"][0]["message"] = "patient 4171 is positive"
    out = sanitize(raw, code_root)
    assert set(out["chain"][0]) == {"type", "frames"}
    assert set(out["chain"][0]["frames"][0]) == {"file", "line"}


# -- smuggling attempts --------------------------------------------------------


def test_a_forged_file_name_never_reaches_the_record(code_root):
    """A job picks the file name of a frame through compile()."""
    raw = chain(["ValueError"], [frame("DO2_row_4171_diagnosis_POSITIVE.py", 1)])
    out = sanitize(raw, code_root)
    assert out["chain"][0]["frames"] == [{"file": EXTERNAL_LABEL, "line": None}]


def test_a_path_outside_the_bundle_is_labelled(code_root, tmp_path):
    outside = tmp_path / "secret.py"
    outside.write_text("x = 1\n")
    out = sanitize(chain(["OSError"], [frame(str(outside), 1)]), code_root)
    assert out["chain"][0]["frames"] == [{"file": EXTERNAL_LABEL, "line": None}]


def test_a_traversal_path_stays_outside(code_root):
    raw = chain(["OSError"], [frame("../../../etc/passwd", 1)])
    out = sanitize(raw, code_root)
    assert out["chain"][0]["frames"][0]["file"] == EXTERNAL_LABEL


def test_a_line_past_the_end_of_the_file_is_dropped(code_root):
    """main.py holds 20 lines, so line 99999 is a forged position."""
    out = sanitize(chain(["ValueError"], [frame("main.py", 99999)]), code_root)
    assert out["chain"][0]["frames"] == [{"file": EXTERNAL_LABEL, "line": None}]


def test_a_custom_exception_reports_its_builtin_base(code_root):
    raw = chain(["DO2DataWasPositive", "ValueError", "Exception"], [])
    out = sanitize(raw, code_root)
    assert out["chain"][0]["type"] == "ValueError"


def test_an_all_custom_mro_falls_back(code_root):
    out = sanitize(chain(["SecretName", "AlsoSecret"], []), code_root)
    assert out["chain"][0]["type"] == FALLBACK_TYPE


def test_the_chain_and_the_frame_count_are_bounded(code_root):
    raw = {
        "chain": [
            {"mro": ["ValueError"], "frames": [frame("main.py", 1)] * 500}
            for _ in range(50)
        ]
    }
    out = sanitize(raw, code_root)
    assert len(out["chain"]) == MAX_CHAIN
    assert len(out["chain"][0]["frames"]) == MAX_FRAMES


@pytest.mark.parametrize(
    "raw",
    [None, {}, {"chain": []}, {"chain": "nope"}, [1, 2, 3], {"chain": [None]}],
)
def test_malformed_input_yields_no_record(raw, code_root):
    assert sanitize(raw, code_root) in (None, {"chain": []})


def test_a_non_integer_line_is_dropped(code_root):
    out = sanitize(chain(["ValueError"], [frame("main.py", "12")]), code_root)
    assert out["chain"][0]["frames"][0]["line"] is None


# -- the file the parent writes -------------------------------------------------


def test_write_frames_record_writes_only_checked_fields(tmp_path, code_root):
    raw_path = tmp_path / "raw.json"
    review = tmp_path / "review"
    raw_path.write_text(
        json.dumps(chain(["ValueError"], [frame("main.py", 2), frame("/etc/shadow", 1)]))
    )
    out = write_frames_record(
        raw_path, review, code_root, snapshot_code_bundle(code_root)
    )
    assert out == review / FRAMES_FILENAME
    record = json.loads(out.read_text())
    assert record["chain"][0]["frames"] == [
        {"file": "main.py", "line": 2},
        {"file": EXTERNAL_LABEL, "line": None},
    ]


def test_no_raw_file_means_no_record(tmp_path, code_root):
    review = tmp_path / "review"
    bundle = snapshot_code_bundle(code_root)
    assert (
        write_frames_record(tmp_path / "absent.json", review, code_root, bundle) is None
    )
    assert not (review / FRAMES_FILENAME).exists()


def test_invalid_json_means_no_record(tmp_path, code_root):
    raw_path = tmp_path / "raw.json"
    raw_path.write_text("{not json")
    bundle = snapshot_code_bundle(code_root)
    assert (
        write_frames_record(raw_path, tmp_path / "review", code_root, bundle) is None
    )


# -- the wrapper that runs inside the job ---------------------------------------


def test_raw_record_holds_no_message():
    try:
        raise ValueError("patient 4171 is positive")
    except ValueError as exc:
        record = raw_record(exc)
    blob = json.dumps(record)
    assert "positive" not in blob
    assert record["chain"][0]["mro"][0] == "ValueError"


def test_raw_record_follows_a_chained_exception():
    try:
        try:
            raise KeyError("inner")
        except KeyError as inner:
            raise RuntimeError("outer") from inner
    except RuntimeError as exc:
        record = raw_record(exc)
    assert [c["mro"][0] for c in record["chain"]] == ["RuntimeError", "KeyError"]
    assert "inner" not in json.dumps(record)


def test_the_wrapper_records_a_crash_and_keeps_the_exit_code(tmp_path):
    entry = tmp_path / "main.py"
    entry.write_text("raise ValueError('secret value')\n")
    raw_path = tmp_path / "raw.json"

    proc = subprocess.run(
        [sys.executable, str(trace_runner_path()), str(entry)],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "SYFT_JOB_RAW_TRACE_PATH": str(raw_path)},
    )

    assert proc.returncode != 0
    assert "ValueError" in proc.stderr  # the real traceback still reaches stderr
    record = json.loads(raw_path.read_text())
    assert record["chain"][0]["mro"][0] == "ValueError"
    assert "secret value" not in raw_path.read_text()

    out = sanitize_raw_trace(record, tmp_path, snapshot_code_bundle(tmp_path))
    assert out["chain"][0]["frames"][-1] == {"file": "main.py", "line": 1}


def test_the_wrapper_leaves_a_clean_run_alone(tmp_path):
    entry = tmp_path / "main.py"
    entry.write_text("print('ok')\n")
    raw_path = tmp_path / "raw.json"
    proc = subprocess.run(
        [sys.executable, str(trace_runner_path()), str(entry)],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "SYFT_JOB_RAW_TRACE_PATH": str(raw_path)},
    )
    assert proc.returncode == 0
    assert "ok" in proc.stdout
    assert not raw_path.exists()


def test_a_deliberate_exit_writes_no_record(tmp_path):
    entry = tmp_path / "main.py"
    entry.write_text("import sys; sys.exit(3)\n")
    raw_path = tmp_path / "raw.json"
    proc = subprocess.run(
        [sys.executable, str(trace_runner_path()), str(entry)],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "SYFT_JOB_RAW_TRACE_PATH": str(raw_path)},
    )
    assert proc.returncode == 3
    assert not raw_path.exists()


def test_a_file_the_job_plants_after_approval_is_external(code_root):
    """run.sh builds a venv inside code/, so the job can write there."""
    bundle = snapshot_code_bundle(code_root)

    planted = code_root / "DO2_row_4171_diagnosis_POSITIVE.py"
    planted.write_text("\n" * 50)

    raw = chain(["ValueError"], [frame(planted.name, 42)])
    out = sanitize_raw_trace(raw, code_root, bundle)
    assert out["chain"][0]["frames"] == [{"file": EXTERNAL_LABEL, "line": None}]


def test_a_file_the_job_grows_keeps_its_approved_length(code_root):
    """A longer file must not widen the range of accepted line numbers."""
    bundle = snapshot_code_bundle(code_root)
    (code_root / "main.py").write_text("\n" * 5000)

    out = sanitize_raw_trace(chain(["ValueError"], [frame("main.py", 900)]), code_root, bundle)
    assert out["chain"][0]["frames"] == [{"file": EXTERNAL_LABEL, "line": None}]


def test_a_venv_file_is_external(code_root):
    bundle = snapshot_code_bundle(code_root)
    venv_file = code_root / ".venv" / "lib" / "evil.py"
    venv_file.parent.mkdir(parents=True)
    venv_file.write_text("x = 1\n")

    raw = chain(["ValueError"], [frame(".venv/lib/evil.py", 1)])
    assert sanitize_raw_trace(raw, code_root, bundle)["chain"][0]["frames"] == [
        {"file": EXTERNAL_LABEL, "line": None}
    ]
