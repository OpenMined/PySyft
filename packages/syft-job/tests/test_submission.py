"""Hashes must not depend on the OS or checkout that produces them.

A submitter on Windows may declare code_hash() over a CRLF checkout while the
data owner receives an LF copy, and Path feeds separators native to the OS, so
both digests fold to one form before hashing.
"""

import hashlib
from pathlib import Path

from syft_job.submission import code_hash, submission_hash


def _write_files(folder: Path, files: dict[str, bytes]) -> None:
    for name, content in files.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(content)


def test_code_hash_matches_between_crlf_and_lf_copies(tmp_path: Path):
    lf_dir = tmp_path / "lf"
    crlf_dir = tmp_path / "crlf"
    _write_files(
        lf_dir, {"run.sh": b"python code/main.py\n", "code/main.py": b"print('hi')\n"}
    )
    _write_files(
        crlf_dir,
        {"run.sh": b"python code/main.py\r\n", "code/main.py": b"print('hi')\r\n"},
    )

    assert code_hash(crlf_dir) == code_hash(lf_dir)


def test_code_hash_folds_crlf_to_lf_bytes(tmp_path: Path):
    files = {
        "run.sh": b"#!/bin/sh\r\npython code/main.py\r\n",
        "code/main.py": b"print('hi')\r\n",
    }
    _write_files(tmp_path, files)

    expected = hashlib.sha256()
    for name in sorted(files):  # "code/main.py" < "run.sh"
        expected.update(name.encode("utf-8"))
        expected.update(b"\0")
        expected.update(hashlib.sha256(files[name].replace(b"\r\n", b"\n")).digest())

    assert code_hash(tmp_path) == expected.hexdigest()


def test_submission_hash_feeds_posix_paths_in_sorted_order(tmp_path: Path):
    files = {
        "run.sh": b"python code/a.py\n",
        "code/B.py": b"b\n",
        "code/a.py": b"a\n",
        "config.yaml": b"name: job\n",
    }
    _write_files(tmp_path, files)

    # Forward slashes, codepoint sort: "code/B.py" < "code/a.py" on every OS,
    # where WindowsPath itself would sort a.py first.
    expected = hashlib.sha256()
    for name in sorted(files):  # "code/B.py" < "code/a.py" < "config.yaml" < "run.sh"
        expected.update(name.encode("utf-8"))
        expected.update(b"\0")
        expected.update(hashlib.sha256(files[name]).digest())

    assert submission_hash(tmp_path) == expected.hexdigest()
