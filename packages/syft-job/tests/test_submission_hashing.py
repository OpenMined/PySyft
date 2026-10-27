# -*- coding: utf-8 -*-
"""Regression tests for syft-job issue #9551:

- code_hash(as_sent=True) normalizes CRLF and LF identically.
- request_hash(as_sent=True) normalizes CRLF and LF identically.
- _digest uses cross-platform deterministic sorting by relative posix paths.
- _as_sent correctly reads UTF-8 files with non-ASCII characters.
- _job_has_all_files and _request_has_all_files accept CRLF files when declared with as_sent=True.
"""

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from syft_job.client import JobClient
from syft_job.models import JobSubmissionMetadata, RequestMetadata
from syft_job.submission import (
    SUBMISSION_HASH_HEADER,
    _as_sent,
    code_hash,
    request_hash,
    submission_hash,
)


def test_code_hash_crlf_lf_equivalence(tmp_path: Path):
    crlf_dir = tmp_path / "crlf"
    lf_dir = tmp_path / "lf"
    for d in (crlf_dir, lf_dir):
        (d / "code").mkdir(parents=True)

    (crlf_dir / "run.sh").write_bytes(b"#!/bin/sh\r\npython code/main.py\r\n")
    (crlf_dir / "code" / "main.py").write_bytes(b"print('hello')\r\n")

    (lf_dir / "run.sh").write_bytes(b"#!/bin/sh\npython code/main.py\n")
    (lf_dir / "code" / "main.py").write_bytes(b"print('hello')\n")

    # Both must produce the exact same declared hash
    assert code_hash(crlf_dir, as_sent=True) == code_hash(lf_dir, as_sent=True)
    # The declared hash must equal the LF raw hash
    assert code_hash(crlf_dir, as_sent=True) == code_hash(lf_dir, as_sent=False)


def test_request_hash_crlf_lf_equivalence(tmp_path: Path):
    crlf_dir = tmp_path / "crlf"
    lf_dir = tmp_path / "lf"
    crlf_dir.mkdir()
    lf_dir.mkdir()

    (crlf_dir / "request.yaml").write_bytes(b"api: test\r\n")
    (crlf_dir / "params.json").write_bytes(b'{"key": "value"}\r\n')

    (lf_dir / "request.yaml").write_bytes(b"api: test\n")
    (lf_dir / "params.json").write_bytes(b'{"key": "value"}\n')

    assert request_hash(crlf_dir, as_sent=True) == request_hash(lf_dir, as_sent=True)
    assert request_hash(crlf_dir, as_sent=True) == request_hash(lf_dir, as_sent=False)


def test_digest_deterministic_file_ordering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sub_dir = tmp_path / "sub"
    (sub_dir / "code").mkdir(parents=True)
    (sub_dir / "run.sh").write_bytes(b"echo 1\n")
    # Mixed-case filenames: 'B.py' (ASCII 66) vs 'a.py' (ASCII 97)
    (sub_dir / "code" / "B.py").write_bytes(b"b\n")
    (sub_dir / "code" / "a.py").write_bytes(b"a\n")

    orig_rglob = Path.rglob

    # Traversal order 1: ascending
    monkeypatch.setattr(
        Path,
        "rglob",
        lambda self, *args, **kwargs: sorted(orig_rglob(self, *args, **kwargs)),
    )
    hash_forward = submission_hash(sub_dir)

    # Traversal order 2: descending / reversed
    monkeypatch.setattr(
        Path,
        "rglob",
        lambda self, *args, **kwargs: sorted(
            orig_rglob(self, *args, **kwargs), reverse=True
        ),
    )
    hash_reversed = submission_hash(sub_dir)

    # Verify digest is identical regardless of filesystem discovery order
    assert hash_forward == hash_reversed

    # In canonical POSIX code-point order:
    # "code/B.py" < "code/a.py" < "run.sh"
    expected = hashlib.sha256()
    for rel_posix, content in [
        ("code/B.py", b"b\n"),
        ("code/a.py", b"a\n"),
        ("run.sh", b"echo 1\n"),
    ]:
        expected.update(rel_posix.encode())
        expected.update(b"\0")
        expected.update(hashlib.sha256(content).digest())

    assert hash_forward == expected.hexdigest()


def test_as_sent_utf8_non_ascii(tmp_path: Path):
    py_file = tmp_path / "unicode.py"
    greeting = "Héllo wörld"
    pi = "π"
    content = f'greeting = "{greeting}"\r\npi = "{pi}"\r\n'.encode("utf-8")
    py_file.write_bytes(content)

    # Must normalize newlines and preserve UTF-8 encoding
    expected = f'greeting = "{greeting}"\npi = "{pi}"\n'.encode("utf-8")
    assert _as_sent(py_file) == expected


def test_as_sent_binary_fallback(tmp_path: Path):
    bin_file = tmp_path / "raw.bin"
    # Invalid UTF-8 bytes containing CRLF
    raw_bytes = b"\xff\xfe\x00\x01\r\n\x80\x81"
    bin_file.write_bytes(raw_bytes)

    # Must fall back to raw read_bytes() without raising UnicodeDecodeError
    assert _as_sent(bin_file) == raw_bytes


def test_job_has_all_files_accepts_crlf(tmp_path: Path):
    job_dir = tmp_path / "test.job"
    (job_dir / "code").mkdir(parents=True)
    (job_dir / "run.sh").write_bytes(b"#!/bin/sh\r\npython code/main.py\r\n")
    (job_dir / "code" / "main.py").write_bytes(b"print('hello')\r\n")

    declared_hash = code_hash(job_dir, as_sent=True)

    client = MagicMock()
    client.manager.submission_dir.return_value = job_dir
    client.manager.read_submission.return_value = JobSubmissionMetadata(
        name="test.job",
        submitted_by="ds@test.org",
        datasite_email="do@test.org",
        submitted_at=datetime.now(timezone.utc),
        headers={SUBMISSION_HASH_HEADER: declared_hash},
    )

    # Verify that CRLF files on disk match the declared hash
    assert JobClient._job_has_all_files(client, MagicMock()) is True

    # Verify that tampered content is rejected
    (job_dir / "code" / "main.py").write_bytes(b"print('tampered')\r\n")
    assert JobClient._job_has_all_files(client, MagicMock()) is False


def test_request_has_all_files_accepts_crlf(tmp_path: Path):
    req_dir = tmp_path / "call.job"
    req_dir.mkdir()
    (req_dir / "request.yaml").write_bytes(b"api: test\r\n")
    (req_dir / "params.json").write_bytes(b'{"key": "value"}\r\n')

    declared_files_hash = request_hash(req_dir, as_sent=True)

    client = MagicMock()
    client.manager.request_dir.return_value = req_dir
    client.manager.read_request.return_value = RequestMetadata(
        api="test",
        submitted_at=datetime.now(timezone.utc),
        files=["params.json"],
        files_hash=declared_files_hash,
    )

    # Verify that CRLF files on disk match the declared files_hash
    assert JobClient._request_has_all_files(client, MagicMock()) is True

    # Verify that tampered content is rejected
    (req_dir / "params.json").write_bytes(b'{"key": "tampered"}\r\n')
    assert JobClient._request_has_all_files(client, MagicMock()) is False


def test_code_hash_mixed_newlines_and_empty_files(tmp_path: Path):
    """Verify mixed CRLF/LF newlines, empty files, and nested paths hash identically."""
    mix_dir = tmp_path / "mix"
    lf_dir = tmp_path / "lf"
    for d in (mix_dir, lf_dir):
        (d / "code" / "sub").mkdir(parents=True)
        (d / "code" / "empty.py").write_bytes(b"")

    (mix_dir / "run.sh").write_bytes(b"#!/bin/sh\r\necho 1\necho 2\r\n")
    (mix_dir / "code" / "sub" / "mod.py").write_bytes(b"x = 1\r\ny = 2\n")

    (lf_dir / "run.sh").write_bytes(b"#!/bin/sh\necho 1\necho 2\n")
    (lf_dir / "code" / "sub" / "mod.py").write_bytes(b"x = 1\ny = 2\n")

    assert code_hash(mix_dir, as_sent=True) == code_hash(lf_dir, as_sent=True)
    assert code_hash(mix_dir, as_sent=True) == code_hash(lf_dir, as_sent=False)


def test_job_has_all_files_rejects_missing_or_extra_files(tmp_path: Path):
    """Verify that missing files or extra unexpected files reject reception."""
    job_dir = tmp_path / "missing_extra.job"
    (job_dir / "code").mkdir(parents=True)
    (job_dir / "run.sh").write_bytes(b"#!/bin/sh\npython code/main.py\n")
    (job_dir / "code" / "main.py").write_bytes(b"print('hello')\n")

    declared_hash = code_hash(job_dir, as_sent=True)

    client = MagicMock()
    client.manager.submission_dir.return_value = job_dir
    client.manager.read_submission.return_value = JobSubmissionMetadata(
        name="missing_extra.job",
        submitted_by="ds@test.org",
        datasite_email="do@test.org",
        submitted_at=datetime.now(timezone.utc),
        headers={SUBMISSION_HASH_HEADER: declared_hash},
    )

    # Sanity check: valid state accepted
    assert JobClient._job_has_all_files(client, MagicMock()) is True

    # Missing file rejected
    (job_dir / "code" / "main.py").unlink()
    assert JobClient._job_has_all_files(client, MagicMock()) is False

    # Restore main.py, add extra unexpected file
    (job_dir / "code" / "main.py").write_bytes(b"print('hello')\n")
    (job_dir / "code" / "extra.py").write_bytes(b"print('extra')\n")
    assert JobClient._job_has_all_files(client, MagicMock()) is False


def test_submission_hash_preserves_raw_disk_integrity(tmp_path: Path):
    """submission_hash must NOT normalize line endings; it pins raw on-disk state."""
    crlf_sub = tmp_path / "crlf_sub"
    lf_sub = tmp_path / "lf_sub"
    for d in (crlf_sub, lf_sub):
        (d / "code").mkdir(parents=True)
        (d / "config.yaml").write_bytes(b"name: job\n")

    (crlf_sub / "run.sh").write_bytes(b"#!/bin/sh\r\n")
    (crlf_sub / "code" / "main.py").write_bytes(b"print(1)\r\n")

    (lf_sub / "run.sh").write_bytes(b"#!/bin/sh\n")
    (lf_sub / "code" / "main.py").write_bytes(b"print(1)\n")

    # Raw on-disk bytes differ, so submission_hash must differ
    assert submission_hash(crlf_sub) != submission_hash(lf_sub)
