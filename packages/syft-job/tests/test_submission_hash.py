"""The submission hash must not depend on how the OS sorts paths.

WindowsPath sorts case-insensitively and PosixPath case-sensitively, so a DS on
Windows and a DO on Linux would feed files in a different order.
"""

import hashlib
from pathlib import Path

from syft_job.submission import code_hash


def test_code_hash_orders_files_by_posix_path(tmp_path: Path):
    files = {
        "run.sh": b"python code/a.py\n",
        "code/B.py": b"b\n",
        "code/a.py": b"a\n",
    }
    for name, content in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(content)

    expected = hashlib.sha256()
    for name in sorted(files):  # "code/B.py" < "code/a.py" < "run.sh"
        expected.update(name.encode())
        expected.update(b"\0")
        expected.update(hashlib.sha256(files[name]).digest())

    assert code_hash(tmp_path) == expected.hexdigest()
