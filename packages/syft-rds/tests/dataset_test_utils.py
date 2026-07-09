"""Shared helpers for the syft-rds product tests.

Uniquely named (not ``utils.py``) to avoid a pytest module-name collision with
``tests/unit/utils.py`` when the full monorepo suite is collected together.
"""

import random
from pathlib import Path


def create_tmp_dataset_files():
    tmp_dir = Path("/tmp/syft-datasets-testing") / str(random.randint(1, 1000000))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mock_path = tmp_dir / "mock.txt"
    private_path = tmp_dir / "private.txt"
    readme_path = tmp_dir / "readme.md"
    mock_path.write_text("Hello, world!")
    private_path.write_text("Hello, world private!")
    readme_path.write_text("Hello, world!")
    return mock_path, private_path, readme_path
