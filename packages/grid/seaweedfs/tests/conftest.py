# stdlib
from pathlib import Path
from secrets import token_hex
import shutil
from tempfile import gettempdir

# third party
import pytest

__all__ = ["random_path"]


@pytest.fixture
def random_path() -> Path:  # type: ignore
    path = Path(gettempdir(), f"{token_hex(8)}")
    yield path

    if path.exists() and path.is_file():
        path.unlink()
    elif path.exists() and path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
