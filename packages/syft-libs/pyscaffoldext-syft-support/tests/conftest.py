"""Place for fixtures and configuration that will be used in most of the tests.
A nice option is to put your ``autouse`` fixtures here.
Functions that can be imported and re-used are more suitable for the ``helpers`` file.
"""
import os
from pathlib import Path
from tempfile import mkdtemp

import pytest

from .helpers import rmpath


@pytest.fixture
def tmpfolder(tmp_path):
    old_path = os.getcwd()
    new_path = mkdtemp(dir=str(tmp_path))
    os.chdir(new_path)
    try:
        yield Path(new_path)
    finally:
        os.chdir(old_path)
        rmpath(new_path)
