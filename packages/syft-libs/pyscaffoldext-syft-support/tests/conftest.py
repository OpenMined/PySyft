"""Place for fixtures and configuration that will be used in most of the tests.
A nice option is to put your ``autouse`` fixtures here.
Functions that can be imported and re-used are more suitable for the ``helpers`` file.
"""
# stdlib
import os

# third party
import pytest

# relative
from .test_helpers import find_package_bin, rmpath, uniqstr


@pytest.fixture
def tmpfolder(tmp_path):
    old_path = os.getcwd()
    new_path = tmp_path / uniqstr()
    new_path.mkdir(parents=True, exist_ok=True)
    os.chdir(str(new_path))
    try:
        yield new_path
    finally:
        os.chdir(old_path)
        rmpath(new_path)


@pytest.fixture(autouse=True)
def cwd(tmpdir):
    """Guarantee a blank folder as workspace"""
    with tmpdir.as_cwd():
        yield tmpdir


@pytest.fixture
def putup():
    return find_package_bin("pyscaffold.cli", "putup")
