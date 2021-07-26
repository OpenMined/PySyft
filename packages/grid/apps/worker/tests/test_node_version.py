# stdlib
import imp
import sys

sys.path.append("../")

# third party
from src import __version__

VERSION = imp.load_source("VERSION", "../../VERSION")


def test_version():
    assert __version__ == VERSION.__version__
