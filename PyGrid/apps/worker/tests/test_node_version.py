# stdlib
import sys

sys.path.append("../")

# third party
from src import __version__


def test_version():
    assert __version__ == "0.5.0"
