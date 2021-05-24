# stdlib
import logging

# third party
import _pytest
import pytest

# syft absolute
import syft as sy


@pytest.mark.slow
def test_load_lib() -> None:
    assert sy.lib.load(lib="tenseal") is None
    assert sy.lib.load("tenseal", "opacus") is None
    assert sy.lib.load(["tenseal", "opacus"]) is None
    assert sy.lib.load(("tenseal", "opacus")) is None
    assert sy.lib.load({"tenseal"}) is None


def test_load_errors(caplog: _pytest.logging.LogCaptureFixture) -> None:
    # Error if a non-supported library is loaded
    with caplog.at_level(logging.CRITICAL, logger="syft.logger"):
        sy.lib.load("non_compatible")
    assert "Unable to load package support for: non_compatible." in caplog.text
    caplog.clear()

    # Error if non-string object type is attempted to be loaded
    with caplog.at_level(logging.CRITICAL, logger="syft.logger"):
        sy.lib.load([True])
    assert "Unable to load package support for: True." in caplog.text
    caplog.clear()

    # Error if a non-iterable object is passed
    with caplog.at_level(logging.CRITICAL, logger="syft.logger"):
        sy.lib.load(True)
    assert "Unable to load package support for any library." in caplog.text
    caplog.clear()
