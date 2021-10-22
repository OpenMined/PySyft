# third party
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
