# syft absolute
import syft as sy


def test_load_lib() -> None:
    assert sy.lib.load(["tenseal", "opacus"]) is None
    assert sy.lib.load("tenseal") is None
    assert sy.lib.load("tenseal", "opacus") is None
    assert sy.lib.load(["tenseal", "opacus"]) is None
