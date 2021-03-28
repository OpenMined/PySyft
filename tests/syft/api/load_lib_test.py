# syft absolute
import syft as sy


def test_load_lib() -> None:
    assert syft.lib.load(["tenseal", "opacus"]) is None
    assert syft.lib.load("tenseal") is None
    assert syft.lib.load("tenseal", "opacus") is None
    assert syft.lib.load(["tenseal", "opacus"]) is None
