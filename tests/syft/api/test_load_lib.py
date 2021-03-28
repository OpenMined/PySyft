# third party
import pytest

# syft absolute
import syft as sy
from syft.lib import load


def test_load_lib() -> None:
    assert load(["tenseal", "opacus"]) is None
    assert load("tenseal") is None
    assert load("tenseal", "opacus") is None
    assert load(["tenseal", "opacus"]) is None

