# syft absolute
from syft.core.common.group import VERIFYALL
from syft.core.common.group import VerifyAll


def test_verify_all() -> None:
    a = VERIFYALL
    b = type(VERIFYALL)()
    c = VerifyAll
    assert a == b and b == c()
    assert type(a) == type(b) and type(b) == c
