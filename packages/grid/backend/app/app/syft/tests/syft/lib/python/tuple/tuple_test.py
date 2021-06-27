# syft absolute
from syft.lib.python.tuple import Tuple

type2test = Tuple


def test_constructors():
    assert Tuple() == ()
    assert Tuple([]) == ()
    assert Tuple([0, 1, 2, 3]) == (0, 1, 2, 3)
    assert Tuple("") == ()
    assert Tuple("spam") == ("s", "p", "a", "m")
    assert Tuple(x for x in range(10) if x % 2) == (1, 3, 5, 7, 9)


def test_truth():
    assert not Tuple()
    assert Tuple((42,))


def test_len():
    assert len(()) == 0
    assert len((0,)) == 1
    assert len((0, 1, 2)) == 3


def test_iadd():
    u = (0, 1)
    u2 = u
    u += (2, 3)
    assert u is not u2


def test_imul():
    u = (0, 1)
    u2 = u
    u *= 3
    assert u is not u2


def test_repr():
    l0 = Tuple()
    l2 = (0, 1, 2)
    a0 = Tuple(l0)
    a2 = Tuple(l2)

    assert str(a0) == repr(l0)
    assert str(a2) == repr(l2)
    assert repr(a0) == "()"
    assert repr(a2), "(0, 1 ==  2)"


def test_hash():
    assert hash(Tuple()) == hash(())
    assert hash(Tuple((1,))) == hash((1,))
    assert hash(Tuple((-1, 0, 1))) == hash((-1, 0, 1))
