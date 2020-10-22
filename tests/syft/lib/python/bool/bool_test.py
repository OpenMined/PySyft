# syft absolute
from syft.lib.python.bool import Bool
from syft.lib.python.dict import Dict
from syft.lib.python.float import Float
from syft.lib.python.int import Int
from syft.lib.python.list import List
from syft.lib.python.string import String
from syft.lib.python.tuple import Tuple

SyFalse = Bool(False)
SyTrue = Bool(True)


def test_repr() -> None:
    assert repr(SyFalse) == String("False")
    assert repr(SyTrue) == String("True")
    assert eval(repr(SyFalse)) == SyFalse
    assert eval(repr(SyTrue)) == SyTrue


def test_str() -> None:
    assert str(SyFalse) == String("False")
    assert str(SyTrue) == String("True")


def test_int() -> None:
    assert int(SyFalse) == Int(0)
    assert int(SyFalse) is not SyFalse
    assert int(SyTrue) == 1
    assert int(SyTrue) is not SyTrue


def test_float() -> None:
    assert float(SyFalse) == Float(0.0)
    assert float(SyFalse) == SyFalse
    assert float(SyTrue) == Float(1.0)
    assert float(SyTrue) == SyTrue


def test_math() -> None:
    # TODO this should work
    pass
    # assert +SyFalse == Int(0)
    # assert +SyFalse == SyFalse
    # assert -SyFalse == Int(0)
    # assert -SyFalse == SyFalse
    # assert abs(SyFalse) == Int(0)
    # assert abs(SyFalse) != SyFalse
    # assert +SyTrue == Int(1)
    # assert +SyTrue != SyTrue
    # assert -SyTrue == Int(-1)
    # assert abs(SyTrue) == Int(1)
    # assert abs(SyTrue) != SyTrue
    # assert ~SyFalse == Int(-1)
    # assert ~SyTrue == Int(-2)
    #
    # assert SyFalse + Int(2) == Int(2)
    # assert SyTrue + Int(2) == Int(3)
    # assert Int(2) + SyFalse == Int(2)
    # assert Int(2) + SyTrue == Int(3)
    #
    # assert SyFalse + SyFalse == Int(0)
    # assert SyFalse + SyFalse != SyFalse
    # assert SyFalse + SyTrue == Int(1)
    # assert SyFalse + SyTrue != SyTrue
    # assert SyTrue + SyFalse == Int(1)
    # assert SyTrue + SyFalse != SyTrue
    # assert SyTrue + SyTrue == Int(2)
    #
    # assert SyTrue - SyTrue == Int(0)
    # assert SyTrue - SyTrue != SyFalse
    # assert SyFalse - SyFalse == Int(0)
    # assert SyFalse - SyFalse != SyFalse
    # assert SyTrue - SyFalse == Int(1)
    # assert SyTrue - SyFalse != SyTrue
    # assert SyFalse - SyTrue == Int(-1)
    #
    # assert SyTrue * Int(1) == Int(1)
    # assert SyFalse * Int(1) == Int(0)
    # assert SyFalse * Int(1) != SyFalse
    #
    # assert SyTrue / Int(1) == Int(1)
    # assert SyTrue / Int(1) != SyTrue
    # assert SyFalse / Int(1) == Int(0)
    # assert SyFalse / Int(1) != SyFalse
    #
    # assert SyTrue % Int(1) == Int(0)
    # assert SyTrue % Int(1) != SyFalse
    # assert SyTrue % Int(2) == Int(1)
    # assert SyTrue % Int(2) != SyTrue
    # assert SyFalse % Int(1) == Int(0)
    # assert SyFalse % Int(1) != SyFalse
    #
    # for b in SyFalse, SyTrue:
    #     for i in Int(0), Int(1), Int(2):
    #         assert b ** i == Int(b) ** i
    #         assert b ** i is not Bool(Int(b) ** i)
    #
    # assert (Int(1) == Int(1)) == SyTrue
    # assert (Int(1) == Int(0)) == SyFalse
    # assert (Int(0) < Int(1)) == SyTrue
    # assert (Int(1) < Int(0)) == SyFalse
    # assert (Int(0) <= Int(0)) == SyTrue
    # assert (Int(1) <= Int(0)) == SyFalse
    # assert (Int(1) > Int(0)) == SyTrue
    # assert (Int(1) > Int(1)) == SyFalse
    # assert (Int(1) >= Int(1)) == SyTrue
    # assert (Int(0) >= Int(1)) == SyFalse
    # assert (Int(0) != Int(1)) == SyTrue
    # assert (Int(0) != Int(0)) == SyFalse
    #
    # x = List([Int(1)])
    # assert (x == x) == SyTrue
    # assert (x == x) != SyFalse
    #
    # assert (1 in x) == SyTrue
    # assert (0 in x) == SyFalse
    # assert (1 not in x) == SyFalse
    # assert (0 not in x) == SyTrue
    #
    # x = Dict({Int(1): Int(2)})
    # assert (x is x) == SyTrue
    # assert (x is not x) == SyFalse
    #
    # assert (Int(1) in x) == SyTrue
    # assert (Int(0) in x) == SyFalse
    # assert (Int(1) not in x) == SyFalse
    # assert (Int(0) not in x) == SyTrue
    #
    # assert not SyTrue == SyFalse
    # assert not SyFalse == SyTrue


def test_convert() -> None:
    assert Bool(10) == SyTrue
    assert Bool(1) == SyTrue
    assert Bool(-1) == SyTrue
    assert Bool(0) == SyFalse
    assert Bool("hello") == SyTrue
    assert Bool("") == SyFalse
    assert Bool() == SyFalse


def test_format() -> None:
    assert String("%d") % SyFalse == "0"
    assert String("%d") % SyTrue == "1"
    assert String("%x") % SyFalse == "0"
    assert String("%x") % SyTrue == "1"


def test_hasattr() -> None:
    assert hasattr([], "append") == SyTrue
    assert hasattr([], "wobble") == SyFalse


def test_callable() -> None:
    assert callable(len) == SyTrue
    assert callable(1) == SyFalse


def test_isinstance() -> None:
    assert isinstance(SyTrue, Bool) == SyTrue
    assert isinstance(SyFalse, Bool) == SyTrue
    assert isinstance(SyTrue, int) == SyTrue
    assert isinstance(SyFalse, int) == SyTrue
    assert isinstance(1, Bool) == SyFalse
    assert isinstance(0, Bool) == SyFalse


def test_issubclass() -> None:
    assert issubclass(Bool, int) == SyTrue
    assert issubclass(int, Bool) == SyFalse


def test_contains() -> None:
    assert (Int(1) in {}) == SyFalse
    assert (Int(1) in {Int(1): Int(1)}) == SyTrue


def test_string():
    assert String("xyz").endswith("z") == SyTrue
    assert String("xyz").endswith("x") == SyFalse
    assert String("xyz0123").isalnum() == SyTrue
    assert String("@#$%").isalnum() == SyFalse
    assert String("xyz").isalpha() == SyTrue
    assert String("@#$%").isalpha() == SyFalse
    assert String("0123").isdigit() == SyTrue
    assert String("xyz").isdigit() == SyFalse
    assert String("xyz").islower() == SyTrue
    assert String("XYZ").islower() == SyFalse
    assert String("0123").isdecimal() == SyTrue
    assert String("xyz").isdecimal() == SyFalse
    assert String("0123").isnumeric() == SyTrue
    assert String("xyz").isnumeric() == SyFalse
    assert String(" ").isspace() == SyTrue
    assert String("\xa0").isspace() == SyTrue
    assert String("\u3000").isspace() == SyTrue
    assert String("XYZ").isspace() == SyFalse
    assert String("X").istitle() == SyTrue
    assert String("x").istitle() == SyFalse
    assert String("XYZ").isupper() == SyTrue
    assert String("xyz").isupper() == SyFalse
    assert String("xyz").startswith("x") == SyTrue
    assert String("xyz").startswith("z") == SyFalse


def test_boolean():
    assert SyTrue & Int(1) == Int(1)
    assert not isinstance(SyTrue & Int(1), Bool)
    assert SyTrue & SyTrue == SyTrue

    assert SyTrue | Int(1) == Int(1)
    assert not isinstance(SyTrue | Int(1), Bool)
    assert SyTrue | SyTrue == SyTrue

    assert SyTrue ^ Int(1) == Int(0)
    assert not isinstance(SyTrue ^ Int(1), Bool)
    assert SyTrue ^ SyTrue == SyFalse


def test_types():
    # types are always SyTrue.
    for t in [
        bool,
        complex,
        dict,
        float,
        int,
        list,
        object,
        set,
        str,
        tuple,
        type,
        Int,
        Float,
        Dict,
        List,
        String,
        Tuple,
        Float,
    ]:
        assert bool(t) == SyTrue


def test_operator():
    # stdlib
    import operator

    assert operator.truth(0) == SyFalse
    assert operator.truth(1) == SyTrue
    assert operator.not_(1) == SyFalse
    assert operator.not_(0) == SyTrue
    assert operator.contains([], 1) == SyFalse
    assert operator.contains([1], 1) == SyTrue
    assert operator.lt(0, 0) == SyFalse
    assert operator.lt(0, 1) == SyTrue
    assert operator.is_(SyTrue, SyTrue) == SyTrue
    assert operator.is_(SyTrue, SyFalse) == SyFalse
    assert operator.is_not(SyTrue, SyTrue) == SyFalse
    assert operator.is_not(SyTrue, SyFalse) == SyTrue


def test_from_bytes():
    assert Bool.from_bytes(b"\x00" * 8, String("big")) == SyFalse
    assert Bool.from_bytes(b"abcd", String("little")) == SyTrue


def test_sane_len():
    # this test just tests our assumptions about __len__
    # this will start failing if __len__ changes assertions
    for badval in [String("illegal"), Int(-1), Int(1 << 32)]:

        class A:
            def __len__(self):
                return badval

        try:
            Bool(A())
        except (Exception) as e_bool:
            try:
                len(A())
            except (Exception) as e_len:
                assert str(e_bool) == str(e_len)


def test_real_and_imag():
    assert SyTrue.real == Int(1)
    assert SyTrue.imag == Int(0)
    assert type(SyTrue.real) is Int
    assert type(SyTrue.imag) is Int
    assert SyFalse.real == Int(0)
    assert SyFalse.imag == Int(0)
    assert type(SyFalse.real) is Int
    assert type(SyFalse.imag) is Int
