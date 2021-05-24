# stdlib
import sys
from typing import Any

# third party
import pytest

# syft absolute
from syft.lib.python.int import Int

VALID_UNDERSCORE_LITERALS = [
    "0_0_0",
    "4_2",
    "1_0000_0000",
    "0b1001_0100",
    "0xffff_ffff",
    "0o5_7_7",
    "1_00_00.5",
    "1_00_00.5e5",
    "1_00_00e5_1",
    "1e1_0",
    ".1_4",
    ".1_4e1",
    "0b_0",
    "0x_f",
    "0o_5",
    "1_00_00j",
    "1_00_00.5j",
    "1_00_00e5_1j",
    ".1_4j",
    "(1_2.5+3_3j)",
    "(.5_6j)",
]

INVALID_UNDERSCORE_LITERALS = [
    # Trailing underscores:
    "0_",
    "42_",
    "1.4j_",
    "0x_",
    "0b1_",
    "0xf_",
    "0o5_",
    "0 if 1_Else 1",
    # Underscores in the base selector:
    "0_b0",
    "0_xf",
    "0_o5",
    # Old-style octal, still disallowed:
    "0_7",
    "09_99",
    # Multiple consecutive underscores:
    "4_______2",
    "0.1__4",
    "0.1__4j",
    "0b1001__0100",
    "0xffff__ffff",
    "0x___",
    "0o5__77",
    "1e1__0",
    "1e1__0j",
    # Underscore right before a dot:
    "1_.4",
    "1_.4j",
    # Underscore right after a dot:
    "1._4",
    "1._4j",
    "._5",
    "._5j",
    # Underscore right after a sign:
    "1.0e+_1",
    "1.0e+_1j",
    # Underscore right before j:
    "1.4_j",
    "1.4e5_j",
    # Underscore right before e:
    "1_e1",
    "1.4_e1",
    "1.4_e1j",
    # Underscore right after e:
    "1e_1",
    "1.4e_1",
    "1.4e_1j",
    # Complex cases with parens:
    "(1+1.5_j_)",
    "(1+1.5_j)",
]

L = [
    ("0", 0),
    ("1", 1),
    ("9", 9),
    ("10", 10),
    ("99", 99),
    ("100", 100),
    ("314", 314),
    (" 314", 314),
    ("314 ", 314),
    ("  \t\t  314  \t\t  ", 314),
    (repr(sys.maxsize), sys.maxsize),
    ("  1x", ValueError),
    ("  1  ", 1),
    ("  1\02  ", ValueError),
    ("", ValueError),
    (" ", ValueError),
    ("  \t\t  ", ValueError),
    ("\u0200", ValueError),
]


class IntSubclass(Int):
    pass


def test_basic() -> None:
    assert Int(314) == 314
    assert Int(3.14) == 3
    # Check that conversion from float truncates towards zero
    assert Int(-3.14) == -3
    assert Int(3.9) == 3
    assert Int(-3.9) == -3
    assert Int(3.5) == 3
    assert Int(-3.5) == -3
    assert Int("-3") == -3
    assert Int(" -3 ") == -3
    assert Int("\N{EM SPACE}-3\N{EN SPACE}") == -3
    # Different base:
    assert Int("10", 16) == 16
    # Test conversion from strings and various anomalies
    for s, v in L:
        for sign in "", "+", "-":
            for prefix in "", " ", "\t", "  \t\t  ":
                ss = prefix + sign + s
                vv = v
                if sign == "-" and v is not ValueError:
                    vv = -v
                try:
                    assert Int(ss) == vv
                except ValueError:
                    pass

    s = repr(-1 - sys.maxsize)
    x = Int(s)
    assert Int(x + 1) == -sys.maxsize
    assert isinstance(x, Int)
    # should return Int
    assert Int(s[1:]) == sys.maxsize + 1

    # should return Int
    x = Int(1e100)
    assert isinstance(x, Int)
    x = Int(-1e100)
    assert isinstance(x, Int)

    # SF bug 434186:  0x80000000/2 != 0x80000000>>1.
    # Worked by accident in Windows release build, but failed in debug build.
    # Failed in all Linux builds.
    x = -1 - sys.maxsize
    assert Int(x >> 1) == x // 2

    x = Int("1" * 600)
    assert isinstance(x, Int)

    # TODO: check why is this not working
    # with pytest.raises(TypeError):
    #     Int(1, 12)

    assert Int("0o123", 0) == 83
    assert Int("0x123", 16) == 291

    # Bug 1679: "0x" is not a valid hex literal
    with pytest.raises(ValueError):
        Int("0x", 16)
    with pytest.raises(ValueError):
        Int("0x", 0)

    with pytest.raises(ValueError):
        Int("0o", 8)
    with pytest.raises(ValueError):
        Int("0o", 0)

    with pytest.raises(ValueError):
        Int("0b", 2)
    with pytest.raises(ValueError):
        Int("0b", 0)

    # SF bug 1334662: Int(string, base) wrong answers
    # Various representations of 2**32 evaluated to 0
    # rather than 2**32 in previous versions

    assert Int("100000000000000000000000000000000", 2) == 4294967296
    assert Int("102002022201221111211", 3) == 4294967296
    assert Int("10000000000000000", 4) == 4294967296
    assert Int("32244002423141", 5) == 4294967296
    assert Int("1550104015504", 6) == 4294967296
    assert Int("211301422354", 7) == 4294967296
    assert Int("40000000000", 8) == 4294967296
    assert Int("12068657454", 9) == 4294967296
    assert Int("4294967296", 10) == 4294967296
    assert Int("1904440554", 11) == 4294967296
    assert Int("9ba461594", 12) == 4294967296
    assert Int("535a79889", 13) == 4294967296
    assert Int("2ca5b7464", 14) == 4294967296
    assert Int("1a20dcd81", 15) == 4294967296
    assert Int("100000000", 16) == 4294967296
    assert Int("a7ffda91", 17) == 4294967296
    assert Int("704he7g4", 18) == 4294967296
    assert Int("4f5aff66", 19) == 4294967296
    assert Int("3723ai4g", 20) == 4294967296
    assert Int("281d55i4", 21) == 4294967296
    assert Int("1fj8b184", 22) == 4294967296
    assert Int("1606k7ic", 23) == 4294967296
    assert Int("mb994ag", 24) == 4294967296
    assert Int("hek2mgl", 25) == 4294967296
    assert Int("dnchbnm", 26) == 4294967296
    assert Int("b28jpdm", 27) == 4294967296
    assert Int("8pfgih4", 28) == 4294967296
    assert Int("76beigg", 29) == 4294967296
    assert Int("5qmcpqg", 30) == 4294967296
    assert Int("4q0jto4", 31) == 4294967296
    assert Int("4000000", 32) == 4294967296
    assert Int("3aokq94", 33) == 4294967296
    assert Int("2qhxjli", 34) == 4294967296
    assert Int("2br45qb", 35) == 4294967296
    assert Int("1z141z4", 36) == 4294967296

    # tests with base 0
    # this fails on 3.0, but in 2.x the old octal syntax is allowed
    assert Int(" 0o123  ", 0) == 83
    assert Int(" 0o123  ", 0) == 83
    assert Int("000", 0) == 0
    assert Int("0o123", 0) == 83
    assert Int("0x123", 0) == 291
    assert Int("0b100", 0) == 4
    assert Int(" 0O123   ", 0) == 83
    assert Int(" 0X123  ", 0) == 291
    assert Int(" 0B100 ", 0) == 4

    # without base still base 10
    assert Int("0123") == 123
    assert Int("0123", 10) == 123

    # tests with prefix and base != 0
    assert Int("0x123", 16) == 291
    assert Int("0o123", 8) == 83
    assert Int("0b100", 2) == 4
    assert Int("0X123", 16) == 291
    assert Int("0O123", 8) == 83
    assert Int("0B100", 2) == 4

    # the code has special checks for the first character after the
    #  type prefix
    with pytest.raises(ValueError):
        Int("0b2", 2)
    with pytest.raises(ValueError):
        Int("0b02", 2)
    with pytest.raises(ValueError):
        Int("0B2", 2)
    with pytest.raises(ValueError):
        Int("0B02", 2)
    with pytest.raises(ValueError):
        Int("0o8", 8)
    with pytest.raises(ValueError):
        Int("0o08", 8)
    with pytest.raises(ValueError):
        Int("0O8", 8)
    with pytest.raises(ValueError):
        Int("0O08", 8)
    with pytest.raises(ValueError):
        Int("0xg", 16)
    with pytest.raises(ValueError):
        Int("0x0g", 16)
    with pytest.raises(ValueError):
        Int("0Xg", 16)
    with pytest.raises(ValueError):
        Int("0X0g", 16)

    # SF bug 1334662: Int(string, base) wrong answers
    # Checks for proper evaluation of 2**32 + 1
    assert Int("100000000000000000000000000000001", 2) == 4294967297
    assert Int("102002022201221111212", 3) == 4294967297
    assert Int("10000000000000001", 4) == 4294967297
    assert Int("32244002423142", 5) == 4294967297
    assert Int("1550104015505", 6) == 4294967297
    assert Int("211301422355", 7) == 4294967297
    assert Int("40000000001", 8) == 4294967297
    assert Int("12068657455", 9) == 4294967297
    assert Int("4294967297", 10) == 4294967297
    assert Int("1904440555", 11) == 4294967297
    assert Int("9ba461595", 12) == 4294967297
    assert Int("535a7988a", 13) == 4294967297
    assert Int("2ca5b7465", 14) == 4294967297
    assert Int("1a20dcd82", 15) == 4294967297
    assert Int("100000001", 16) == 4294967297
    assert Int("a7ffda92", 17) == 4294967297
    assert Int("704he7g5", 18) == 4294967297
    assert Int("4f5aff67", 19) == 4294967297
    assert Int("3723ai4h", 20) == 4294967297
    assert Int("281d55i5", 21) == 4294967297
    assert Int("1fj8b185", 22) == 4294967297
    assert Int("1606k7id", 23) == 4294967297
    assert Int("mb994ah", 24) == 4294967297
    assert Int("hek2mgm", 25) == 4294967297
    assert Int("dnchbnn", 26) == 4294967297
    assert Int("b28jpdn", 27) == 4294967297
    assert Int("8pfgih5", 28) == 4294967297
    assert Int("76beigh", 29) == 4294967297
    assert Int("5qmcpqh", 30) == 4294967297
    assert Int("4q0jto5", 31) == 4294967297
    assert Int("4000001", 32) == 4294967297
    assert Int("3aokq95", 33) == 4294967297
    assert Int("2qhxjlj", 34) == 4294967297
    assert Int("2br45qc", 35) == 4294967297
    assert Int("1z141z5", 36) == 4294967297


@pytest.mark.xfail
def test_underscores() -> None:
    for lit in VALID_UNDERSCORE_LITERALS:
        if any(ch in lit for ch in ".eEjJ"):
            continue
        assert Int(lit, 0) == eval(lit)
        # TODO: check why is this not working
        assert Int(lit, 0), Int(lit.replace("_", "") == 0)
    for lit in INVALID_UNDERSCORE_LITERALS:
        if any(ch in lit for ch in ".eEjJ"):
            continue
        with pytest.raises(ValueError):
            Int(lit, 0)
    # Additional test cases with bases != 0, only for the constructor:
    assert Int("1_00", 3) == 9
    assert Int("0_100") == 100  # not valid as a literal!
    assert Int(b"1_00") == 100  # byte underscore
    with pytest.raises(ValueError):
        Int("_100")
    with pytest.raises(ValueError):
        Int("+_100")
    with pytest.raises(ValueError):
        Int("1__00")
    with pytest.raises(ValueError):
        Int("100_")


def test_small_Ints() -> None:
    # Bug #3236: Return small longs from PyLong_FromString
    # TODO: These are not going to work if we use is, brainstorm for a workaround
    assert Int("10") == 10
    assert Int("-1") == -1
    assert Int(b"10") == 10
    assert Int(b"-1") == -1


def test_no_args() -> None:
    assert Int() == 0


@pytest.mark.xfail
def test_keyword_args() -> None:
    # Test invoking Int() using keyword arguments.
    assert Int("100", base=2) == 4
    with pytest.raises(TypeError):
        Int(x=1.2)
    with pytest.raises(TypeError):
        Int(x="100", base=2)
    # TODO these should fail
    with pytest.raises(TypeError):
        Int(base=10)
    with pytest.raises(TypeError):
        Int(base=0)


def test_Int_base_limits() -> None:
    """Testing the supported limits of the Int() base parameter."""
    assert Int("0", 5) == 0
    with pytest.raises(ValueError):
        Int("0", 1)
    with pytest.raises(ValueError):
        Int("0", 37)
    with pytest.raises(ValueError):
        Int("0", -909)  # An old magic value base from Python 2.
    with pytest.raises(ValueError):
        Int("0", base=0 - (2 ** 234))
    with pytest.raises(ValueError):
        Int("0", base=2 ** 234)
    # Bases 2 through 36 are supported.
    for base in range(2, 37):
        assert Int("0", base=base) == 0


def test_Int_base_bad_types() -> None:
    """Not Integer types are not valid bases; issue16772."""
    with pytest.raises(TypeError):
        Int("0", 5.5)
    with pytest.raises(TypeError):
        Int("0", 5.0)


def test_Int_base_indexable() -> None:
    class MyIndexable(object):
        def __init__(self, value: Any):
            self.value = value

        def __index__(self):
            return self.value

    # Check out of range bases.
    for base in 2 ** 100, -(2 ** 100), 1, 37:
        with pytest.raises(ValueError):
            Int("43", base)

    # Check in-range bases.
    assert Int("101", base=MyIndexable(2)) == 5
    assert Int("101", base=MyIndexable(10)) == 101
    assert Int("101", base=MyIndexable(36)) == 1 + 36 ** 2


def test_Int_memoryview() -> None:
    assert Int(memoryview(b"123")[1:3]) == 23
    assert Int(memoryview(b"123\x00")[1:3]) == 23
    assert Int(memoryview(b"123 ")[1:3]) == 23
    assert Int(memoryview(b"123A")[1:3]) == 23
    assert Int(memoryview(b"1234")[1:3]) == 23


def test_string_float() -> None:
    with pytest.raises(ValueError):
        Int("1.2")


@pytest.mark.xfail
def test_Intconversion() -> None:
    # Test __Int__()
    class ClassicMissingMethods:
        pass

    with pytest.raises(TypeError):
        Int(ClassicMissingMethods())

    class MissingMethods(object):
        pass

    with pytest.raises(TypeError):
        Int(MissingMethods())

    class Foo0:
        def __Int__(self):
            return 42

    # TODO this should work
    assert Int(Foo0()) == 42

    class Classic:
        pass

    for base in (object, Classic):

        class IntOverridesTrunc(base):
            def __Int__(self):
                return 42

            def __trunc__(self):
                return -12

        # TODO this should work
        assert Int(IntOverridesTrunc()) == 42

        class JustTrunc(base):
            def __trunc__(self):
                return 42

        # TODO this should work
        assert Int(JustTrunc()) == 42

        class ExceptionalTrunc(base):
            def __trunc__(self):
                1 / 0

        with pytest.raises(ZeroDivisionError):
            Int(ExceptionalTrunc())

        for trunc_result_base in (object, Classic):

            class Index(trunc_result_base):
                def __index__(self):
                    return 42

            class TruncReturnsNonInt(base):
                def __trunc__(self):
                    return Index()

            # TODO this should work
            assert Int(TruncReturnsNonInt()) == 42

            class Intable(trunc_result_base):
                def __Int__(self):
                    return 42

            class TruncReturnsNonIndex(base):
                def __trunc__(self):
                    return Intable()

            # TODO this should work
            assert Int(TruncReturnsNonInt()) == 42

            class NonIntegral(trunc_result_base):
                def __trunc__(self):
                    # Check that we avoid infinite recursion.
                    return NonIntegral()

            class TruncReturnsNonIntegral(base):
                def __trunc__(self):
                    return NonIntegral()

            try:
                Int(TruncReturnsNonIntegral())
            except TypeError as e:
                assert str(e) == "__trunc__ returned non-Integral" " (type NonIntegral)"
            else:
                raise Exception("Failed to raise TypeError with %s")

            # Regression test for bugs.python.org/issue16060.
            class BadInt(trunc_result_base):
                def __Int__(self):
                    return 42.0

            class TruncReturnsBadInt(base):
                def __trunc__(self):
                    return BadInt()

            with pytest.raises(TypeError):
                Int(TruncReturnsBadInt())


def test_Int_subclass_with_index() -> None:
    class MyIndex(Int):
        def __index__(self):
            return 42

    class BadIndex(Int):
        def __index__(self):
            return 42.0

    my_Int = MyIndex(7)
    assert Int(my_Int) == 7
    assert Int(my_Int) == 7

    assert Int(BadIndex()) == 0


@pytest.mark.xfail
def test_Int_subclass_with_Int() -> None:
    class MyInt(Int):
        def __Int__(self):
            return 42

    class BadInt(Int):
        def __Int__(self):
            return 42.0

    # TODO this should work
    my_Int = MyInt(7)
    assert Int(my_Int) == 7
    assert Int(my_Int) == 42

    # TODO this should work
    my_Int = BadInt(7)
    assert Int(my_Int) == 7
    with pytest.raises(TypeError):
        Int(my_Int)


@pytest.mark.xfail
def test_Int_returns_Int_subclass() -> None:
    class BadIndex:
        def __index__(self):
            return True

    class BadIndex2(Int):
        def __index__(self):
            return True

    class BadInt:
        def __Int__(self):
            return True

    class BadInt2(Int):
        def __Int__(self):
            return True

    class TruncReturnsBadIndex:
        def __trunc__(self):
            return BadIndex()

    class TruncReturnsBadInt:
        def __trunc__(self):
            return BadInt()

    class TruncReturnsIntSubclass:
        def __trunc__(self):
            return True

    # TODO: this should work
    bad_Int = BadIndex()
    with pytest.raises(DeprecationWarning):
        n = Int(bad_Int)
    assert Int(n) == 1
    # this is not going to work
    assert type(n) is Int

    bad_Int = BadIndex2()
    n = Int(bad_Int)
    assert Int(n) == 0

    # TODO: this should work
    bad_Int = BadInt()
    with pytest.raises(DeprecationWarning):
        n = Int(bad_Int)
    assert Int(n) == 1
    # not going to work
    # self.assertIs(type(n), Int)

    # TODO: this should work
    bad_Int = BadInt2()
    with pytest.raises(DeprecationWarning):
        n = Int(bad_Int)
    assert Int(n) == 1
    # not going to work
    # self.assertIs(type(n), Int)


def test_error_message() -> None:
    def check(s, base=None):
        with pytest.raises(ValueError):
            if base is None:
                Int(s)
            else:
                Int(s, base)

    check("\xbd")
    check("123\xbd")
    check("  123 456  ")

    check("123\x00")
    # SF bug 1545497: embedded NULs were not detected with explicit base
    check("123\x00", 10)
    check("123\x00 245", 20)
    check("123\x00 245", 16)
    check("123\x00245", 20)
    check("123\x00245", 16)
    # byte string with embedded NUL
    check(b"123\x00")
    check(b"123\x00", 10)
    # non-UTF-8 byte string
    check(b"123\xbd")
    check(b"123\xbd", 10)
    # lone surrogate in Unicode string
    check("123\ud800")
    check("123\ud800", 10)


def test_protobof_schema():
    assert Int.get_protobuf_schema()


def test_bytes():
    assert (
        Int.from_bytes(Int(42).to_bytes(4, "big", signed=True), "big", signed=True)
        == 42
    )
    assert (
        Int.from_bytes(Int(-42).to_bytes(4, "big", signed=True), "big", signed=True)
        == -42
    )
