import pytest
from syft.lib.python.int import Int
import sys

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


def test_basic():
    assert Int(value=314) == 314
    assert Int(value=3.14) == 3
    # # Check that conversion from float truncates towards zero
    assert Int(value=-3.14) == -3
    assert Int(value=3.9) == 3
    assert Int(value=-3.9) == -3
    assert Int(value=3.5) == 3
    assert Int(value=-3.5) == -3
    assert Int(value="-3") == -3
    assert Int(value=" -3 ") == -3
    assert Int(value="\N{EM SPACE}-3\N{EN SPACE}") == -3
    # # Different base:
    assert Int(value="10", base=16) == 16
    # # Test conversion from strings and various anomalies

    for s, v in L:
        for sign in "", "+", "-":
            for prefix in "", " ", "\t", "  \t\t  ":
                ss = prefix + sign + s
                vv = v
                if sign == "-" and v is not ValueError:
                    vv = -v
                try:
                    assert Int(value=ss) == vv
                except ValueError:
                    pass

    s = repr(-1 - sys.maxsize)
    x = Int(value=s)
    assert x + 1 == -sys.maxsize
    assert isinstance(x, int)

    # should return int
    assert Int(value=s[1:]) == sys.maxsize + 1

    # # should return int
    x = Int(value=1e100)
    assert isinstance(x, int)

    x = Int(value=-1e100)
    assert isinstance(x, int)

    # SF bug 434186:  0x80000000/2 != 0x80000000>>1.
    # Worked by accident in Windows release build, but failed in debug build.
    # Failed in all Linux builds.
    x = -1 - sys.maxsize
    assert x >> 1 == x // 2

    x = Int(value="1" * 600)
    assert isinstance(x, int)

    # self.assertRaises(TypeError, int, 1, 12)

    assert Int(value="0o123", base=0) == 83
    assert Int(value="0x123", base=16) == 291

    # Bug 1679: "0x" is not a valid hex literal
    # assert Raises(ValueError, int, "0x", 16)
    # assert Raises(ValueError, int, "0x", 0)

    # self.assertRaises(ValueError, int, "0o", 8)
    # self.assertRaises(ValueError, int, "0o", 0)
    #
    # self.assertRaises(ValueError, int, "0b", 2)
    # self.assertRaises(ValueError, int, "0b", 0)
    #
    # # SF bug 1334662: int(string, base) wrong answers
    # # Various representations of 2**32 evaluated to 0
    # # rather than 2**32 in previous versions

    assert Int(value="100000000000000000000000000000000", base=2) == 4294967296
    assert Int(value="102002022201221111211", base=3) == 4294967296
    assert Int(value="10000000000000000", base=4) == 4294967296
    assert Int(value="32244002423141", base=5) == 4294967296
    assert Int(value="1550104015504", base=6) == 4294967296
    assert Int(value="211301422354", base=7) == 4294967296
    assert Int(value="40000000000", base=8) == 4294967296
    assert Int(value="12068657454", base=9) == 4294967296
    assert Int(value="4294967296", base=10) == 4294967296
    assert Int(value="1904440554", base=11) == 4294967296
    assert Int(value="9ba461594", base=12) == 4294967296
    assert Int(value="535a79889", base=13) == 4294967296
    assert Int(value="2ca5b7464", base=14) == 4294967296
    assert Int(value="1a20dcd81", base=15) == 4294967296
    assert Int(value="100000000", base=16) == 4294967296
    assert Int(value="a7ffda91", base=17) == 4294967296
    assert Int(value="704he7g4", base=18) == 4294967296
    assert Int(value="4f5aff66", base=19) == 4294967296
    assert Int(value="3723ai4g", base=20) == 4294967296
    assert Int(value="281d55i4", base=21) == 4294967296
    assert Int(value="1fj8b184", base=22) == 4294967296
    assert Int(value="1606k7ic", base=23) == 4294967296
    assert Int(value="mb994ag", base=24) == 4294967296
    assert Int(value="hek2mgl", base=25) == 4294967296
    assert Int(value="dnchbnm", base=26) == 4294967296
    assert Int(value="b28jpdm", base=27) == 4294967296
    assert Int(value="8pfgih4", base=28) == 4294967296
    assert Int(value="76beigg", base=29) == 4294967296
    assert Int(value="5qmcpqg", base=30) == 4294967296
    assert Int(value="4q0jto4", base=31) == 4294967296
    assert Int(value="4000000", base=32) == 4294967296
    assert Int(value="3aokq94", base=33) == 4294967296
    assert Int(value="2qhxjli", base=34) == 4294967296
    assert Int(value="2br45qb", base=35) == 4294967296
    assert Int(value="1z141z4", base=36) == 4294967296

    # tests with base 03 2.x the old octal syntax is allowed
    assert Int(value=" 0o123  ", base=0) == 83
    assert Int(value=" 0o123  ", base=0) == 83
    assert Int(value="000", base=0) == 0
    assert Int(value="0o123", base=0) == 83
    assert Int(value="0x123", base=0) == 291
    assert Int(value="0b100", base=0) == 4
    assert Int(value=" 0O123   ", base=0) == 83
    assert Int(value=" 0X123  ", base=0) == 291
    assert Int(value=" 0B100 ", base=0) == 4

    # without base still base 10
    assert Int(value="0123") == 123
    assert Int(value="0123", base=10) == 123

    # tests with prefix and base != 0
    assert Int(value="0x123", base=16) == 291
    assert Int(value="0o123", base=8) == 83
    assert Int(value="0b100", base=2) == 4
    assert Int(value="0X123", base=16) == 291
    assert Int(value="0O123", base=8) == 83
    assert Int(value="0B100", base=2) == 4

    # the code has special checks for the first character after the
    #  type prefix
    # self.assertRaises(ValueError, int, '0b2', 2)
    # self.assertRaises(ValueError, int, '0b02', 2)
    # self.assertRaises(ValueError, int, '0B2', 2)
    # self.assertRaises(ValueError, int, '0B02', 2)
    # self.assertRaises(ValueError, int, '0o8', 8)
    # self.assertRaises(ValueError, int, '0o08', 8)
    # self.assertRaises(ValueError, int, '0O8', 8)
    # self.assertRaises(ValueError, int, '0O08', 8)
    # self.assertRaises(ValueError, int, '0xg', 16)
    # self.assertRaises(ValueError, int, '0x0g', 16)
    # self.assertRaises(ValueError, int, '0Xg', 16)
    # self.assertRaises(ValueError, int, '0X0g', 16)

    # # SF bug 1334662: int(string, base) wrong answers
    # # Checks for proper evaluation of 2**32 + 1
    assert Int(value="100000000000000000000000000000001", base=2) == 4294967297
    assert Int(value="102002022201221111212", base=3) == 4294967297
    assert Int(value="10000000000000001", base=4) == 4294967297
    assert Int(value="32244002423142", base=5) == 4294967297
    assert Int(value="1550104015505", base=6) == 4294967297
    assert Int(value="211301422355", base=7) == 4294967297
    assert Int(value="40000000001", base=8) == 4294967297
    assert Int(value="12068657455", base=9) == 4294967297
    assert Int(value="4294967297", base=10) == 4294967297
    assert Int(value="1904440555", base=11) == 4294967297
    assert Int(value="9ba461595", base=12) == 4294967297
    assert Int(value="535a7988a", base=13) == 4294967297
    assert Int(value="2ca5b7465", base=14) == 4294967297
    assert Int(value="1a20dcd82", base=15) == 4294967297
    assert Int(value="100000001", base=16) == 4294967297
    assert Int(value="a7ffda92", base=17) == 4294967297
    assert Int(value="704he7g5", base=18) == 4294967297
    assert Int(value="4f5aff67", base=19) == 4294967297
    assert Int(value="3723ai4h", base=20) == 4294967297
    assert Int(value="281d55i5", base=21) == 4294967297
    assert Int(value="1fj8b185", base=22) == 4294967297
    assert Int(value="1606k7id", base=23) == 4294967297
    assert Int(value="mb994ah", base=24) == 4294967297
    assert Int(value="hek2mgm", base=25) == 4294967297
    assert Int(value="dnchbnn", base=26) == 4294967297
    assert Int(value="b28jpdn", base=27) == 4294967297
    assert Int(value="8pfgih5", base=28) == 4294967297
    assert Int(value="76beigh", base=29) == 4294967297
    assert Int(value="5qmcpqh", base=30) == 4294967297
    assert Int(value="4q0jto5", base=31) == 4294967297
    assert Int(value="4000001", base=32) == 4294967297
    assert Int(value="3aokq95", base=33) == 4294967297
    assert Int(value="2qhxjlj", base=34) == 4294967297
    assert Int(value="2br45qc", base=35) == 4294967297
    assert Int(value="1z141z5", base=36) == 4294967297


def test_int_underscores():
    for lit in VALID_UNDERSCORE_LITERALS:
        if any(ch in lit for ch in ".eEjJ"):
            continue
        assert Int(value=lit, base=0) == eval(lit)

    for lit in INVALID_UNDERSCORE_LITERALS:
        if any(ch in lit for ch in ".eEjJ"):
            continue
    assert Int(value="1_00", base=3) == 9
    assert Int(value="0_100") == 100  # not valid as a literal!
    assert Int(value=b"1_00") == 100  # byte underscore
    # self.assertRaises(ValueError, int, "_100")
    # self.assertRaises(ValueError, int, "+_100")
    # self.assertRaises(ValueError, int, "1__00")
    # self.assertRaises(ValueError, int, "100_")


def test_small_ints():
    assert Int(value="10") is 10
    assert Int(value="-1") is -1
    assert Int(value=b"10") is 10
    assert Int(value=b"-1") is -1


def test_no_args():
    assert Int() == 0


def test_accepted_bases():
    for base in range(2, 37):
        assert Int(value="0", base=base) == 0


def test_int_memoryview():
    assert Int(value=memoryview(b"123")[1:3]) == 23
    assert Int(value=memoryview(b"123\x00")[1:3]) == 23
    assert Int(value=memoryview(b"123 ")[1:3]) == 23
    assert Int(value=memoryview(b"123A")[1:3]) == 23
    assert Int(value=memoryview(b"1234")[1:3]) == 23


def test_check_base_ranges():
    for base in 2 ** 100, -(2 ** 100), 1, 37:
        with pytest.raises(ValueError):
            Int(value="43", base=base)


def test_int_base_indexable():
    class MyIndexable(object):
        def __init__(self, value):
            self.value = value

        def __index__(self):
            return self.value

    assert Int(value="101", base=MyIndexable(2)) == 5
    assert Int(value="101", base=MyIndexable(10)) == 101
    assert Int(value="101", base=MyIndexable(36)) == 1 + 36 ** 2


def test_int_base_bad_types():
    """Not integer types are not valid bases; issue16772."""
    with pytest.raises(TypeError):
        Int(value="0", base=5.5)
    with pytest.raises(TypeError):
        Int(value="0", base=5.0)


def test_non_numeric_input_types():
    # Test possible non-numeric types for the argument x, including
    # subclasses of the explicitly documented accepted types.
    class CustomStr(str):
        pass

    class CustomBytes(bytes):
        pass

    class CustomByteArray(bytearray):
        pass

    factories = [
        bytes,
        bytearray,
        lambda b: CustomStr(b.decode()),
        CustomBytes,
        CustomByteArray,
        memoryview,
    ]

    try:
        from array import array
    except ImportError:
        pass
    else:
        factories.append(lambda b: array("B", b))

    for f in factories:
        x = f(b"100")
        assert Int(value=x) == 100
        if isinstance(x, (str, bytes, bytearray)):
            assert Int(value=x, base=2) == 4
        else:
            msg = "can't convert non-string"
            with pytest.raises(TypeError) as e:
                Int(value=x, base=2)
            assert str(e.value) == msg

            with pytest.raises(ValueError, "invalid literal"):
                Int(value=f(b"A" * 0x10))


def test_string_float():
    with pytest.raises(ValueError):
        Int(value="1.2")


def test_intconversion():
    class ClassicMissingMethods:
        pass

    with pytest.raises(TypeError):
        Int(value=ClassicMissingMethods)

    class MissingMethods(object):
        pass

    with pytest.raises(TypeError):
        Int(value=MissingMethods())

    class Foo0:
        def __int__(self):
            return 42

    assert Int(value=Foo0()) == 42

    class Classic:
        pass

    for base in (object, Classic):

        class IntOverridesTrunc(base):
            def __int__(self):
                return 42

            def __trunc__(self):
                return -12

        assert Int(IntOverridesTrunc()) == 42

        class JustTrunc(base):
            def __trunc__(self):
                return 42

        assert Int(value=JustTrunc()) == 42

        class ExceptionalTrunc(base):
            def __trunc__(self):
                1 / 0

        with pytest.raises(ZeroDivisionError):
            Int(value=ExceptionalTrunc())

        for trunc_result_base in (object, Classic):

            class Index(trunc_result_base):
                def __index__(self):
                    return 42

            class TruncReturnsNonInt(base):
                def __trunc__(self):
                    return Index()

            assert Int(value=TruncReturnsNonInt()) == 42

            class Intable(trunc_result_base):
                def __int__(self):
                    return 42

            class TruncReturnsNonIndex(base):
                def __trunc__(self):
                    return Intable()

            assert Int(value=TruncReturnsNonInt()) == 42

            class NonIntegral(trunc_result_base):
                def __trunc__(self):
                    # Check that we avoid infinite recursion.
                    return NonIntegral()

            class TruncReturnsNonIntegral(base):
                def __trunc__(self):
                    return NonIntegral()

            try:
                Int(value=TruncReturnsNonIntegral())
            except TypeError as e:
                assert str(e) == "__trunc__ returned non-Integral (type NonIntegral)"

            class BadInt(trunc_result_base):
                def __int__(self):
                    return 42.0

            class TruncReturnsBadInt(base):
                def __trunc__(self):
                    return BadInt()

            with pytest.raises(TypeError):
                Int(value=TruncReturnsBadInt())


def test_int_subclass_with_index():
    class MyIndex(int):
        def __index__(self):
            return 42

    class BadIndex(int):
        def __index__(self):
            return 42.0

    my_int = MyIndex(7)
    assert my_int == 7
    assert Int(value=my_int), 7

    assert Int(value=BadIndex()) == 0


def test_int_subclass_with_int():
    class MyInt(int):
        def __int__(self):
            return 42

    class BadInt(int):
        def __int__(self):
            return 42.0

    my_int = MyInt(7)
    assert my_int == 7
    assert Int(value=my_int) == 42

    my_int = BadInt(7)
    assert my_int == 7
    with pytest.raises(TypeError):
        Int(value=my_int)


def test_int_returns_int_subclass():
    class BadIndex:
        def __index__(self):
            return True

    class BadIndex2(int):
        def __index__(self):
            return True

    class BadInt:
        def __int__(self):
            return True

    class BadInt2(int):
        def __int__(self):
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

    bad_int = BadIndex()
    with pytest.raises(DeprecationWarning):
        n = Int(bad_int)

    assert n == 1
    assert type(n) is Int

    bad_int = BadIndex2()
    n = Int(value=bad_int)
    assert n == 0
    assert type(n) is int

    bad_int = BadInt()
    with pytest.raises(DeprecationWarning):
        n = Int(value=bad_int)

    assert n == 1
    assert type(n) is int

    bad_int = BadInt2()
    with pytest.raises(DeprecationWarning):
        n = Int(value=bad_int)

    assert n == 1
    assert type(n) is int

    bad_int = TruncReturnsBadIndex()
    with pytest.raises(DeprecationWarning):
        n = Int(value=bad_int)

    assert n == 1
    assert n is int

    assert n == 1
    assert type(n) is Int


def test_error_message():
    def check(s, base=None):
        with pytest.raises(ValueError) as cm:
            if base is None:
                Int(value=s)
            else:
                Int(value=s, base=base)

        msg_err = (
            f"invalid literal for int() with base {10 if base is None else base}: '{s}'"
        )
        assert str(cm.value) == msg_err

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
