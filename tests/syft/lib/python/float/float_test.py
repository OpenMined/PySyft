"""
Tests copied from cpython test suite:
https://github.com/python/cpython/blob/3.9/Lib/test/test_float.py
"""

# stdlib
from math import copysign
from math import isinf
from math import isnan
from math import ldexp
import operator
import random
import sys
import time

# third party
import pytest

# syft absolute
from syft.lib.python.float import Float

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

INF = Float("inf")
NAN = Float("nan")

have_getformat = hasattr(Float, "__getformat__")


class FloatSubclass(Float):
    pass


class OtherFloatSubclass(Float):
    pass


class F:
    def __float__(self):
        return OtherFloatSubclass(42.0)


def test_Float():
    assert Float(3.14) == 3.14
    assert Float(314) == 314.0
    assert Float("  3.14  ") == 3.14
    with pytest.raises(ValueError):
        Float("  0x3.1  ")
    with pytest.raises(ValueError):
        Float("  -0x3.p-1  ")
    with pytest.raises(ValueError):
        Float("  +0x3.p-1  ")
    with pytest.raises(ValueError):
        Float("++3.14")
    with pytest.raises(ValueError):
        Float("+-3.14")
    with pytest.raises(ValueError):
        Float("-+3.14")
    with pytest.raises(ValueError):
        Float("--3.14")
    with pytest.raises(ValueError):
        Float(".nan")
    with pytest.raises(ValueError):
        Float("+.inf")
    with pytest.raises(ValueError):
        Float(".")
    with pytest.raises(ValueError):
        Float("-.")
    with pytest.raises(TypeError):
        Float({})

    with pytest.raises(TypeError):
        Float({})
    # assertRaisesRegex(TypeError, "not 'dict'", Float, {})
    # Lone surrogate
    with pytest.raises(ValueError):
        Float("\uD8F0")

    # check that we don't accept alternate exponent markers
    with pytest.raises(ValueError):
        Float("-1.7d29")

    with pytest.raises(ValueError):
        Float("3D-14")

    assert Float("  \u0663.\u0661\u0664  ") == 3.14
    assert Float("\N{EM SPACE}3.14\N{EN SPACE}") == 3.14

    # extra long strings should not be a problem
    Float(b"." + b"1" * 1000)
    Float("." + "1" * 1000)

    with pytest.raises(ValueError):
        Float("\u3053\u3093\u306b\u3061\u306f")


def test_underscores():
    for lit in VALID_UNDERSCORE_LITERALS:
        if not any(ch in lit for ch in "jJxXoObB"):
            assert Float(lit) == eval(lit)
            # TODO this should work!!!
            assert Float(lit) == Float(lit.replace("_", ""))
    for lit in INVALID_UNDERSCORE_LITERALS:
        if lit in ("0_7", "09_99"):  # octals are not recognized here
            continue
        if not any(ch in lit for ch in "jJxXoObB"):
            with pytest.raises(ValueError):
                Float(lit)
    # Additional test cases; nan and inf are never valid as literals,
    # only in the Float() constructor, but we don't allow underscores
    # in or around them.
    with pytest.raises(ValueError):
        Float("_NaN")
    with pytest.raises(ValueError):
        Float("Na_N")
    with pytest.raises(ValueError):
        Float("IN_F")
    with pytest.raises(ValueError):
        Float("-_INF")
    with pytest.raises(ValueError):
        Float("-INF_")
    # Check that we handle bytes values correctly.
    with pytest.raises(ValueError):
        Float(b"0_.\xff9")


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
        # stdlib
        from array import array
    except ImportError:
        pass
    else:
        factories.append(lambda b: array("B", b))

    for f in factories:
        x = f(b" 3.14  ")
        assert Float(x) == 3.14
        with pytest.raises(ValueError):
            Float(f(b"A" * 0x10))


def test_Float_memoryview():
    assert Float(memoryview(b"12.3")[1:4]) == 2.3
    assert Float(memoryview(b"12.3\x00")[1:4]) == 2.3
    assert Float(memoryview(b"12.3 ")[1:4]) == 2.3
    assert Float(memoryview(b"12.3A")[1:4]) == 2.3
    assert Float(memoryview(b"12.34")[1:4]) == 2.3


def test_error_message():
    def check(s):
        with pytest.raises(ValueError):
            Float(s)

    check("\xbd")
    check("123\xbd")
    check("  123 456  ")
    check(b"  123 456  ")

    # non-ascii digits (error came from non-digit '!')
    check("\u0663\u0661\u0664!")
    # embedded NUL
    check("123\x00")
    check("123\x00 245")
    check("123\x00245")
    # byte string with embedded NUL
    check(b"123\x00")
    # non-UTF-8 byte string
    check(b"123\xa0")


def test_Float_with_comma():
    # set locale to something that doesn't use '.' for the decimal point    assert Float("  3.14  ") == 3.14
    # Float must not accept the locale specific decimal point but
    # it still has to accept the normal python syntax
    assert Float("+3.14  ") == 3.14
    assert Float("-3.14  ") == -3.14
    assert Float(".14  ") == 0.14
    assert Float("3.  ") == 3.0
    assert Float("3.e3  ") == 3000.0
    assert Float("3.2e3  ") == 3200.0
    assert Float("2.5e-1  ") == 0.25
    assert Float("5e-1") == 0.5
    with pytest.raises(ValueError):
        Float("  3,14  ")
    with pytest.raises(ValueError):
        Float("  +3,14  ")
    with pytest.raises(ValueError):
        Float("  -3,14  ")
    with pytest.raises(ValueError):
        Float("  0x3.1  ")
    with pytest.raises(ValueError):
        Float("  -0x3.p-1  ")
    with pytest.raises(ValueError):
        Float("  +0x3.p-1  ")
    assert Float("  25.e-1  ") == 2.5


def test_Floatconversion():
    # Make sure that calls to __Float__() work properly
    class Foo1(object):
        def __float__(self):
            return 42.0

    class Foo2(Float):
        def __float__(self):
            return 42.0

    class Foo3(Float):
        def __new__(cls, value=0.0):
            return Float.__new__(cls, 2 * value)

        def __float__(self):
            return self

    class Foo4(Float):
        def __float__(self):
            return 42

    # Issue 5759: __Float__ not called on str subclasses (though it is on
    # unicode subclasses).
    class FooStr(str):
        def __float__(self):
            return Float(str(self)) + 1

    assert Float(Foo1()) == 42.0
    assert Float(Foo2()) == 42.0

    with pytest.raises(TypeError):
        Float(Foo4(42))
    assert Float(FooStr("8")) == 9.0

    class Foo5:
        def __Float__(self):
            return ""

    with pytest.raises(TypeError):
        time.sleep(Foo5())

    # using __index__ in init was added in python 3.8
    # https://github.com/python/cpython/commit/bdbad71b9def0b86433de12cecca022eee91bd9f

    if sys.version_info >= (3, 8):

        class MyIndex:
            def __init__(self, value):
                self.value = value

            def __index__(self):
                return self.value

        assert Float(MyIndex(42)) == 42.0
        with pytest.raises(OverflowError):
            Float(MyIndex(2 ** 2000))

    class MyInt:
        def __int__(self):
            return 42

    with pytest.raises(TypeError):
        Float(MyInt())


def test_keyword_args():
    with pytest.raises(TypeError):
        Float(x="3.14")


def test_is_integer():
    assert not Float(1.1).is_integer()
    assert Float(1.0).is_integer()
    assert not Float("nan").is_integer()
    assert not Float("inf").is_integer()


def test_Floatasratio():
    for f, ratio in [
        (0.875, (7, 8)),
        (-0.875, (-7, 8)),
        (0.0, (0, 1)),
        (11.5, (23, 2)),
    ]:
        assert f.as_integer_ratio() == ratio

    for i in range(10000):
        f = random.random()
        f *= 10 ** random.randint(-100, 100)
        n, d = f.as_integer_ratio()
        assert Float(n).__truediv__(d) == f

    with pytest.raises(OverflowError):
        Float("inf").as_integer_ratio()
    with pytest.raises(OverflowError):
        Float("-inf").as_integer_ratio()
    with pytest.raises(ValueError):
        Float("nan").as_integer_ratio()


def test_Float_containment():
    Floats = (INF, -INF, 0.0, 1.0, NAN)
    for f in Floats:
        assert f in [f]
        assert f in (f,)
        assert f in {f}
        assert f in {f: None}
        assert [f].count(f), 1 == f"[].count('{f!r}') != 1"
        assert f in Floats

    for f in Floats:
        # nonidentical containers, same type, same contents
        assert [f] == [f]
        assert (f,) == (f,)
        assert {f} == {f}
        assert {f: None} == {f: None}
        # identical containers
        l, t, s, d = [f], (f,), {f}, {f: None}
        assert l == l  # noqa: E741
        assert t == t  # noqa: E741
        assert s == s  # noqa: E741
        assert d == d  # noqa: E741


def test_Float_mod():
    # Check behaviour of % operator for IEEE 754 special cases.
    # In particular, check signs of zeros.
    mod = operator.mod

    assert mod(Float(-1.0), Float(1.0)) == 0.0
    assert mod(Float(-1e-100), Float(1.0)) == 1.0
    assert mod(Float(-0.0), Float(1.0)) == 0.0
    assert mod(Float(0.0), Float(1.0)) == 0.0
    assert mod(Float(1e-100), Float(1.0)) == 1e-100
    assert mod(Float(1.0), Float(1.0)) == 0.0

    assert mod(Float(-1.0), Float(-1.0)) == -0.0
    assert mod(Float(-1e-100), Float(-1.0)) == -1e-100
    assert mod(Float(-0.0), Float(-1.0)) == -0.0
    assert mod(Float(0.0), Float(-1.0)) == -0.0
    assert mod(Float(1e-100), Float(-1.0)) == -1.0
    assert mod(Float(1.0), Float(-1.0)) == -0.0


def test_Float_pow():
    # test builtin pow and ** operator for IEEE 754 special cases.
    # Special cases taken from section F.9.4.4 of the C99 specification

    for pow_op in pow, operator.pow:
        # x**NAN is NAN for any x except 1
        assert isnan(pow_op(-INF, NAN))
        assert isnan(pow_op(Float(Float(-2.0)), NAN))
        assert isnan(pow_op(Float(-1.0), NAN))
        assert isnan(pow_op(Float(-0.5), NAN))
        assert isnan(pow_op(Float(0.5), NAN))
        assert isnan(pow_op(Float(2.0), NAN))
        assert isnan(pow_op(INF, NAN))
        assert isnan(pow_op(NAN, NAN))

        # NAN**y is NAN for any y except +-0
        assert isnan(pow_op(NAN, -INF))
        assert isnan(pow_op(NAN, Float(-2.0)))
        assert isnan(pow_op(NAN, Float(-1.0)))
        assert isnan(pow_op(NAN, Float(-0.5)))
        assert isnan(pow_op(NAN, Float(0.5)))
        assert isnan(pow_op(NAN, Float(1.0)))
        assert isnan(pow_op(NAN, Float(2.0)))
        assert isnan(pow_op(NAN, INF))

        # (+-0)**y raises ZeroDivisionError for y a negative odd integer
        with pytest.raises(ZeroDivisionError):
            pow_op(Float(-0.0), Float(-1.0))

        with pytest.raises(ZeroDivisionError):
            pow_op(Float(0.0), Float(-1.0))

        # (+-0)**y raises ZeroDivisionError for y finite and negative
        # but not an odd integer
        with pytest.raises(ZeroDivisionError):
            pow_op(Float(-0.0), Float(-2.0))
        with pytest.raises(ZeroDivisionError):
            pow_op(Float(-0.0), Float(-0.5))
        with pytest.raises(ZeroDivisionError):
            pow_op(Float(0.0), Float(-2.0))
        with pytest.raises(ZeroDivisionError):
            pow_op(Float(0.0), Float(-0.5))

        # (+-0)**y is +-0 for y a positive odd integer
        assert pow_op(Float(-0.0), Float(1.0)) == -0.0
        assert pow_op(Float(0.0), Float(1.0)) == 0.0
        # (+-0)**y is 0 for y finite and positive but not an odd integer
        assert pow_op(Float(-0.0), Float(0.5)) == 0.0
        assert pow_op(Float(-0.0), Float(2.0)) == 0.0
        assert pow_op(Float(0.0), Float(0.5)) == 0.0
        assert pow_op(Float(0.0), Float(2.0)) == 0.0

        # (-1)**+-inf is 1
        assert pow_op(Float(-1.0), -INF) == 1.0

        # 1**y is 1 for any y, even if y is an infinity or nan
        assert pow_op(Float(1.0), -INF) == 1.0
        assert pow_op(Float(1.0), Float(-2.0)) == 1.0
        assert pow_op(Float(1.0), Float(-1.0)) == 1.0
        assert pow_op(Float(1.0), Float(-0.5)) == 1.0
        assert pow_op(Float(1.0), Float(-0.0)) == 1.0
        assert pow_op(Float(1.0), Float(0.0)) == 1.0
        assert pow_op(Float(1.0), Float(0.5)) == 1.0
        assert pow_op(Float(1.0), Float(1.0)) == 1.0
        assert pow_op(Float(1.0), Float(2.0)) == 1.0
        assert pow_op(Float(1.0), INF) == 1.0
        assert pow_op(Float(1.0), NAN) == 1.0

        # x**+-0 is 1 for any x, even if x is a zero, infinity, or nan
        assert pow_op(-INF, Float(0.0)) == 1.0
        assert pow_op(Float(-2.0), Float(0.0)) == 1.0
        assert pow_op(Float(-1.0), Float(0.0)) == 1.0
        assert pow_op(Float(-0.5), Float(0.0)) == 1.0
        assert pow_op(Float(-0.0), Float(0.0)) == 1.0
        assert pow_op(Float(0.0), Float(0.0)) == 1.0
        assert pow_op(Float(0.5), Float(0.0)) == 1.0
        assert pow_op(Float(1.0), Float(0.0)) == 1.0
        assert pow_op(Float(2.0), Float(0.0)) == 1.0
        assert pow_op(INF, Float(0.0)) == 1.0
        assert pow_op(NAN, Float(0.0)) == 1.0
        assert pow_op(-INF, Float(-0.0)) == 1.0
        assert pow_op(Float(-2.0), Float(-0.0)) == 1.0
        assert pow_op(Float(-1.0), Float(-0.0)) == 1.0
        assert pow_op(Float(-0.5), Float(-0.0)) == 1.0
        assert pow_op(Float(-0.0), Float(-0.0)) == 1.0
        assert pow_op(Float(0.0), Float(-0.0)) == 1.0
        assert pow_op(Float(0.5), Float(-0.0)) == 1.0
        assert pow_op(Float(1.0), Float(-0.0)) == 1.0
        assert pow_op(Float(2.0), Float(-0.0)) == 1.0
        assert pow_op(INF, Float(-0.0)) == 1.0
        assert pow_op(NAN, Float(-0.0)) == 1.0

        # x**y defers to complex pow for finite negative x and
        # non-integral y.
        assert isinstance(pow_op(Float(-2.0), Float(-0.5)), complex)
        assert isinstance(pow_op(Float(-2.0), Float(0.5)), complex)
        assert isinstance(pow_op(Float(-1.0), Float(-0.5)), complex)
        assert isinstance(pow_op(-1.0, Float(0.5)), complex)
        assert isinstance(pow_op(Float(-0.5), Float(-0.5)), complex)
        assert isinstance(pow_op(Float(-0.5), Float(0.5)), complex)

        # x**-INF is INF for abs(x) < 1
        assert pow_op(Float(-0.5), -INF) == INF
        assert pow_op(Float(-0.0), -INF) == INF
        assert pow_op(Float(0.0), -INF) == INF
        assert pow_op(Float(0.5), -INF) == INF

        # x**-INF is 0 for abs(x) > 1
        assert pow_op(-INF, -INF) == 0.0
        assert pow_op(Float(-2.0), -INF) == 0.0
        assert pow_op(Float(2.0), -INF) == 0.0
        assert pow_op(INF, -INF) == 0.0

        # x**INF is 0 for abs(x) < 1
        assert pow_op(Float(-0.5), INF) == 0.0
        assert pow_op(Float(-0.0), INF) == 0.0
        assert pow_op(Float(0.0), INF) == 0.0
        assert pow_op(Float(0.5), INF) == 0.0

        # x**INF is INF for abs(x) > 1
        assert pow_op(-INF, INF) == INF
        assert pow_op(Float(-2.0), INF) == INF
        assert pow_op(Float(2.0), INF) == INF
        assert pow_op(INF, INF) == INF

        # (-INF)**y is -0.0 for y a negative odd integer
        assert pow_op(-INF, Float(-1.0)) == -0.0

        # (-INF)**y is 0.0 for y negative but not an odd integer
        assert pow_op(-INF, Float(-0.5)) == 0.0
        assert pow_op(-INF, Float(-2.0)) == 0.0

        # (-INF)**y is -INF for y a positive odd integer
        assert pow_op(-INF, Float(1.0)) == -INF

        # (-INF)**y is INF for y positive but not an odd integer
        assert pow_op(-INF, Float(0.5)) == INF
        assert pow_op(-INF, Float(2.0)) == INF

        # INF**y is INF for y positive
        assert pow_op(INF, Float(0.5)) == INF
        assert pow_op(INF, Float(1.0)) == INF
        assert pow_op(INF, Float(2.0)) == INF

        # INF**y is 0.0 for y negative
        assert pow_op(INF, Float(-2.0)) == 0.0
        assert pow_op(INF, Float(-1.0)) == 0.0
        assert pow_op(INF, Float(-0.5)) == 0.0

        # basic checks not covered by the special cases above
        assert pow_op(Float(-2.0), Float(-2.0)) == 0.25
        assert pow_op(Float(-2.0), Float(-1.0)) == Float(-0.5)
        assert pow_op(Float(-2.0), Float(-0.0)) == Float(1.0)
        assert pow_op(Float(-2.0), Float(0.0)) == Float(1.0)
        assert pow_op(Float(-2.0), Float(1.0)) == Float(-2.0)
        assert pow_op(Float(-2.0), Float(2.0)) == Float(4.0)
        assert pow_op(Float(-1.0), Float(-2.0)) == Float(1.0)
        assert pow_op(Float(-1.0), Float(-1.0)) == Float(-1.0)
        assert pow_op(Float(-1.0), Float(-0.0)) == Float(1.0)
        assert pow_op(Float(-1.0), Float(0.0)) == Float(1.0)
        assert pow_op(Float(-1.0), Float(1.0)) == Float(-1.0)
        assert pow_op(Float(-1.0), Float(2.0)) == Float(1.0)
        assert pow_op(Float(2.0), Float(-2.0)) == Float(0.25)
        assert pow_op(Float(2.0), Float(-1.0)) == Float(0.5)
        assert pow_op(Float(2.0), Float(-0.0)) == Float(1.0)
        assert pow_op(Float(2.0), Float(0.0)) == Float(1.0)
        assert pow_op(Float(2.0), Float(1.0)) == Float(2.0)
        assert pow_op(Float(2.0), Float(2.0)) == Float(4.0)

        # 1 ** large and -1 ** large; some libms apparently
        # have problems with these
        assert pow_op(Float(1.0), -1e100) == Float(1.0)
        assert pow_op(Float(1.0), 1e100) == Float(1.0)
        assert pow_op(Float(-1.0), -1e100) == Float(1.0)
        assert pow_op(Float(-1.0), 1e100) == Float(1.0)

        # check sign for results that underflow to 0
        assert pow_op(Float(-2.0), Float(-2000.0)) == Float(0.0)
        assert isinstance(pow_op(Float(-2.0), Float(-2000.5)), complex)
        assert pow_op(Float(-2.0), Float(-2001.0)) == Float(-0.0)
        assert pow_op(Float(2.0), Float(-2000.0)) == Float(0.0)
        assert pow_op(Float(2.0), Float(-2000.5)) == Float(0.0)
        assert pow_op(Float(2.0), Float(-2001.0)) == Float(0.0)
        assert pow_op(Float(-0.5), Float(2000.0)) == Float(0.0)
        assert isinstance(pow_op(Float(-0.5), Float(2000.5)), complex)
        assert pow_op(Float(-0.5), Float(2001.0)) == Float(-0.0)
        assert pow_op(Float(0.5), Float(2000.0)) == Float(0.0)
        assert pow_op(Float(0.5), Float(2000.5)) == Float(0.0)
        assert pow_op(Float(0.5), Float(2001.0)) == Float(0.0)


def test_short_repr():
    # test short Float repr introduced in Python 3.1.  One aspect
    # of this repr is that we get some degree of str -> Float ->
    # str roundtripping.  In particular, for any numeric string
    # containing 15 or fewer significant digits, those exact same
    # digits (modulo trailing zeros) should appear in the output.
    # No more repr(0.03) -> "0.029999999999999999"!

    test_strings = [
        # output always includes *either* a decimal point and at
        # least one digit after that point, or an exponent.
        "0.0",
        "1.0",
        "0.01",
        "0.02",
        "0.03",
        "0.04",
        "0.05",
        "1.23456789",
        "10.0",
        "100.0",
        # values >= 1e16 get an exponent...
        "1000000000000000.0",
        "9999999999999990.0",
        "1e+16",
        "1e+17",
        # ... and so do values < 1e-4
        "0.001",
        "0.001001",
        "0.00010000000000001",
        "0.0001",
        "9.999999999999e-05",
        "1e-05",
        # values designed to provoke failure if the FPU rounding
        # precision isn't set correctly
        "8.72293771110361e+25",
        "7.47005307342313e+26",
        "2.86438000439698e+28",
        "8.89142905246179e+28",
        "3.08578087079232e+35",
    ]

    for s in test_strings:
        negs = "-" + s
        assert s == repr(Float(s))
        assert negs == repr(Float(negs))
        # Since Python 3.2, repr and str are identical
        assert repr(Float(s)) == str(Float(s))
        assert repr(Float(negs)) == str(Float(negs))


def test_inf_nan():
    with pytest.raises(OverflowError):
        round(INF)
    with pytest.raises(OverflowError):
        round(-INF)
    with pytest.raises(ValueError):
        round(NAN)
    with pytest.raises(TypeError):
        round(INF, 0.0)
    with pytest.raises(TypeError):
        round(-INF, 1.0)
    with pytest.raises(TypeError):
        round(NAN("ceci n'est pas un integer"))
    with pytest.raises(TypeError):
        round(-0.0, 1j)


def test_large_n():
    for n in [324, 325, 400, 2 ** 31 - 1, 2 ** 31, 2 ** 32, 2 ** 100]:
        assert round(123.456, n) == 123.456
        assert round(-123.456, n) == -123.456
        assert round(1e300, n) == 1e300
        assert round(1e-320, n) == 1e-320
    assert round(1e150, 300) == 1e150
    assert round(1e300, 307) == 1e300
    assert round(-3.1415, 308) == -3.1415
    assert round(1e150, 309) == 1e150
    assert round(1.4e-315, 315) == 1e-315


def test_small_n():
    for n in [-308, -309, -400, 1 - 2 ** 31, -(2 ** 31), -(2 ** 31) - 1, -(2 ** 100)]:
        assert round(123.456, n) == 0.0
        assert round(-123.456, n) == -0.0
        assert round(1e300, n) == 0.0
        assert round(1e-320, n) == 0.0


def test_overflow():
    with pytest.raises(OverflowError):
        round(1.6e308, -308)
    with pytest.raises(OverflowError):
        round(-1.7e308, -308)


def test_previous_round_bugs():
    # particular cases that have occurred in bug reports
    assert round(562949953421312.5, 1) == 562949953421312.5
    assert round(56294995342131.5, 3) == 56294995342131.5
    # round-half-even
    assert round(Float(25.0), -1) == Float(20.0)
    assert round(Float(35.0), -1) == Float(40.0)
    assert round(Float(45.0), -1) == Float(40.0)
    assert round(Float(55.0), -1) == Float(60.0)
    assert round(Float(65.0), -1) == Float(60.0)
    assert round(Float(75.0), -1) == Float(80.0)
    assert round(Float(85.0), -1) == Float(80.0)
    assert round(Float(95.0), -1) == Float(100.0)


@pytest.mark.slow
def test_matches_Float_format():
    # round should give the same results as Float formatting
    for i in range(500):
        x = i / 1000.0
        assert Float(format(x, ".0f")) == round(x, 0)
        assert Float(format(x, ".1f")) == round(x, 1)
        assert Float(format(x, ".2f")) == round(x, 2)
        assert Float(format(x, ".3f")) == round(x, 3)

    for i in range(5, 5000, 10):
        x = i / 1000.0
        assert Float(format(x, ".0f")) == round(x, 0)
        assert Float(format(x, ".1f")) == round(x, 1)
        assert Float(format(x, ".2f")) == round(x, 2)
        assert Float(format(x, ".3f")) == round(x, 3)

    for i in range(500):
        x = random.random()
        assert Float(format(x, ".0f")) == round(x, 0)
        assert Float(format(x, ".1f")) == round(x, 1)
        assert Float(format(x, ".2f")) == round(x, 2)
        assert Float(format(x, ".3f")) == round(x, 3)


def test_None_ndigits():
    for x in round(1.23), round(1.23, None), round(1.23, ndigits=None):
        assert x == 1
        assert isinstance(x, int)
    for x in round(1.78), round(1.78, None), round(1.78, ndigits=None):
        assert x == 2
        assert isinstance(x, int)


# Beginning with Python 2.6 Float has cross platform compatible
# ways to create and represent inf and nan


def test_inf_from_str():
    assert isinf(Float("inf"))
    assert isinf(Float("+inf"))
    assert isinf(Float("-inf"))
    assert isinf(Float("infinity"))
    assert isinf(Float("+infinity"))
    assert isinf(Float("-infinity"))

    assert repr(Float("inf")) == "inf"
    assert repr(Float("+inf")) == "inf"
    assert repr(Float("-inf")) == "-inf"
    assert repr(Float("infinity")) == "inf"
    assert repr(Float("+infinity")) == "inf"
    assert repr(Float("-infinity")) == "-inf"

    assert repr(Float("INF")) == "inf"
    assert repr(Float("+Inf")) == "inf"
    assert repr(Float("-iNF")) == "-inf"
    assert repr(Float("Infinity")) == "inf"
    assert repr(Float("+iNfInItY")) == "inf"
    assert repr(Float("-INFINITY")) == "-inf"

    assert str(Float("inf")) == "inf"
    assert str(Float("+inf")) == "inf"
    assert str(Float("-inf")) == "-inf"
    assert str(Float("infinity")) == "inf"
    assert str(Float("+infinity")) == "inf"
    assert str(Float("-infinity")) == "-inf"

    with pytest.raises(ValueError):
        Float("info")
    with pytest.raises(ValueError):
        Float("+info")
    with pytest.raises(ValueError):
        Float("-info")
    with pytest.raises(ValueError):
        Float("in")
    with pytest.raises(ValueError):
        Float("+in")
    with pytest.raises(ValueError):
        Float("-in")
    with pytest.raises(ValueError):
        Float("infinit")
    with pytest.raises(ValueError):
        Float("+Infin")
    with pytest.raises(ValueError):
        Float("-INFI")
    with pytest.raises(ValueError):
        Float("infinitys")

    with pytest.raises(ValueError):
        Float("++Inf")
    with pytest.raises(ValueError):
        Float("-+inf")
    with pytest.raises(ValueError):
        Float("+-infinity")
    with pytest.raises(ValueError):
        Float("--Infinity")


def test_inf_as_str():
    assert repr(1e300 * 1e300) == "inf"
    assert repr(-1e300 * 1e300) == "-inf"

    assert str(1e300 * 1e300) == "inf"
    assert str(-1e300 * 1e300) == "-inf"


def test_nan_from_str():
    assert isnan(Float("nan"))
    assert isnan(Float("+nan"))
    assert isnan(Float("-nan"))

    assert repr(Float("nan")) == "nan"
    assert repr(Float("+nan")) == "nan"
    assert repr(Float("-nan")) == "nan"

    assert repr(Float("NAN")) == "nan"
    assert repr(Float("+NAn")) == "nan"
    assert repr(Float("-NaN")) == "nan"

    assert str(Float("nan")) == "nan"
    assert str(Float("+nan")) == "nan"
    assert str(Float("-nan")) == "nan"

    with pytest.raises(ValueError):
        Float("nana")
    with pytest.raises(ValueError):
        Float("+nana")
    with pytest.raises(ValueError):
        Float("-nana")
    with pytest.raises(ValueError):
        Float("na")
    with pytest.raises(ValueError):
        Float("+na")
    with pytest.raises(ValueError):
        Float("-na")

    with pytest.raises(ValueError):
        Float("++nan")
    with pytest.raises(ValueError):
        Float("-+NAN")
    with pytest.raises(ValueError):
        Float("+-NaN")
    with pytest.raises(ValueError):
        Float("--nAn")


def test_nan_as_str():
    assert repr(1e300 * 1e300 * 0) == "nan"
    assert repr(-1e300 * 1e300 * 0) == "nan"

    assert str(1e300 * 1e300 * 0) == "nan"
    assert str(-1e300 * 1e300 * 0) == "nan"


def test_inf_signs():
    assert copysign(1.0, Float("inf")) == 1.0
    assert copysign(1.0, Float("-inf")) == -1.0


def test_nan_signs():
    # When using the dtoa.c code, the sign of Float('nan') should
    # be predictable.
    assert copysign(1.0, Float("nan")) == 1.0
    assert copysign(1.0, Float("-nan")) == -1.0


fromHex = Float.fromhex
toHex = Float.hex

MAX = fromHex("0x.fffffffffffff8p+1024")  # max normal
MIN = fromHex("0x1p-1022")  # min normal
TINY = fromHex("0x0.0000000000001p-1022")  # min subnormal
EPS = fromHex("0x0.0000000000001p0")  # diff between 1.0 and next Float up


def identical(x, y):
    # check that Floats x and y are identical, or that both
    # are NaNs
    if isnan(x) or isnan(y):
        if isnan(x) == isnan(y):
            return
    elif x == y and (x != 0.0 or copysign(1.0, x) == copysign(1.0, y)):
        return
    pytest.fail(f"{x!r} not identical to {y!r}")


def test_ends():
    identical(MIN, ldexp(1.0, -1022))
    identical(TINY, ldexp(1.0, -1074))
    identical(EPS, ldexp(1.0, -52))
    identical(MAX, 2.0 * (ldexp(1.0, 1023) - ldexp(1.0, 970)))


def test_invalid_inputs():
    invalid_inputs = [
        "infi",  # misspelt infinities and nans
        "-Infinit",
        "++inf",
        "-+Inf",
        "--nan",
        "+-NaN",
        "snan",
        "NaNs",
        "nna",
        "an",
        "nf",
        "nfinity",
        "inity",
        "iinity",
        "0xnan",
        "",
        " ",
        "x1.0p0",
        "0xX1.0p0",
        "+ 0x1.0p0",  # internal whitespace
        "- 0x1.0p0",
        "0 x1.0p0",
        "0x 1.0p0",
        "0x1 2.0p0",
        "+0x1 .0p0",
        "0x1. 0p0",
        "-0x1.0 1p0",
        "-0x1.0 p0",
        "+0x1.0p +0",
        "0x1.0p -0",
        "0x1.0p 0",
        "+0x1.0p+ 0",
        "-0x1.0p- 0",
        "++0x1.0p-0",  # double signs
        "--0x1.0p0",
        "+-0x1.0p+0",
        "-+0x1.0p0",
        "0x1.0p++0",
        "+0x1.0p+-0",
        "-0x1.0p-+0",
        "0x1.0p--0",
        "0x1.0.p0",
        "0x.p0",  # no hex digits before or after point
        "0x1,p0",  # wrong decimal point character
        "0x1pa",
        "0x1p\uff10",  # fullwidth Unicode digits
        "\uff10x1p0",
        "0x\uff11p0",
        "0x1.\uff10p0",
        "0x1p0 \n 0x2p0",
        "0x1p0\0 0x1p0",  # embedded null byte is not end of string
    ]
    for x in invalid_inputs:
        try:
            result = fromHex(x)
        except ValueError:
            pass
        else:
            pytest.fail(
                "Expected Float.fromhex(%r) to raise ValueError; "
                "got %r instead" % (x, result)
            )
