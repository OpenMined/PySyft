import operator
import random
import time
import pytest
from syft.lib.python.float import Float
from test import support
from tests.syft.lib.python.util import (
    VALID_UNDERSCORE_LITERALS,
    INVALID_UNDERSCORE_LITERALS,
)

from math import isinf, isnan

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


def test_float():
    assert Float(3.14) == 3.14
    assert Float(314) == 314.0
    assert Float("  3.14  ") == 3.14
    with pytest.raises(ValueError) as e:
        Float("  0x3.1  ")
    with pytest.raises(ValueError) as e:
        Float("  -0x3.p-1  ")
    with pytest.raises(ValueError) as e:
        Float("  +0x3.p-1  ")
    with pytest.raises(ValueError) as e:
        Float("++3.14")
    with pytest.raises(ValueError) as e:
        Float("+-3.14")
    with pytest.raises(ValueError) as e:
        Float("-+3.14")
    with pytest.raises(ValueError) as e:
        Float("--3.14")
    with pytest.raises(ValueError) as e:
        Float(".nan")
    with pytest.raises(ValueError) as e:
        Float("+.inf")
    with pytest.raises(ValueError) as e:
        Float(".")
    with pytest.raises(ValueError) as e:
        Float("-.")
    with pytest.raises(TypeError) as e:
        Float({})

    with pytest.raises(TypeError) as e:
        Float({})

    # Lone surrogate
    with pytest.raises(ValueError) as e:
        Float("\uD8F0")

    # check that we don't accept alternate exponent markers
    with pytest.raises(ValueError) as e:
        Float("-1.7d29")

    with pytest.raises(ValueError) as e:
        Float("3D-14")

    assert Float("  \u0663.\u0661\u0664  ") == 3.14
    assert Float("\N{EM SPACE}3.14\N{EN SPACE}") == 3.14

    # extra long strings should not be a problem
    Float(b"." + b"1" * 1000)
    Float("." + "1" * 1000)

    with pytest.raises(ValueError) as e:
        Float("\u3053\u3093\u306b\u3061\u306f")


def test_underscores():
    for lit in VALID_UNDERSCORE_LITERALS:
        if not any(ch in lit for ch in "jJxXoObB"):
            assert Float(lit) == eval(lit)
            # TODO this should work!!!
            # assert Float(lit), Float(lit.replace("_", ""))

    for lit in INVALID_UNDERSCORE_LITERALS:
        if lit in ("0_7", "09_99"):  # octals are not recognized here
            continue
        if not any(ch in lit for ch in "jJxXoObB"):
            with pytest.raises(ValueError) as e:
                Float(lit)

    # Additional test cases; nan and inf are never valid as literals,
    # only in the Float() constructor, but we don't allow underscores
    # in or around them.
    with pytest.raises(ValueError) as e:
        Float("_NaN")
    with pytest.raises(ValueError) as e:
        Float("Na_N")
    with pytest.raises(ValueError) as e:
        Float("IN_F")
    with pytest.raises(ValueError) as e:
        Float("-_INF")
    with pytest.raises(ValueError) as e:
        Float("-INF_")
    # Check that we handle bytes values correctly.
    with pytest.raises(ValueError) as e:
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
        from array import array
    except ImportError:
        pass
    else:
        factories.append(lambda b: array("B", b))

    for f in factories:
        x = f(b" 3.14  ")
        assert Float(x) == 3.14
        with pytest.raises(ValueError) as e:
            Float(f(b"A" * 0x10))


def test_Float_memoryview():
    assert Float(memoryview(b"12.3")[1:4]) == 2.3
    assert Float(memoryview(b"12.3\x00")[1:4]) == 2.3
    assert Float(memoryview(b"12.3 ")[1:4]) == 2.3
    assert Float(memoryview(b"12.3A")[1:4]) == 2.3
    assert Float(memoryview(b"12.34")[1:4]) == 2.3


def test_error_message():
    def check(s):
        with pytest.raises(ValueError) as cm:
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
    # set locale to something that doesn't use '.' for the decimal point
    # Float must not accept the locale specific decimal point but
    # it still has to accept the normal python syntax

    assert Float("  3.14  ") == 3.14
    assert Float("+3.14  ") == 3.14
    assert Float("-3.14  ") == -3.14
    assert Float(".14  ") == 0.14
    assert Float("3.  ") == 3.0
    assert Float("3.e3  ") == 3000.0
    assert Float("3.2e3  ") == 3200.0
    assert Float("2.5e-1  ") == 0.25
    assert Float("5e-1") == 0.5
    with pytest.raises(ValueError) as e:
        Float("  3,14  ")
    with pytest.raises(ValueError) as e:
        Float("  +3,14  ")
    with pytest.raises(ValueError) as e:
        Float("  -3,14  ")
    with pytest.raises(ValueError) as e:
        Float("  0x3.1  ")
    with pytest.raises(ValueError) as e:
        Float("  -0x3.p-1  ")
    with pytest.raises(ValueError) as e:
        Float("  +0x3.p-1  ")
    assert Float("  25.e-1  ") == 2.5


def test_Floatconversion():
    # Make sure that calls to __float__() work properly
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

    with pytest.raises(TypeError) as e:
        Float(Foo4(42))
    assert Float(FooStr("8")) == 9.0

    class Foo5:
        def __Float__(self):
            return ""

    with pytest.raises(TypeError) as e:
        time.sleep(Foo5())

    class MyIndex:
        def __init__(self, value):
            self.value = value

        def __index__(self):
            return self.value

    assert Float(MyIndex(42)) == 42.0
    with pytest.raises(OverflowError) as e:
        Float(MyIndex(2 ** 2000))

    class MyInt:
        def __int__(self):
            return 42

    with pytest.raises(TypeError) as e:
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
        assert [f].count(f), 1 == "[].count('%r') != 1" % f
        assert f in Floats

    for f in Floats:
        # nonidentical containers, same type, same contents
        assert [f] == [f]
        assert (f,) == (f,)
        assert {f} == {f}
        assert {f: None} == {f: None}

        # identical containers
        l, t, s, d = [f], (f,), {f}, {f: None}
        assert l == l
        assert t == t
        assert s == s
        assert d == d


@support.requires_IEEE_754
def test_Float_mod():
    # Check behaviour of % operator for IEEE 754 special cases.
    # In particular, check signs of zeros.
    mod = operator.mod

    assert mod(-1.0, 1.0) == 0.0
    assert mod(-1e-100, 1.0) == 1.0
    assert mod(-0.0, 1.0) == 0.0
    assert mod(0.0, 1.0) == 0.0
    assert mod(1e-100, 1.0) == 1e-100
    assert mod(1.0, 1.0) == 0.0

    assert mod(-1.0, -1.0) == -0.0
    assert mod(-1e-100, -1.0) == -1e-100
    assert mod(-0.0, -1.0) == -0.0
    assert mod(0.0, -1.0) == -0.0
    assert mod(1e-100, -1.0) == -1.0
    assert mod(1.0, -1.0) == -0.0


def test_Float_pow():
    # test builtin pow and ** operator for IEEE 754 special cases.
    # Special cases taken from section F.9.4.4 of the C99 specification

    for pow_op in pow, operator.pow:
        # x**NAN is NAN for any x except 1
        assert isnan(pow_op(-INF, NAN))
        assert isnan(pow_op(-2.0, NAN))
        assert isnan(pow_op(-1.0, NAN))
        assert isnan(pow_op(-0.5, NAN))
        assert isnan(pow_op(0.5, NAN))
        assert isnan(pow_op(2.0, NAN))
        assert isnan(pow_op(INF, NAN))
        assert isnan(pow_op(NAN, NAN))

        # NAN**y is NAN for any y except +-0
        assert isnan(pow_op(NAN, -INF))
        assert isnan(pow_op(NAN, -2.0))
        assert isnan(pow_op(NAN, -1.0))
        assert isnan(pow_op(NAN, -0.5))
        assert isnan(pow_op(NAN, 0.5))
        assert isnan(pow_op(NAN, 1.0))
        assert isnan(pow_op(NAN, 2.0))
        assert isnan(pow_op(NAN, INF))

        # (+-0)**y raises ZeroDivisionError for y a negative odd integer
        with pytest.raises(ZeroDivisionError) as e:
            pow_op(-0.0, -1.0)
        with pytest.raises(ZeroDivisionError) as e:
            pow_op(0.0, -1.0)

        # (+-0)**y raises ZeroDivisionError for y finite and negative
        # but not an odd integer
        with pytest.raises(ZeroDivisionError) as e:
            pow_op(-0.0, -2.0)
        with pytest.raises(ZeroDivisionError) as e:
            pow_op(-0.0, -0.5)
        with pytest.raises(ZeroDivisionError) as e:
            pow_op(0.0, -2.0)
        with pytest.raises(ZeroDivisionError) as e:
            pow_op(0.0, -0.5)

        # (+-0)**y is +-0 for y a positive odd integer
        assert pow_op(-0.0, 1.0) == -0.0
        assert pow_op(0.0, 1.0) == 0.0

        # (+-0)**y is 0 for y finite and positive but not an odd integer
        assert pow_op(-0.0, 0.5) == 0.0
        assert pow_op(-0.0, 2.0) == 0.0
        assert pow_op(0.0, 0.5) == 0.0
        assert pow_op(0.0, 2.0) == 0.0

        # (-1)**+-inf is 1
        assert pow_op(-1.0, -INF) == 1.0

        # 1**y is 1 for any y, even if y is an infinity or nan
        assert pow_op(1.0, -INF) == 1.0
        assert pow_op(1.0, -2.0) == 1.0
        assert pow_op(1.0, -1.0) == 1.0
        assert pow_op(1.0, -0.5) == 1.0
        assert pow_op(1.0, -0.0) == 1.0
        assert pow_op(1.0, 0.0) == 1.0
        assert pow_op(1.0, 0.5) == 1.0
        assert pow_op(1.0, 1.0) == 1.0
        assert pow_op(1.0, 2.0) == 1.0
        assert pow_op(1.0, INF) == 1.0
        assert pow_op(1.0, NAN) == 1.0

        # x**+-0 is 1 for any x, even if x is a zero, infinity, or nan
        assert pow_op(-INF, 0.0) == 1.0
        assert pow_op(-2.0, 0.0) == 1.0
        assert pow_op(-1.0, 0.0) == 1.0
        assert pow_op(-0.5, 0.0) == 1.0
        assert pow_op(-0.0, 0.0) == 1.0
        assert pow_op(0.0, 0.0) == 1.0
        assert pow_op(0.5, 0.0) == 1.0
        assert pow_op(1.0, 0.0) == 1.0
        assert pow_op(2.0, 0.0) == 1.0
        assert pow_op(INF, 0.0) == 1.0
        assert pow_op(NAN, 0.0) == 1.0
        assert pow_op(-INF, -0.0) == 1.0
        assert pow_op(-2.0, -0.0) == 1.0
        assert pow_op(-1.0, -0.0) == 1.0
        assert pow_op(-0.5, -0.0) == 1.0
        assert pow_op(-0.0, -0.0) == 1.0
        assert pow_op(0.0, -0.0) == 1.0
        assert pow_op(0.5, -0.0) == 1.0
        assert pow_op(1.0, -0.0) == 1.0
        assert pow_op(2.0, -0.0) == 1.0
        assert pow_op(INF, -0.0) == 1.0
        assert pow_op(NAN, -0.0) == 1.0

        # x**y defers to complex pow for finite negative x and
        # non-integral y.
        assert type(pow_op(-2.0, -0.5)) == complex
        assert type(pow_op(-2.0, 0.5)) == complex
        assert type(pow_op(-1.0, -0.5)) == complex
        assert type(pow_op(-1.0, 0.5)) == complex
        assert type(pow_op(-0.5, -0.5)) == complex
        assert type(pow_op(-0.5, 0.5)) == complex

        # x**-INF is INF for abs(x) < 1
        assert pow_op(-0.5, -INF) == INF
        assert pow_op(-0.0, -INF) == INF
        assert pow_op(0.0, -INF) == INF
        assert pow_op(0.5, -INF) == INF

        # x**-INF is 0 for abs(x) > 1
        assert pow_op(-INF, -INF) == 0.0
        assert pow_op(-2.0, -INF) == 0.0
        assert pow_op(2.0, -INF) == 0.0
        assert pow_op(INF, -INF) == 0.0

        # x**INF is 0 for abs(x) < 1
        assert pow_op(-0.5, INF) == 0.0
        assert pow_op(-0.0, INF) == 0.0
        assert pow_op(0.0, INF) == 0.0
        assert pow_op(0.5, INF) == 0.0

        # x**INF is INF for abs(x) > 1
        assert pow_op(-INF, INF) == INF
        assert pow_op(-2.0, INF) == INF
        assert pow_op(2.0, INF) == INF
        assert pow_op(INF, INF) == INF

        # (-INF)**y is -0.0 for y a negative odd integer
        assert pow_op(-INF, -1.0) == -0.0

        # (-INF)**y is 0.0 for y negative but not an odd integer
        assert pow_op(-INF, -0.5) == 0.0
        assert pow_op(-INF, -2.0) == 0.0

        # (-INF)**y is -INF for y a positive odd integer
        assert pow_op(-INF, 1.0) == -INF

        # (-INF)**y is INF for y positive but not an odd integer
        assert pow_op(-INF, 0.5) == INF
        assert pow_op(-INF, 2.0) == INF

        # INF**y is INF for y positive
        assert pow_op(INF, 0.5) == INF
        assert pow_op(INF, 1.0) == INF
        assert pow_op(INF, 2.0) == INF

        # INF**y is 0.0 for y negative
        assert pow_op(INF, -2.0) == 0.0
        assert pow_op(INF, -1.0) == 0.0
        assert pow_op(INF, -0.5) == 0.0

        # basic checks not covered by the special cases above
        assert pow_op(-2.0, -2.0) == 0.25
        assert pow_op(-2.0, -1.0) == -0.5
        assert pow_op(-2.0, -0.0) == 1.0
        assert pow_op(-2.0, 0.0) == 1.0
        assert pow_op(-2.0, 1.0) == -2.0
        assert pow_op(-2.0, 2.0) == 4.0
        assert pow_op(-1.0, -2.0) == 1.0
        assert pow_op(-1.0, -1.0) == -1.0
        assert pow_op(-1.0, -0.0) == 1.0
        assert pow_op(-1.0, 0.0) == 1.0
        assert pow_op(-1.0, 1.0) == -1.0
        assert pow_op(-1.0, 2.0) == 1.0
        assert pow_op(2.0, -2.0) == 0.25
        assert pow_op(2.0, -1.0) == 0.5
        assert pow_op(2.0, -0.0) == 1.0
        assert pow_op(2.0, 0.0) == 1.0
        assert pow_op(2.0, 1.0) == 2.0
        assert pow_op(2.0, 2.0) == 4.0

        # 1 ** large and -1 ** large; some libms apparently
        # have problems with these
        assert pow_op(1.0, -1e100) == 1.0
        assert pow_op(1.0, 1e100) == 1.0
        assert pow_op(-1.0, -1e100) == 1.0
        assert pow_op(-1.0, 1e100) == 1.0

        # check sign for results that underflow to 0
        assert pow_op(-2.0, -2000.0) == 0.0
        assert type(pow_op(-2.0, -2000.5)) == complex
        assert pow_op(-2.0, -2001.0) == -0.0
        assert pow_op(2.0, -2000.0) == 0.0
        assert pow_op(2.0, -2000.5) == 0.0
        assert pow_op(2.0, -2001.0) == 0.0
        assert pow_op(-0.5, 2000.0) == 0.0
        assert type(pow_op(-0.5, 2000.5)) == complex
        assert pow_op(-0.5, 2001.0) == -0.0
        assert pow_op(0.5, 2000.0) == 0.0
        assert pow_op(0.5, 2000.5) == 0.0
        assert pow_op(0.5, 2001.0) == 0.0


BE_DOUBLE_INF = b"\x7f\xf0\x00\x00\x00\x00\x00\x00"
LE_DOUBLE_INF = bytes(reversed(BE_DOUBLE_INF))
BE_DOUBLE_NAN = b"\x7f\xf8\x00\x00\x00\x00\x00\x00"
LE_DOUBLE_NAN = bytes(reversed(BE_DOUBLE_NAN))

BE_Float_INF = b"\x7f\x80\x00\x00"
LE_Float_INF = bytes(reversed(BE_Float_INF))
BE_Float_NAN = b"\x7f\xc0\x00\x00"
LE_Float_NAN = bytes(reversed(BE_Float_NAN))


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
    with pytest.raises(OverflowError) as e:
        round(INF)
    with pytest.raises(OverflowError) as e:
        round(-INF)
    with pytest.raises(ValueError) as e:
        round(NAN)
    with pytest.raises(TypeError) as e:
        round(INF, 0.0)
    with pytest.raises(TypeError) as e:
        round(-INF, 1.0)
    with pytest.raises(TypeError) as e:
        round(NAN("ceci n'est pas un integer"))
    with pytest.raises(TypeError) as e:
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
    with pytest.raises(OverflowError) as e:
        round(1.6e308, -308)
    with pytest.raises(OverflowError) as e:
        round(-1.7e308, -308)


def test_previous_round_bugs():
    # particular cases that have occurred in bug reports
    assert round(25.0, -1) == 20.0
    assert round(35.0, -1) == 40.0
    assert round(45.0, -1) == 40.0
    assert round(55.0, -1) == 60.0
    assert round(65.0, -1) == 60.0
    assert round(75.0, -1) == 80.0
    assert round(85.0, -1) == 80.0
    assert round(95.0, -1) == 100.0


def test_matches_Float_format():
    # round should give the same results as Float formatting
    for i in range(500):
        x = i / 1000.0
        assert Float(format(x, ".0f")) == round(x, 0)
        assert Float(format(x, ".1f")) == round(x, 1)
        assert Float(format(x, ".2f")) == round(x, 2)

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

    with pytest.raises(ValueError) as e:
        Float("info")
    with pytest.raises(ValueError) as e:
        Float("+info")
    with pytest.raises(ValueError) as e:
        Float("-info")
    with pytest.raises(ValueError) as e:
        Float("in")
    with pytest.raises(ValueError) as e:
        Float("+in")
    with pytest.raises(ValueError) as e:
        Float("-in")
    with pytest.raises(ValueError) as e:
        Float("infinit")
    with pytest.raises(ValueError) as e:
        Float("+Infin")
    with pytest.raises(ValueError) as e:
        Float("-INFI")
    with pytest.raises(ValueError) as e:
        Float("infinitys")

    with pytest.raises(ValueError) as e:
        Float("++Inf")
    with pytest.raises(ValueError) as e:
        Float("-+inf")
    with pytest.raises(ValueError) as e:
        Float("+-infinity")
    with pytest.raises(ValueError) as e:
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

    with pytest.raises(ValueError) as e:
        Float("nana")
    with pytest.raises(ValueError) as e:
        Float("+nana")
    with pytest.raises(ValueError) as e:
        Float("-nana")
    with pytest.raises(ValueError) as e:
        Float("na")
    with pytest.raises(ValueError) as e:
        Float("+na")
    with pytest.raises(ValueError) as e:
        Float("-na")

    with pytest.raises(ValueError) as e:
        Float("++nan")
    with pytest.raises(ValueError) as e:
        Float("-+NAN")
    with pytest.raises(ValueError) as e:
        Float("+-NaN")
    with pytest.raises(ValueError) as e:
        Float("--nAn")


def test_nan_as_str():
    assert repr(1e300 * 1e300 * 0) == "nan"
    assert repr(-1e300 * 1e300 * 0) == "nan"

    assert str(1e300 * 1e300 * 0) == "nan"
    assert str(-1e300 * 1e300 * 0) == "nan"


fromHex = float.fromhex


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
            raise Exception("Expected to work or just ValueError!")
