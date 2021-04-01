# flake8: noqa
"""
Tests copied from cpython test suite:
https://github.com/python/cpython/blob/3.9/Lib/test/test_complex.py
"""

# stdlib
from math import atan2
from math import copysign
from math import isnan
import operator
from random import random
import sys
import unittest

# third party
import pytest

# syft absolute
from syft.lib.python.complex import Complex

INF = float("inf")
NAN = float("nan")

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


class ComplexTest(unittest.TestCase):
    @staticmethod
    def assertIs(a, b):
        assert a == b

    def assertAlmostEqual(self, a, b):
        if isinstance(a, complex):
            if isinstance(b, complex):
                unittest.TestCase.assertAlmostEqual(self, a.real, b.real)
                unittest.TestCase.assertAlmostEqual(self, a.imag, b.imag)
            else:
                unittest.TestCase.assertAlmostEqual(self, a.real, b)
                unittest.TestCase.assertAlmostEqual(self, a.imag, 0.0)
        else:
            if isinstance(b, complex):
                unittest.TestCase.assertAlmostEqual(self, a, b.real)
                unittest.TestCase.assertAlmostEqual(self, 0.0, b.imag)
            else:
                unittest.TestCase.assertAlmostEqual(self, a, b)

    def assertCloseAbs(self, x, y, eps=1e-9):
        """Return true iff floats x and y "are close"."""
        # put the one with larger magnitude second
        if abs(x) > abs(y):
            x, y = y, x
        if y == 0:
            return abs(x) < eps
        if x == 0:
            return abs(y) < eps
        # check that relative difference < eps
        self.assertTrue(abs((x - y) / y) < eps)

    def assertFloatsAreIdentical(self, x, y):
        """assert that floats x and y are identical, in the sense that:
        (1) both x and y are nans, or
        (2) both x and y are infinities, with the same sign, or
        (3) both x and y are zeros, with the same sign, or
        (4) x and y are both finite and nonzero, and x == y

        """
        msg = "floats {!r} and {!r} are not identical"

        if isnan(x) or isnan(y):
            if isnan(x) and isnan(y):
                return
        elif x == y:
            if x != 0.0:
                return
            # both zero; check that signs match
            elif copysign(1.0, x) == copysign(1.0, y):
                return
            else:
                msg += ": zeros have different signs"
        self.fail(msg.format(x, y))

    def assertClose(self, x, y, eps=1e-9):
        """Return true iff complexes x and y "are close"."""
        self.assertCloseAbs(x.real, y.real, eps)
        self.assertCloseAbs(x.imag, y.imag, eps)

    def check_div(self, x, y):
        """Compute Complex z=x*y, and check that z/x==y and z/y==x."""
        z = x * y
        if x != 0:
            q = z / x
            self.assertClose(q, y)
            q = z.__truediv__(x)
            self.assertClose(q, y)
        if y != 0:
            q = z / y
            self.assertClose(q, x)
            q = z.__truediv__(y)
            self.assertClose(q, x)

    @pytest.mark.slow
    def test_truediv(self):
        simple_real = [float(i) for i in range(-5, 6)]
        simple_complex = [Complex(x, y) for x in simple_real for y in simple_real]
        for x in simple_complex:
            for y in simple_complex:
                self.check_div(x, y)

        # A naive Complex division algorithm (such as in 2.0) is very prone to
        # nonsense errors for these (overflows and underflows).
        self.check_div(Complex(1e200, 1e200), 1 + 0j)
        self.check_div(Complex(1e-200, 1e-200), 1 + 0j)

        # Just for fun.
        for i in range(100):
            self.check_div(Complex(random(), random()), Complex(random(), random()))

        self.assertRaises(ZeroDivisionError, Complex.__truediv__, 1 + 1j, 0 + 0j)
        self.assertRaises(OverflowError, pow, 1e200 + 1j, 1e200 + 1j)

        self.assertAlmostEqual(Complex.__truediv__(2 + 0j, 1 + 1j), 1 - 1j)
        self.assertRaises(ZeroDivisionError, Complex.__truediv__, 1 + 1j, 0 + 0j)

        for denom_real, denom_imag in [(0, NAN), (NAN, 0), (NAN, NAN)]:
            z = Complex(0, 0) / Complex(denom_real, denom_imag)
            self.assertTrue(isnan(z.real))
            self.assertTrue(isnan(z.imag))

    def test_floordiv(self):
        self.assertRaises(TypeError, Complex.__floordiv__, 3 + 0j, 1.5 + 0j)
        self.assertRaises(TypeError, Complex.__floordiv__, 3 + 0j, 0 + 0j)

    def test_richcompare(self):
        self.assertIs(Complex.__eq__(1 + 1j, 1 << 10000), False)
        self.assertIs(Complex.__lt__(1 + 1j, None), NotImplemented)
        self.assertIs(Complex.__eq__(1 + 1j, 1 + 1j), True)
        self.assertIs(Complex.__eq__(1 + 1j, 2 + 2j), False)
        self.assertIs(Complex.__ne__(1 + 1j, 1 + 1j), False)
        self.assertIs(Complex.__ne__(1 + 1j, 2 + 2j), True)
        for i in range(1, 100):
            f = i / 100.0
            self.assertIs(Complex.__eq__(f + 0j, f), True)
            self.assertIs(Complex.__ne__(f + 0j, f), False)
            self.assertIs(Complex.__eq__(Complex(f, f), f), False)
            self.assertIs(Complex.__ne__(Complex(f, f), f), True)
        self.assertIs(Complex.__lt__(1 + 1j, 2 + 2j), NotImplemented)
        self.assertIs(Complex.__le__(1 + 1j, 2 + 2j), NotImplemented)
        self.assertIs(Complex.__gt__(1 + 1j, 2 + 2j), NotImplemented)
        self.assertIs(Complex.__ge__(1 + 1j, 2 + 2j), NotImplemented)
        self.assertRaises(TypeError, operator.lt, 1 + 1j, 2 + 2j)
        self.assertRaises(TypeError, operator.le, 1 + 1j, 2 + 2j)
        self.assertRaises(TypeError, operator.gt, 1 + 1j, 2 + 2j)
        self.assertRaises(TypeError, operator.ge, 1 + 1j, 2 + 2j)
        self.assertIs(operator.eq(1 + 1j, 1 + 1j), True)
        self.assertIs(operator.eq(1 + 1j, 2 + 2j), False)
        self.assertIs(operator.ne(1 + 1j, 1 + 1j), False)
        self.assertIs(operator.ne(1 + 1j, 2 + 2j), True)

    @pytest.mark.slow
    def test_richcompare_boundaries(self):
        def check(n, deltas, is_equal, imag=0.0):
            for delta in deltas:
                i = n + delta
                z = Complex(i, imag)
                self.assertIs(Complex.__eq__(z, i), is_equal(delta))
                self.assertIs(Complex.__ne__(z, i), not is_equal(delta))

        # For IEEE-754 doubles the following should hold:
        #    x in [2 ** (52 + i), 2 ** (53 + i + 1)] -> x mod 2 ** i == 0
        # where the interval is representable, of course.
        for i in range(1, 10):
            pow = 52 + i
            mult = 2 ** i
            check(2 ** pow, range(1, 101), lambda delta: delta % mult == 0)
            check(2 ** pow, range(1, 101), lambda delta: False, float(i))
        check(2 ** 53, range(-100, 0), lambda delta: True)

    def test_mod(self):
        # % is no longer supported on Complex numbers
        self.assertRaises(TypeError, (1 + 1j).__mod__, 0 + 0j)
        self.assertRaises(TypeError, lambda: (3.33 + 4.43j) % 0)
        self.assertRaises(TypeError, (1 + 1j).__mod__, 4.3j)

    def test_divmod(self):
        self.assertRaises(TypeError, divmod, 1 + 1j, 1 + 0j)
        self.assertRaises(TypeError, divmod, 1 + 1j, 0 + 0j)

    def test_pow(self):
        self.assertAlmostEqual(pow(1 + 1j, 0 + 0j), 1.0)
        self.assertAlmostEqual(pow(0 + 0j, 2 + 0j), 0.0)
        self.assertRaises(ZeroDivisionError, pow, 0 + 0j, 1j)
        self.assertAlmostEqual(pow(1j, -1), 1 / 1j)
        self.assertAlmostEqual(pow(1j, 200), 1)
        self.assertRaises(ValueError, pow, 1 + 1j, 1 + 1j, 1 + 1j)

        a = 3.33 + 4.43j
        self.assertEqual(a ** 0j, 1)
        self.assertEqual(a ** 0.0 + 0.0j, 1)

        self.assertEqual(3j ** 0j, 1)
        self.assertEqual(3j ** 0, 1)

        try:
            0j ** a
        except ZeroDivisionError:
            pass
        else:
            self.fail("should fail 0.0 to negative or Complex power")

        try:
            0j ** (3 - 2j)
        except ZeroDivisionError:
            pass
        else:
            self.fail("should fail 0.0 to negative or Complex power")

        # The following is used to exercise certain code paths
        self.assertEqual(a ** 105, a ** 105)
        self.assertEqual(a ** -105, a ** -105)
        self.assertEqual(a ** -30, a ** -30)

        self.assertEqual(0.0j ** 0, 1)

        b = 5.1 + 2.3j
        self.assertRaises(ValueError, pow, a, b, 0)

    def test_boolcontext(self):
        for i in range(100):
            self.assertTrue(Complex(random() + 1e-6, random() + 1e-6))
        self.assertTrue(not Complex(0.0, 0.0))

    def test_conjugate(self):
        self.assertClose(Complex(5.3, 9.8).conjugate(), 5.3 - 9.8j)

    @pytest.mark.slow
    def test_constructor(self):
        class OS:
            def __init__(self, value):
                self.value = value

            def __complex__(self):
                return self.value

        class NS(object):
            def __init__(self, value):
                self.value = value

            def __complex__(self):
                return self.value

        self.assertEqual(Complex(OS(1 + 10j)), 1 + 10j)
        self.assertEqual(Complex(NS(1 + 10j)), 1 + 10j)
        self.assertRaises(TypeError, Complex, OS(None))
        self.assertRaises(TypeError, Complex, NS(None))
        self.assertRaises(TypeError, Complex, {})
        self.assertRaises(TypeError, Complex, NS(1.5))
        self.assertRaises(TypeError, Complex, NS(1))

        self.assertAlmostEqual(Complex("1+10j"), 1 + 10j)
        self.assertAlmostEqual(Complex(10), 10 + 0j)
        self.assertAlmostEqual(Complex(10.0), 10 + 0j)
        self.assertAlmostEqual(Complex(10), 10 + 0j)
        self.assertAlmostEqual(Complex(10 + 0j), 10 + 0j)
        self.assertAlmostEqual(Complex(1, 10), 1 + 10j)
        self.assertAlmostEqual(Complex(1, 10), 1 + 10j)
        self.assertAlmostEqual(Complex(1, 10.0), 1 + 10j)
        self.assertAlmostEqual(Complex(1, 10), 1 + 10j)
        self.assertAlmostEqual(Complex(1, 10), 1 + 10j)
        self.assertAlmostEqual(Complex(1, 10.0), 1 + 10j)
        self.assertAlmostEqual(Complex(1.0, 10), 1 + 10j)
        self.assertAlmostEqual(Complex(1.0, 10), 1 + 10j)
        self.assertAlmostEqual(Complex(1.0, 10.0), 1 + 10j)
        self.assertAlmostEqual(Complex(3.14 + 0j), 3.14 + 0j)
        self.assertAlmostEqual(Complex(3.14), 3.14 + 0j)
        self.assertAlmostEqual(Complex(314), 314.0 + 0j)
        self.assertAlmostEqual(Complex(314), 314.0 + 0j)
        self.assertAlmostEqual(Complex(3.14 + 0j, 0j), 3.14 + 0j)
        self.assertAlmostEqual(Complex(3.14, 0.0), 3.14 + 0j)
        self.assertAlmostEqual(Complex(314, 0), 314.0 + 0j)
        self.assertAlmostEqual(Complex(314, 0), 314.0 + 0j)
        self.assertAlmostEqual(Complex(0j, 3.14j), -3.14 + 0j)
        self.assertAlmostEqual(Complex(0.0, 3.14j), -3.14 + 0j)
        self.assertAlmostEqual(Complex(0j, 3.14), 3.14j)
        self.assertAlmostEqual(Complex(0.0, 3.14), 3.14j)
        self.assertAlmostEqual(Complex("1"), 1 + 0j)
        self.assertAlmostEqual(Complex("1j"), 1j)
        self.assertAlmostEqual(Complex(), 0)
        self.assertAlmostEqual(Complex("-1"), -1)
        self.assertAlmostEqual(Complex("+1"), +1)
        self.assertAlmostEqual(Complex("(1+2j)"), 1 + 2j)
        self.assertAlmostEqual(Complex("(1.3+2.2j)"), 1.3 + 2.2j)
        self.assertAlmostEqual(Complex("3.14+1J"), 3.14 + 1j)
        self.assertAlmostEqual(Complex(" ( +3.14-6J )"), 3.14 - 6j)
        self.assertAlmostEqual(Complex(" ( +3.14-J )"), 3.14 - 1j)
        self.assertAlmostEqual(Complex(" ( +3.14+j )"), 3.14 + 1j)
        self.assertAlmostEqual(Complex("J"), 1j)
        self.assertAlmostEqual(Complex("( j )"), 1j)
        self.assertAlmostEqual(Complex("+J"), 1j)
        self.assertAlmostEqual(Complex("( -j)"), -1j)
        self.assertAlmostEqual(Complex("1e-500"), 0.0 + 0.0j)
        self.assertAlmostEqual(Complex("-1e-500j"), 0.0 - 0.0j)
        self.assertAlmostEqual(Complex("-1e-500+1e-500j"), -0.0 + 0.0j)

        class complex2(complex):
            pass

        self.assertAlmostEqual(Complex(complex2(1 + 1j)), 1 + 1j)
        self.assertAlmostEqual(Complex(real=17, imag=23), 17 + 23j)
        self.assertAlmostEqual(Complex(real=17 + 23j), 17 + 23j)
        self.assertAlmostEqual(Complex(real=17 + 23j, imag=23), 17 + 46j)
        self.assertAlmostEqual(Complex(real=1 + 2j, imag=3 + 4j), -3 + 5j)

        # check that the sign of a zero in the real or imaginary part
        # is preserved when constructing from two floats.  (These checks
        # are harmless on systems without support for signed zeros.)
        def split_zeros(x):
            """Function that produces different results for 0. and -0."""
            return atan2(x, -1.0)

        self.assertEqual(split_zeros(Complex(1.0, 0.0).imag), split_zeros(0.0))
        self.assertEqual(split_zeros(Complex(1.0, -0.0).imag), split_zeros(-0.0))
        self.assertEqual(split_zeros(Complex(0.0, 1.0).real), split_zeros(0.0))
        self.assertEqual(split_zeros(Complex(-0.0, 1.0).real), split_zeros(-0.0))

        # c = 3.14 + 1j
        # self.assertTrue(Complex(c) is c)
        # del c

        self.assertRaises(TypeError, Complex, "1", "1")
        self.assertRaises(TypeError, Complex, 1, "1")

        # SF bug 543840:  Complex(string) accepts strings with \0
        # Fixed in 2.3.
        self.assertRaises(ValueError, Complex, "1+1j\0j")

        self.assertRaises(TypeError, int, 5 + 3j)
        self.assertRaises(TypeError, int, 5 + 3j)
        self.assertRaises(TypeError, float, 5 + 3j)
        self.assertRaises(ValueError, Complex, "")
        # FIXME improve Complex.__new__() to have the same TypeError as complex
        # self.assertRaises(TypeError, Complex, None)
        # self.assertRaisesRegex(TypeError, "not 'NoneType'", Complex, None)
        self.assertRaises(ValueError, Complex, "\0")
        self.assertRaises(ValueError, Complex, "3\09")
        self.assertRaises(TypeError, Complex, "1", "2")
        self.assertRaises(TypeError, Complex, "1", 42)
        self.assertRaises(TypeError, Complex, 1, "2")
        self.assertRaises(ValueError, Complex, "1+")
        self.assertRaises(ValueError, Complex, "1+1j+1j")
        self.assertRaises(ValueError, Complex, "--")
        self.assertRaises(ValueError, Complex, "(1+2j")
        self.assertRaises(ValueError, Complex, "1+2j)")
        self.assertRaises(ValueError, Complex, "1+(2j)")
        self.assertRaises(ValueError, Complex, "(1+2j)123")
        self.assertRaises(ValueError, Complex, "x")
        self.assertRaises(ValueError, Complex, "1j+2")
        self.assertRaises(ValueError, Complex, "1e1ej")
        self.assertRaises(ValueError, Complex, "1e++1ej")
        self.assertRaises(ValueError, Complex, ")1+2j(")
        self.assertRaisesRegex(
            TypeError,
            "first argument must be a string or a number, not 'dict'",
            Complex,
            {1: 2},
            1,
        )
        self.assertRaisesRegex(
            TypeError,
            "second argument must be a number, not 'dict'",
            Complex,
            1,
            {1: 2},
        )
        # the following three are accepted by Python 2.6
        self.assertRaises(ValueError, Complex, "1..1j")
        self.assertRaises(ValueError, Complex, "1.11.1j")
        self.assertRaises(ValueError, Complex, "1e1.1j")

        # check that Complex accepts long unicode strings
        self.assertEqual(type(Complex("1" * 500)), Complex)
        # check whitespace processing
        self.assertEqual(Complex("\N{EM SPACE}(\N{EN SPACE}1+1j ) "), 1 + 1j)
        # Invalid unicode string
        # See bpo-34087
        self.assertRaises(ValueError, Complex, "\u3053\u3093\u306b\u3061\u306f")

        class EvilExc(Exception):
            pass

        class evilcomplex:
            def __complex__(self):
                raise EvilExc

        self.assertRaises(EvilExc, Complex, evilcomplex())

        class float2:
            def __init__(self, value):
                self.value = value

            def __float__(self):
                return self.value

        self.assertAlmostEqual(Complex(float2(42.0)), 42)
        self.assertAlmostEqual(Complex(real=float2(17.0), imag=float2(23.0)), 17 + 23j)
        self.assertRaises(TypeError, Complex, float2(None))

        # FIXME doesn't work
        # class MyIndex:
        #     def __init__(self, value):
        #         self.value = value

        #     def __index__(self):
        #         return self.value

        # self.assertAlmostEqual(Complex(MyIndex(42)), 42.0 + 0.0j)
        # self.assertAlmostEqual(Complex(123, MyIndex(42)), 123.0 + 42.0j)
        # self.assertRaises(OverflowError, Complex, MyIndex(2 ** 2000))
        # self.assertRaises(OverflowError, Complex, 123, MyIndex(2 ** 2000))

        class MyInt:
            def __int__(self):
                return 42

        self.assertRaises(TypeError, Complex, MyInt())
        self.assertRaises(TypeError, Complex, 123, MyInt())

        class complex0(Complex):
            """Test usage of __complex__() when inheriting from 'Complex'"""

            def __complex__(self):
                return 42j

        class complex1(Complex):
            """Test usage of __complex__() with a __new__() method"""

            def __new__(self, value=0j):
                return Complex.__new__(self, 2 * value)

            def __complex__(self):
                return self

        class complex2(Complex):
            """Make sure that __complex__() calls fail if anything other than a
            Complex is returned"""

            def __complex__(self):
                return None

        self.assertEqual(Complex(complex0(1j)), 42j)
        if sys.version_info >= (3, 7):
            # Only deprecated from python 3.7
            with self.assertWarns(DeprecationWarning):
                self.assertEqual(Complex(complex1(1j)), 2j)
        self.assertRaises(TypeError, Complex, complex2(1j))

    # @support.requires_IEEE_754
    def test_constructor_special_numbers(self):
        class complex2(complex):
            pass

        for x in 0.0, -0.0, INF, -INF, NAN:
            for y in 0.0, -0.0, INF, -INF, NAN:
                with self.subTest(x=x, y=y):
                    z = Complex(x, y)
                    self.assertFloatsAreIdentical(z.real, x)
                    self.assertFloatsAreIdentical(z.imag, y)
                    z = complex2(x, y)
                    self.assertIs(type(z), complex2)
                    self.assertFloatsAreIdentical(z.real, x)
                    self.assertFloatsAreIdentical(z.imag, y)
                    z = Complex(complex2(x, y))
                    self.assertIs(type(z), Complex)
                    self.assertFloatsAreIdentical(z.real, x)
                    self.assertFloatsAreIdentical(z.imag, y)
                    z = complex2(Complex(x, y))
                    self.assertIs(type(z), complex2)
                    self.assertFloatsAreIdentical(z.real, x)
                    self.assertFloatsAreIdentical(z.imag, y)

    def test_underscores(self):
        # check underscores
        for lit in VALID_UNDERSCORE_LITERALS:
            if not any(ch in lit for ch in "xXoObB"):
                self.assertEqual(Complex(lit), eval(lit))
                self.assertEqual(Complex(lit), Complex(lit.replace("_", "")))
        for lit in INVALID_UNDERSCORE_LITERALS:
            if lit in ("0_7", "09_99"):  # octals are not recognized here
                continue
            if not any(ch in lit for ch in "xXoObB"):
                self.assertRaises(ValueError, Complex, lit)

    def test_hash(self):
        for x in range(-30, 30):
            self.assertEqual(hash(x), hash(Complex(x, 0)))
            x /= 3.0  # now check against floating point
            self.assertEqual(hash(x), hash(Complex(x, 0.0)))

    def test_abs(self):
        nums = [Complex(x / 3.0, y / 7.0) for x in range(-9, 9) for y in range(-9, 9)]
        for num in nums:
            self.assertAlmostEqual((num.real ** 2 + num.imag ** 2) ** 0.5, abs(num))

    def test_repr_str(self):
        def test(v, expected, test_fn=self.assertEqual):
            test_fn(repr(v), expected)
            test_fn(str(v), expected)

        test(1 + 6j, "(1+6j)")
        test(1 - 6j, "(1-6j)")

        test(-(1 + 0j), "(-1+-0j)", test_fn=self.assertNotEqual)

        test(Complex(1.0, INF), "(1+infj)")
        test(Complex(1.0, -INF), "(1-infj)")
        test(Complex(INF, 1), "(inf+1j)")
        test(Complex(-INF, INF), "(-inf+infj)")
        test(Complex(NAN, 1), "(nan+1j)")
        test(Complex(1, NAN), "(1+nanj)")
        test(Complex(NAN, NAN), "(nan+nanj)")

        test(Complex(0, INF), "infj")
        test(Complex(0, -INF), "-infj")
        test(Complex(0, NAN), "nanj")

        self.assertEqual(1 - 6j, Complex(repr(1 - 6j)))
        self.assertEqual(1 + 6j, Complex(repr(1 + 6j)))
        self.assertEqual(-6j, Complex(repr(-6j)))
        self.assertEqual(6j, Complex(repr(6j)))

    # @support.requires_IEEE_754
    def test_negative_zero_repr_str(self):
        def test(v, expected, test_fn=self.assertEqual):
            test_fn(repr(v), expected)
            test_fn(str(v), expected)

        test(Complex(0.0, 1.0), "1j")
        test(Complex(-0.0, 1.0), "(-0+1j)")
        test(Complex(0.0, -1.0), "-1j")
        test(Complex(-0.0, -1.0), "(-0-1j)")

        test(Complex(0.0, 0.0), "0j")
        test(Complex(0.0, -0.0), "-0j")
        test(Complex(-0.0, 0.0), "(-0+0j)")
        test(Complex(-0.0, -0.0), "(-0-0j)")

    def test_neg(self):
        self.assertEqual(-(1 + 6j), -1 - 6j)

    # FIXME doesn't work
    # def test_file(self):
    #     a = 3.33 + 4.43j
    #     b = 5.1 + 2.3j

    #     fo = None
    #     try:
    #         fo = open(support.TESTFN, "w")
    #         print(a, b, file=fo)
    #         fo.close()
    #         fo = open(support.TESTFN, "r")
    #         self.assertEqual(fo.read(), ("%s %s\n" % (a, b)))
    #     finally:
    #         if (fo is not None) and (not fo.closed):
    #             fo.close()
    #         support.unlink(support.TESTFN)

    def test_getnewargs(self):
        self.assertEqual((1 + 2j).__getnewargs__(), (1.0, 2.0))
        self.assertEqual((1 - 2j).__getnewargs__(), (1.0, -2.0))
        self.assertEqual((2j).__getnewargs__(), (0.0, 2.0))
        self.assertEqual((-0j).__getnewargs__(), (0.0, -0.0))
        self.assertEqual(Complex(0, INF).__getnewargs__(), (0.0, INF))
        self.assertEqual(Complex(INF, 0).__getnewargs__(), (INF, 0.0))

    # @support.requires_IEEE_754
    def test_plus_minus_0j(self):
        # test that -0j and 0j literals are not identified
        z1, z2 = 0j, -0j
        self.assertEqual(atan2(z1.imag, -1.0), atan2(0.0, -1.0))
        self.assertEqual(atan2(z2.imag, -1.0), atan2(-0.0, -1.0))

    # @support.requires_IEEE_754
    def test_negated_imaginary_literal(self):
        z0 = -0j
        z1 = -7j
        z2 = -1e1000j
        # Note: In versions of Python < 3.2, a negated imaginary literal
        # accidentally ended up with real part 0.0 instead of -0.0, thanks to a
        # modification during CST -> AST translation (see issue #9011).  That's
        # fixed in Python 3.2.
        self.assertFloatsAreIdentical(z0.real, -0.0)
        self.assertFloatsAreIdentical(z0.imag, -0.0)
        self.assertFloatsAreIdentical(z1.real, -0.0)
        self.assertFloatsAreIdentical(z1.imag, -7.0)
        self.assertFloatsAreIdentical(z2.real, -0.0)
        self.assertFloatsAreIdentical(z2.imag, -INF)

    # @support.requires_IEEE_754
    def test_overflow(self):
        self.assertEqual(Complex("1e500"), Complex(INF, 0.0))
        self.assertEqual(Complex("-1e500j"), Complex(0.0, -INF))
        self.assertEqual(Complex("-1e500+1.8e308j"), Complex(-INF, INF))

    # @support.requires_IEEE_754
    def test_repr_roundtrip(self):
        vals = [0.0, 1e-500, 1e-315, 1e-200, 0.0123, 3.1415, 1e50, INF, NAN]
        vals += [-v for v in vals]

        # Complex(repr(z)) should recover z exactly, even for Complex
        # numbers involving an infinity, nan, or negative zero
        for x in vals:
            for y in vals:
                z = Complex(x, y)
                roundtrip = Complex(repr(z))
                self.assertFloatsAreIdentical(z.real, roundtrip.real)
                self.assertFloatsAreIdentical(z.imag, roundtrip.imag)

        # if we predefine some constants, then eval(repr(z)) should
        # also work, except that it might change the sign of zeros
        inf, nan = float("inf"), float("nan")
        infj, nanj = Complex(0.0, inf), Complex(0.0, nan)  # noqa: F841
        for x in vals:
            for y in vals:
                z = Complex(x, y)
                roundtrip = eval(repr(z))
                # adding 0.0 has no effect beside changing -0.0 to 0.0
                self.assertFloatsAreIdentical(0.0 + z.real, 0.0 + roundtrip.real)
                self.assertFloatsAreIdentical(0.0 + z.imag, 0.0 + roundtrip.imag)

    def test_format(self):
        # empty format string is same as str()
        self.assertEqual(format(1 + 3j, ""), str(1 + 3j))
        self.assertEqual(format(1.5 + 3.5j, ""), str(1.5 + 3.5j))
        self.assertEqual(format(3j, ""), str(3j))
        self.assertEqual(format(3.2j, ""), str(3.2j))
        self.assertEqual(format(3 + 0j, ""), str(3 + 0j))
        self.assertEqual(format(3.2 + 0j, ""), str(3.2 + 0j))

        # empty presentation type should still be analogous to str,
        # even when format string is nonempty (issue #5920).
        self.assertEqual(format(3.2 + 0j, "-"), str(3.2 + 0j))
        self.assertEqual(format(3.2 + 0j, "<"), str(3.2 + 0j))
        z = 4 / 7.0 - 100j / 7.0
        self.assertEqual(format(z, ""), str(z))
        self.assertEqual(format(z, "-"), str(z))
        self.assertEqual(format(z, "<"), str(z))
        self.assertEqual(format(z, "10"), str(z))
        z = Complex(0.0, 3.0)
        self.assertEqual(format(z, ""), str(z))
        self.assertEqual(format(z, "-"), str(z))
        self.assertEqual(format(z, "<"), str(z))
        self.assertEqual(format(z, "2"), str(z))
        z = Complex(-0.0, 2.0)
        self.assertEqual(format(z, ""), str(z))
        self.assertEqual(format(z, "-"), str(z))
        self.assertEqual(format(z, "<"), str(z))
        self.assertEqual(format(z, "3"), str(z))

        self.assertEqual(format(1 + 3j, "g"), "1+3j")
        self.assertEqual(format(3j, "g"), "0+3j")
        self.assertEqual(format(1.5 + 3.5j, "g"), "1.5+3.5j")

        self.assertEqual(format(1.5 + 3.5j, "+g"), "+1.5+3.5j")
        self.assertEqual(format(1.5 - 3.5j, "+g"), "+1.5-3.5j")
        self.assertEqual(format(1.5 - 3.5j, "-g"), "1.5-3.5j")
        self.assertEqual(format(1.5 + 3.5j, " g"), " 1.5+3.5j")
        self.assertEqual(format(1.5 - 3.5j, " g"), " 1.5-3.5j")
        self.assertEqual(format(-1.5 + 3.5j, " g"), "-1.5+3.5j")
        self.assertEqual(format(-1.5 - 3.5j, " g"), "-1.5-3.5j")

        self.assertEqual(format(-1.5 - 3.5e-20j, "g"), "-1.5-3.5e-20j")
        self.assertEqual(format(-1.5 - 3.5j, "f"), "-1.500000-3.500000j")
        self.assertEqual(format(-1.5 - 3.5j, "F"), "-1.500000-3.500000j")
        self.assertEqual(format(-1.5 - 3.5j, "e"), "-1.500000e+00-3.500000e+00j")
        self.assertEqual(format(-1.5 - 3.5j, ".2e"), "-1.50e+00-3.50e+00j")
        self.assertEqual(format(-1.5 - 3.5j, ".2E"), "-1.50E+00-3.50E+00j")
        self.assertEqual(format(-1.5e10 - 3.5e5j, ".2G"), "-1.5E+10-3.5E+05j")

        self.assertEqual(format(1.5 + 3j, "<20g"), "1.5+3j              ")
        self.assertEqual(format(1.5 + 3j, "*<20g"), "1.5+3j**************")
        self.assertEqual(format(1.5 + 3j, ">20g"), "              1.5+3j")
        self.assertEqual(format(1.5 + 3j, "^20g"), "       1.5+3j       ")
        self.assertEqual(format(1.5 + 3j, "<20"), "(1.5+3j)            ")
        self.assertEqual(format(1.5 + 3j, ">20"), "            (1.5+3j)")
        self.assertEqual(format(1.5 + 3j, "^20"), "      (1.5+3j)      ")
        self.assertEqual(format(1.123 - 3.123j, "^20.2"), "     (1.1-3.1j)     ")

        self.assertEqual(format(1.5 + 3j, "20.2f"), "          1.50+3.00j")
        self.assertEqual(format(1.5 + 3j, ">20.2f"), "          1.50+3.00j")
        self.assertEqual(format(1.5 + 3j, "<20.2f"), "1.50+3.00j          ")
        self.assertEqual(
            format(1.5e20 + 3j, "<20.2f"), "150000000000000000000.00+3.00j"
        )
        self.assertEqual(
            format(1.5e20 + 3j, ">40.2f"), "          150000000000000000000.00+3.00j"
        )
        self.assertEqual(
            format(1.5e20 + 3j, "^40,.2f"), "  150,000,000,000,000,000,000.00+3.00j  "
        )
        self.assertEqual(
            format(1.5e21 + 3j, "^40,.2f"), " 1,500,000,000,000,000,000,000.00+3.00j "
        )
        self.assertEqual(
            format(1.5e21 + 3000j, ",.2f"), "1,500,000,000,000,000,000,000.00+3,000.00j"
        )

        # Issue 7094: Alternate formatting (specified by #)
        self.assertEqual(format(1 + 1j, ".0e"), "1e+00+1e+00j")
        self.assertEqual(format(1 + 1j, "#.0e"), "1.e+00+1.e+00j")
        self.assertEqual(format(1 + 1j, ".0f"), "1+1j")
        self.assertEqual(format(1 + 1j, "#.0f"), "1.+1.j")
        self.assertEqual(format(1.1 + 1.1j, "g"), "1.1+1.1j")
        self.assertEqual(format(1.1 + 1.1j, "#g"), "1.10000+1.10000j")

        # Alternate doesn't make a difference for these, they format the same with or without it
        self.assertEqual(format(1 + 1j, ".1e"), "1.0e+00+1.0e+00j")
        self.assertEqual(format(1 + 1j, "#.1e"), "1.0e+00+1.0e+00j")
        self.assertEqual(format(1 + 1j, ".1f"), "1.0+1.0j")
        self.assertEqual(format(1 + 1j, "#.1f"), "1.0+1.0j")

        # Misc. other alternate tests
        self.assertEqual(format((-1.5 + 0.5j), "#f"), "-1.500000+0.500000j")
        self.assertEqual(format((-1.5 + 0.5j), "#.0f"), "-2.+0.j")
        self.assertEqual(format((-1.5 + 0.5j), "#e"), "-1.500000e+00+5.000000e-01j")
        self.assertEqual(format((-1.5 + 0.5j), "#.0e"), "-2.e+00+5.e-01j")
        self.assertEqual(format((-1.5 + 0.5j), "#g"), "-1.50000+0.500000j")
        self.assertEqual(format((-1.5 + 0.5j), ".0g"), "-2+0.5j")
        self.assertEqual(format((-1.5 + 0.5j), "#.0g"), "-2.+0.5j")

        # zero padding is invalid
        self.assertRaises(ValueError, (1.5 + 0.5j).__format__, "010f")

        # '=' alignment is invalid
        self.assertRaises(ValueError, (1.5 + 3j).__format__, "=20")

        # integer presentation types are an error
        for t in "bcdoxX":
            self.assertRaises(ValueError, (1.5 + 0.5j).__format__, t)

        # make sure everything works in ''.format()
        self.assertEqual(f"*{3.14159 + 2.71828j:.3f}*", "*3.142+2.718j*")

        # issue 3382
        self.assertEqual(format(Complex(NAN, NAN), "f"), "nan+nanj")
        self.assertEqual(format(Complex(1, NAN), "f"), "1.000000+nanj")
        self.assertEqual(format(Complex(NAN, 1), "f"), "nan+1.000000j")
        self.assertEqual(format(Complex(NAN, -1), "f"), "nan-1.000000j")
        self.assertEqual(format(Complex(NAN, NAN), "F"), "NAN+NANj")
        self.assertEqual(format(Complex(1, NAN), "F"), "1.000000+NANj")
        self.assertEqual(format(Complex(NAN, 1), "F"), "NAN+1.000000j")
        self.assertEqual(format(Complex(NAN, -1), "F"), "NAN-1.000000j")
        self.assertEqual(format(Complex(INF, INF), "f"), "inf+infj")
        self.assertEqual(format(Complex(1, INF), "f"), "1.000000+infj")
        self.assertEqual(format(Complex(INF, 1), "f"), "inf+1.000000j")
        self.assertEqual(format(Complex(INF, -1), "f"), "inf-1.000000j")
        self.assertEqual(format(Complex(INF, INF), "F"), "INF+INFj")
        self.assertEqual(format(Complex(1, INF), "F"), "1.000000+INFj")
        self.assertEqual(format(Complex(INF, 1), "F"), "INF+1.000000j")
        self.assertEqual(format(Complex(INF, -1), "F"), "INF-1.000000j")
