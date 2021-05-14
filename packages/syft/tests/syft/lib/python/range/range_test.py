# flake8: noqa
"""
Tests copied from cpython test suite:
https://github.com/python/cpython/blob/3.9/Lib/test/test_range.py
"""

# stdlib
import itertools
import pickle
import sys
import unittest

try:
    # stdlib
    from test.support import ALWAYS_EQ
except ImportError:

    class _ALWAYS_EQ:
        """
        https://github.com/python/cpython/blob/3.9/Lib/test/support/__init__.py
        Object that is equal to anything.
        """

        def __eq__(self, other):
            return True

        def __ne__(self, other):
            return False

    ALWAYS_EQ = _ALWAYS_EQ()


# third party
import pytest

# syft absolute
from syft.lib.python.bool import Bool
from syft.lib.python.int import Int
from syft.lib.python.range import Range


# pure Python implementations (3 args only), for comparison
def pyRange(start, stop, step):
    if (start - stop) // step < 0:
        # replace stop with next element in the sequence of integers
        # that are congruent to start modulo step.
        stop += (start - stop) % step
        while start != stop:
            yield start
            start += step


def pyRange_reversed(start, stop, step):
    stop += (start - stop) % step
    return pyRange(stop - step, start - step, -step)


class RangeTest(unittest.TestCase):
    def assert_iterators_equal(self, xs, ys, test_id, limit=None):
        # check that an iterator xs matches the expected results ys,
        # up to a given limit.
        if limit is not None:
            xs = itertools.islice(xs, limit)
            ys = itertools.islice(ys, limit)
        sentinel = object()
        pairs = itertools.zip_longest(xs, ys, fillvalue=sentinel)
        for i, (x, y) in enumerate(pairs):
            if x == y:
                continue
            elif x == sentinel:
                self.fail(
                    "{}: iterator ended unexpectedly "
                    "at position {}; expected {}".format(test_id, i, y)
                )
            elif y == sentinel:
                self.fail(f"{test_id}: unexpected excess element {x} at position {i}")
            else:
                self.fail(
                    f"{test_id}: wrong element at position {i}; expected {y}, got {x}"
                )

    def test_Range(self):
        self.assertEqual(list(Range(3)), [0, 1, 2])
        self.assertEqual(list(Range(1, 5)), [1, 2, 3, 4])
        self.assertEqual(list(Range(0)), [])
        self.assertEqual(list(Range(-3)), [])
        self.assertEqual(list(Range(1, 10, 3)), [1, 4, 7])
        self.assertEqual(list(Range(5, -5, -3)), [5, 2, -1, -4])

        a = 10
        b = 100
        c = 50

        self.assertEqual(list(Range(a, a + 2)), [a, a + 1])
        self.assertEqual(list(Range(a + 2, a, -1)), [a + 2, a + 1])
        self.assertEqual(list(Range(a + 4, a, -2)), [a + 4, a + 2])

        seq = list(Range(a, b, c))
        self.assertIn(a, seq)
        self.assertNotIn(b, seq)
        self.assertEqual(len(seq), 2)

        seq = list(Range(b, a, -c))
        self.assertIn(b, seq)
        self.assertNotIn(a, seq)
        self.assertEqual(len(seq), 2)

        seq = list(Range(-a, -b, -c))
        self.assertIn(-a, seq)
        self.assertNotIn(-b, seq)
        self.assertEqual(len(seq), 2)

        self.assertRaises(TypeError, Range)
        # TODO : fix
        # self.assertRaises(TypeError, Range(1, 2, 3, 4)
        self.assertRaises(ValueError, Range, 1, 2, 0)

        self.assertRaises(TypeError, Range, 0.0, 2, 1)
        self.assertRaises(TypeError, Range, 1, 2.0, 1)
        self.assertRaises(TypeError, Range, 1, 2, 1.0)
        self.assertRaises(TypeError, Range, 1e100, 1e101, 1e101)

        self.assertRaises(TypeError, Range, 0, "spam")
        self.assertRaises(TypeError, Range, 0, 42, "spam")

        self.assertEqual(len(Range(0, sys.maxsize, sys.maxsize - 1)), 2)

        r = Range(-sys.maxsize, sys.maxsize, 2)
        self.assertEqual(len(r), sys.maxsize)

    @pytest.mark.xfail
    def test_Range_constructor_error_messages(self):
        with self.assertRaisesRegex(
            TypeError, "Range expected at least 1 argument, got 0"
        ):
            Range()

        with self.assertRaisesRegex(
            TypeError, "Range expected at most 3 arguments, got 6"
        ):
            Range(1, 2, 3, 4, 5, 6)

    def test_large_operands(self):
        x = Range(10 ** 20, 10 ** 20 + 10, 3)
        self.assertEqual(len(x), 4)
        self.assertEqual(len(list(x)), 4)

        x = Range(10 ** 20 + 10, 10 ** 20, 3)
        self.assertEqual(len(x), 0)
        self.assertEqual(len(list(x)), 0)
        # self.assertFalse(x)
        assert Bool(False) == x.__bool__()

        x = Range(10 ** 20, 10 ** 20 + 10, -3)
        self.assertEqual(len(x), 0)
        self.assertEqual(len(list(x)), 0)
        # self.assertFalse(x)
        assert Bool(False) == x.__bool__()

        x = Range(10 ** 20 + 10, 10 ** 20, -3)
        self.assertEqual(len(x), 4)
        self.assertEqual(len(list(x)), 4)
        # self.assertTrue(x)
        assert Bool(True) == x.__bool__()

        # Now test Range() with longs
        for x in [Range(-(2 ** 100)), Range(0, -(2 ** 100)), Range(0, 2 ** 100, -1)]:
            self.assertEqual(list(x), [])
            # self.assertFalse(x)
            assert Bool(False) == x.__bool__()

        a = int(10 * sys.maxsize)
        b = int(100 * sys.maxsize)
        c = int(50 * sys.maxsize)

        self.assertEqual(list(Range(a, a + 2)), [a, a + 1])
        self.assertEqual(list(Range(a + 2, a, -1)), [a + 2, a + 1])
        self.assertEqual(list(Range(a + 4, a, -2)), [a + 4, a + 2])

        seq = list(Range(a, b, c))
        self.assertIn(a, seq)
        self.assertNotIn(b, seq)
        self.assertEqual(len(seq), 2)
        self.assertEqual(seq[0], a)
        self.assertEqual(seq[-1], a + c)

        seq = list(Range(b, a, -c))
        self.assertIn(b, seq)
        self.assertNotIn(a, seq)
        self.assertEqual(len(seq), 2)
        self.assertEqual(seq[0], b)
        self.assertEqual(seq[-1], b - c)

        seq = list(Range(-a, -b, -c))
        self.assertIn(-a, seq)
        self.assertNotIn(-b, seq)
        self.assertEqual(len(seq), 2)
        self.assertEqual(seq[0], -a)
        self.assertEqual(seq[-1], -a - c)

    def test_large_Range(self):
        # Check long Ranges (len > sys.maxsize)
        # len() is expected to fail due to limitations of the __len__ protocol
        def _Range_len(x):
            try:
                length = len(x)
            except OverflowError:
                step = x[1] - x[0]
                length = 1 + ((x[-1] - x[0]) // step)
            return length

        a = -sys.maxsize
        b = sys.maxsize
        expected_len = b - a
        x = Range(a, b)
        self.assertIn(a, x)
        self.assertNotIn(b, x)
        self.assertRaises(OverflowError, len, x)
        assert Bool(True) == x.__bool__()
        self.assertEqual(_Range_len(x), expected_len)
        self.assertEqual(x[0], a)
        idx = sys.maxsize + 1
        self.assertEqual(x[idx], a + idx)
        self.assertEqual(x[idx : idx + 1][0], a + idx)  # noqa: E203
        with self.assertRaises(IndexError):
            x[-expected_len - 1]
        with self.assertRaises(IndexError):
            x[expected_len]

        a = 0
        b = 2 * sys.maxsize
        expected_len = b - a
        x = Range(a, b)
        self.assertIn(a, x)
        self.assertNotIn(b, x)
        self.assertRaises(OverflowError, len, x)
        assert Bool(True) == x.__bool__()
        self.assertEqual(_Range_len(x), expected_len)
        self.assertEqual(x[0], a)
        idx = sys.maxsize + 1
        self.assertEqual(x[idx], a + idx)
        self.assertEqual(x[idx : idx + 1][0], a + idx)  # noqa: E203
        with self.assertRaises(IndexError):
            x[-expected_len - 1]
        with self.assertRaises(IndexError):
            x[expected_len]

        a = 0
        b = sys.maxsize ** 10
        c = 2 * sys.maxsize
        expected_len = 1 + (b - a) // c
        x = Range(a, b, c)
        self.assertIn(a, x)
        self.assertNotIn(b, x)
        self.assertRaises(OverflowError, len, x)
        assert Bool(True) == x.__bool__()
        self.assertEqual(_Range_len(x), expected_len)
        self.assertEqual(x[0], a)
        idx = sys.maxsize + 1
        self.assertEqual(x[idx], a + (idx * c))
        self.assertEqual(x[idx : idx + 1][0], a + (idx * c))  # noqa: E203
        with self.assertRaises(IndexError):
            x[-expected_len - 1]
        with self.assertRaises(IndexError):
            x[expected_len]

        a = sys.maxsize ** 10
        b = 0
        c = -2 * sys.maxsize
        expected_len = 1 + (b - a) // c
        x = Range(a, b, c)
        self.assertIn(a, x)
        self.assertNotIn(b, x)
        self.assertRaises(OverflowError, len, x)
        assert Bool(True) == x.__bool__()
        self.assertEqual(_Range_len(x), expected_len)
        self.assertEqual(x[0], a)
        idx = sys.maxsize + 1
        self.assertEqual(x[idx], a + (idx * c))
        self.assertEqual(x[idx : idx + 1][0], a + (idx * c))  # noqa: E203
        with self.assertRaises(IndexError):
            x[-expected_len - 1]
        with self.assertRaises(IndexError):
            x[expected_len]

    def test_invalid_invocation(self):
        self.assertRaises(TypeError, Range)
        # fourth value could be ID
        # self.assertRaises(TypeError, Range, 1, 2, 3, 4)
        self.assertRaises(ValueError, Range, 1, 2, 0)
        a = int(10 * sys.maxsize)
        self.assertRaises(ValueError, Range, a, a + 1, int(0))
        self.assertRaises(TypeError, Range, 1.0, 1.0, 1.0)
        self.assertRaises(TypeError, Range, 1e100, 1e101, 1e101)
        self.assertRaises(TypeError, Range, 0, "spam")
        self.assertRaises(TypeError, Range, 0, 42, "spam")
        # Exercise various combinations of bad arguments, to check
        # refcounting logic
        self.assertRaises(TypeError, Range, 0.0)
        self.assertRaises(TypeError, Range, 0, 0.0)
        self.assertRaises(TypeError, Range, 0.0, 0)
        self.assertRaises(TypeError, Range, 0.0, 0.0)
        self.assertRaises(TypeError, Range, 0, 0, 1.0)
        self.assertRaises(TypeError, Range, 0, 0.0, 1)
        self.assertRaises(TypeError, Range, 0, 0.0, 1.0)
        self.assertRaises(TypeError, Range, 0.0, 0, 1)
        self.assertRaises(TypeError, Range, 0.0, 0, 1.0)
        self.assertRaises(TypeError, Range, 0.0, 0.0, 1)
        self.assertRaises(TypeError, Range, 0.0, 0.0, 1.0)

    def test_index(self):
        u = Range(2)
        self.assertEqual(u.index(0), 0)
        self.assertEqual(u.index(1), 1)
        self.assertRaises(ValueError, u.index, 2)

        u = Range(-2, 3)
        self.assertEqual(u.count(0), 1)
        self.assertEqual(u.index(0), 2)
        self.assertRaises(TypeError, u.index)

        class BadExc(Exception):
            pass

        class BadCmp:
            def __eq__(self, other):
                if other == 2:
                    raise BadExc()
                return False

        a = Range(4)
        self.assertRaises(BadExc, a.index, BadCmp())

        a = Range(-2, 3)
        self.assertEqual(a.index(0), 2)
        self.assertEqual(Range(1, 10, 3).index(4), 1)
        self.assertEqual(Range(1, -10, -3).index(-5), 2)

        self.assertEqual(Range(10 ** 20).index(1), 1)
        self.assertEqual(Range(10 ** 20).index(10 ** 20 - 1), 10 ** 20 - 1)

        self.assertRaises(ValueError, Range(1, 2 ** 100, 2).index, 2 ** 87)
        self.assertEqual(Range(1, 2 ** 100, 2).index(2 ** 87 + 1), 2 ** 86)

        self.assertEqual(Range(10).index(ALWAYS_EQ), 0)

    def test_user_index_method(self):
        bignum = 2 * sys.maxsize
        smallnum = 42

        # User-defined class with an __index__ method
        class I1:
            def __init__(self, n):
                self.n = int(n)

            def __index__(self):
                return self.n

        self.assertEqual(list(Range(I1(bignum), I1(bignum + 1))), [bignum])
        self.assertEqual(list(Range(I1(smallnum), I1(smallnum + 1))), [smallnum])

        # User-defined class with a failing __index__ method
        class IX:
            def __index__(self):
                raise RuntimeError

        self.assertRaises(RuntimeError, Range, IX())

        # User-defined class with an invalid __index__ method
        class IN:
            def __index__(self):
                return "not a number"

        self.assertRaises(TypeError, Range, IN())

        # Test use of user-defined classes in slice indices.
        with pytest.raises(AssertionError):
            # Due to different locations
            self.assertEqual(Range(10)[: I1(5)], Range(5))

        with self.assertRaises(RuntimeError):
            Range(0, 10)[: IX()]

        with self.assertRaises(TypeError):
            Range(0, 10)[: IN()]

    def test_count(self):
        self.assertEqual(Range(3).count(-1), 0)
        self.assertEqual(Range(3).count(0), 1)
        self.assertEqual(Range(3).count(1), 1)
        self.assertEqual(Range(3).count(2), 1)
        self.assertEqual(Range(3).count(3), 0)
        self.assertIs(type(Range(3).count(-1)), Int)
        self.assertIs(type(Range(3).count(1)), Int)
        self.assertEqual(Range(10 ** 20).count(1), 1)
        self.assertEqual(Range(10 ** 20).count(10 ** 20), 0)
        self.assertEqual(Range(3).index(1), 1)
        self.assertEqual(Range(1, 2 ** 100, 2).count(2 ** 87), 0)
        self.assertEqual(Range(1, 2 ** 100, 2).count(2 ** 87 + 1), 1)

        self.assertEqual(Range(10).count(ALWAYS_EQ), 10)

        self.assertEqual(len(Range(sys.maxsize, sys.maxsize + 10)), 10)

    def test_repr(self):
        self.assertEqual(repr(Range(1))[:35], "<syft.lib.python.range.Range object")
        self.assertEqual(repr(Range(1, 2))[:35], "<syft.lib.python.range.Range object")
        self.assertEqual(
            repr(Range(1, 2, 3))[:35], "<syft.lib.python.range.Range object"
        )

    @pytest.mark.xfail
    def test_pickling(self):
        testcases = [
            (13,),
            (0, 11),
            (-22, 10),
            (20, 3, -1),
            (13, 21, 3),
            (-2, 2, 2),
            (2 ** 65, 2 ** 65 + 2),
        ]
        for proto in Range(pickle.HIGHEST_PROTOCOL + 1):
            for t in testcases:
                with self.subTest(proto=proto, test=t):
                    r = Range(*t)
                    self.assertEqual(
                        list(pickle.loads(pickle.dumps(r, proto))), list(r)
                    )

    @pytest.mark.xfail
    def test_iterator_pickling(self):
        testcases = [
            (13,),
            (0, 11),
            (-22, 10),
            (20, 3, -1),
            (13, 21, 3),
            (-2, 2, 2),
            (2 ** 65, 2 ** 65 + 2),
        ]
        for proto in Range(pickle.HIGHEST_PROTOCOL + 1):
            for t in testcases:
                it = itorg = iter(Range(*t))
                data = list(Range(*t))

                d = pickle.dumps(it, proto)
                it = pickle.loads(d)
                self.assertEqual(type(itorg), type(it))
                self.assertEqual(list(it), data)

                it = pickle.loads(d)
                try:
                    next(it)
                except StopIteration:
                    continue
                d = pickle.dumps(it, proto)
                it = pickle.loads(d)
                self.assertEqual(list(it), data[1:])

    @pytest.mark.xfail
    def test_exhausted_iterator_pickling(self):
        for proto in Range(pickle.HIGHEST_PROTOCOL + 1):
            r = Range(2 ** 65, 2 ** 65 + 2)
            i = iter(r)
            while True:
                r = next(i)
                if r == 2 ** 65 + 1:
                    break
            d = pickle.dumps(i, proto)
            i2 = pickle.loads(d)
            self.assertEqual(list(i), [])
            self.assertEqual(list(i2), [])

    @pytest.mark.xfail
    def test_large_exhausted_iterator_pickling(self):
        for proto in Range(pickle.HIGHEST_PROTOCOL + 1):
            r = Range(20)
            i = iter(r)
            while True:
                r = next(i)
                if r == 19:
                    break
            d = pickle.dumps(i, proto)
            i2 = pickle.loads(d)
            self.assertEqual(list(i), [])
            self.assertEqual(list(i2), [])

    def test_odd_bug(self):
        # This used to raise a "SystemError: NULL result without error"
        # because the Range validation step was eating the exception
        # before NULL was returned.
        with self.assertRaises(TypeError):
            Range([], 1, -1)

    def test_types(self):
        # Non-integer objects *equal* to any of the Range's items are supposed
        # to be contained in the Range.
        self.assertIn(1.0, Range(3))
        self.assertIn(True, Range(3))
        self.assertIn(1 + 0j, Range(3))

        self.assertIn(ALWAYS_EQ, Range(3))

        # Objects are never coerced into other types for comparison.
        class C2:
            def __int__(self):
                return 1

            def __index__(self):
                return 1

        self.assertNotIn(C2(), Range(3))
        # ..except if explicitly told so.
        self.assertIn(int(C2()), Range(3))

        # Check that the Range.__contains__ optimization is only
        # used for ints, not for instances of subclasses of int.
        class C3(int):
            def __eq__(self, other):
                return True

        self.assertIn(C3(11), Range(10))
        self.assertIn(C3(11), list(Range(10)))

    def test_strided_limits(self):
        r = Range(0, 101, 2)
        self.assertIn(0, r)
        self.assertNotIn(1, r)
        self.assertIn(2, r)
        self.assertNotIn(99, r)
        self.assertIn(100, r)
        self.assertNotIn(101, r)

        r = Range(0, -20, -1)
        self.assertIn(0, r)
        self.assertIn(-1, r)
        self.assertIn(-19, r)
        self.assertNotIn(-20, r)

        r = Range(0, -20, -2)
        self.assertIn(-18, r)
        self.assertNotIn(-19, r)
        self.assertNotIn(-20, r)

    def test_empty(self):
        r = Range(0)
        self.assertNotIn(0, r)
        self.assertNotIn(1, r)

        r = Range(0, -10)
        self.assertNotIn(0, r)
        self.assertNotIn(-1, r)
        self.assertNotIn(1, r)

    @pytest.mark.xfail
    def test_Range_iterators(self):
        # exercise 'fast' iterators, that use a Rangeiterobject internally.
        # see issue 7298
        limits = [
            base + jiggle
            for M in (2 ** 32, 2 ** 64)
            for base in (-M, -M // 2, 0, M // 2, M)
            for jiggle in (-2, -1, 0, 1, 2)
        ]
        test_Ranges = [
            (start, end, step)
            for start in limits
            for end in limits
            for step in (-(2 ** 63), -(2 ** 31), -2, -1, 1, 2)
        ]

        for start, end, step in test_Ranges:
            iter1 = Range(start, end, step)
            iter2 = pyRange(start, end, step)
            test_id = f"Range({start}, {end}, {step})"
            # check first 100 entries
            self.assert_iterators_equal(iter1, iter2, test_id, limit=100)

            iter1 = reversed(Range(start, end, step))
            iter2 = pyRange_reversed(start, end, step)
            test_id = f"reversed(Range({start}, {end}, {step}))"
            self.assert_iterators_equal(iter1, iter2, test_id, limit=100)

    def test_Range_iterators_invocation(self):
        # verify Range iterators instances cannot be created by
        # calling their type
        Rangeiter_type = type(iter(Range(0)))
        self.assertRaises(TypeError, Rangeiter_type, 1, 3, 1)
        long_Rangeiter_type = type(iter(Range(1 << 1000)))
        self.assertRaises(TypeError, long_Rangeiter_type, 1, 3, 1)

    def test_slice(self):
        def check(start, stop, step=None):
            i = slice(start, stop, step)
            self.assertEqual(list(r[i]), list(r)[i])
            self.assertEqual(len(r[i]), len(list(r)[i]))

        for r in [
            Range(10),
            Range(0),
            Range(1, 9, 3),
            Range(8, 0, -3),
            Range(sys.maxsize + 1, sys.maxsize + 10),
        ]:
            check(0, 2)
            check(0, 20)
            check(1, 2)
            check(20, 30)
            check(-30, -20)
            check(-1, 100, 2)
            check(0, -1)
            check(-1, -3, -1)

    def test_contains(self):
        r = Range(10)
        self.assertIn(0, r)
        self.assertIn(1, r)
        self.assertIn(5.0, r)
        self.assertNotIn(5.1, r)
        self.assertNotIn(-1, r)
        self.assertNotIn(10, r)
        self.assertNotIn("", r)
        r = Range(9, -1, -1)
        self.assertIn(0, r)
        self.assertIn(1, r)
        self.assertIn(5.0, r)
        self.assertNotIn(5.1, r)
        self.assertNotIn(-1, r)
        self.assertNotIn(10, r)
        self.assertNotIn("", r)
        r = Range(0, 10, 2)
        self.assertIn(0, r)
        self.assertNotIn(1, r)
        self.assertNotIn(5.0, r)
        self.assertNotIn(5.1, r)
        self.assertNotIn(-1, r)
        self.assertNotIn(10, r)
        self.assertNotIn("", r)
        r = Range(9, -1, -2)
        self.assertNotIn(0, r)
        self.assertIn(1, r)
        self.assertIn(5.0, r)
        self.assertNotIn(5.1, r)
        self.assertNotIn(-1, r)
        self.assertNotIn(10, r)
        self.assertNotIn("", r)

    def test_reverse_iteration(self):
        for r in [
            Range(10),
            Range(0),
            Range(1, 9, 3),
            Range(8, 0, -3),
            Range(sys.maxsize + 1, sys.maxsize + 10),
        ]:
            self.assertEqual(list(reversed(r)), list(r)[::-1])

    def test_issue11845(self):
        r = Range(*slice(1, 18, 2).indices(20))
        values = {
            None,
            0,
            1,
            -1,
            2,
            -2,
            5,
            -5,
            19,
            -19,
            20,
            -20,
            21,
            -21,
            30,
            -30,
            99,
            -99,
        }
        for i in values:
            for j in values:
                for k in values - {0}:
                    r[i:j:k]

    def test_comparison(self):
        test_Ranges = [
            Range(0),
            Range(0, -1),
            Range(1, 1, 3),
            Range(1),
            Range(5, 6),
            Range(5, 6, 2),
            Range(5, 7, 2),
            Range(2),
            Range(0, 4, 2),
            Range(0, 5, 2),
            Range(0, 6, 2),
        ]
        # test_tuples = List(map(Tuple, test_Ranges))

        # Check that equality of Ranges matches equality of the corresponding
        # tuples for each pair from the test lists above.
        Ranges_eq = [a == b for a in test_Ranges for b in test_Ranges]
        # tuples_eq = [a == b for a in test_tuples for b in test_tuples]
        # self.assertEqual(Ranges_eq, tuples_eq)

        # Check that != correctly gives the logical negation of ==
        Ranges_ne = [a != b for a in test_Ranges for b in test_Ranges]
        self.assertEqual(Ranges_ne, [not x for x in Ranges_eq])

        # Equal Ranges should have equal hashes.
        # for a in test_Ranges:
        #     for b in test_Ranges:
        #         if a == b:
        #             self.assertEqual(hash(a), hash(b))

        # Ranges are unequal to other types (even sequence types)
        self.assertIs(Range(0) == (), False)
        self.assertIs(() == Range(0), False)
        self.assertIs(Range(2) == [0, 1], False)

        # Huge integers aren't a problem.
        # Due to different locations of range objects
        with pytest.raises(AssertionError):
            self.assertEqual(Range(0, 2 ** 100 - 1, 2), Range(0, 2 ** 100, 2))
            self.assertEqual(
                hash(Range(0, 2 ** 100 - 1, 2)), hash(Range(0, 2 ** 100, 2))
            )
            self.assertNotEqual(Range(0, 2 ** 100, 2), Range(0, 2 ** 100 + 1, 2))
            self.assertEqual(
                Range(2 ** 200, 2 ** 201 - 2 ** 99, 2 ** 100),
                Range(2 ** 200, 2 ** 201, 2 ** 100),
            )
            self.assertEqual(
                hash(Range(2 ** 200, 2 ** 201 - 2 ** 99, 2 ** 100)),
                hash(Range(2 ** 200, 2 ** 201, 2 ** 100)),
            )
            self.assertNotEqual(
                Range(2 ** 200, 2 ** 201, 2 ** 100),
                Range(2 ** 200, 2 ** 201 + 1, 2 ** 100),
            )

        # Order comparisons are not implemented for Ranges.
        with self.assertRaises(TypeError):
            Range(0) < Range(0)
        with self.assertRaises(TypeError):
            Range(0) > Range(0)
        with self.assertRaises(TypeError):
            Range(0) <= Range(0)
        with self.assertRaises(TypeError):
            Range(0) >= Range(0)

    def test_attributes(self):
        # test the start, stop and step attributes of Range objects
        self.assert_attrs(Range(0), 0, 0, 1)
        self.assert_attrs(Range(10), 0, 10, 1)
        self.assert_attrs(Range(-10), 0, -10, 1)
        self.assert_attrs(Range(0, 10, 1), 0, 10, 1)
        self.assert_attrs(Range(0, 10, 3), 0, 10, 3)
        self.assert_attrs(Range(10, 0, -1), 10, 0, -1)
        self.assert_attrs(Range(10, 0, -3), 10, 0, -3)
        self.assert_attrs(Range(True), 0, 1, 1)
        self.assert_attrs(Range(False, True), 0, 1, 1)
        self.assert_attrs(Range(False, True, True), 0, 1, 1)

    def assert_attrs(self, Rangeobj, start, stop, step):
        self.assertEqual(Rangeobj.start, start)
        self.assertEqual(Rangeobj.stop, stop)
        self.assertEqual(Rangeobj.step, step)

        # PyPrimitive Range returns syft int and bool
        # self.assertIs(type(Rangeobj.start), int)
        # self.assertIs(type(Rangeobj.stop), int)
        # self.assertIs(type(Rangeobj.step), int)

        with self.assertRaises(AttributeError):
            Rangeobj.start = 0
        with self.assertRaises(AttributeError):
            Rangeobj.stop = 10
        with self.assertRaises(AttributeError):
            Rangeobj.step = 1

        with self.assertRaises(AttributeError):
            del Rangeobj.start
        with self.assertRaises(AttributeError):
            del Rangeobj.stop
        with self.assertRaises(AttributeError):
            del Rangeobj.step


if __name__ == "__main__":
    unittest.main()
