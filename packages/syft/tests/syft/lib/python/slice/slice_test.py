# stdlib
import itertools
import operator
import sys
from test import support
import unittest
import weakref

# third party
import pytest

# syft absolute
from syft.lib.python.list import List
from syft.lib.python.slice import Slice
from syft.lib.python.string import String
from syft.lib.python.tuple import Tuple


def evaluate_slice_index(arg):
    """
    Helper function to convert a slice argument to an integer, and raise
    TypeError with a suitable message on failure.

    """
    if hasattr(arg, "__index__"):
        return operator.index(arg)
    else:
        raise TypeError(
            "slice indices must be integers or " "None or have an __index__ method"
        )


def slice_indices(slice, length):
    """
    Reference implementation for the slice.indices method.

    """
    # Compute step and length as integers.
    length = operator.index(length)
    step = 1 if slice.step is None else evaluate_slice_index(slice.step)

    # Raise ValueError for negative length or zero step.
    if length < 0:
        raise ValueError("length should not be negative")
    if step == 0:
        raise ValueError("slice step cannot be zero")

    # Find lower and upper bounds for start and stop.
    lower = -1 if step < 0 else 0
    upper = length - 1 if step < 0 else length

    # Compute start.
    if slice.start is None:
        start = upper if step < 0 else lower
    else:
        start = evaluate_slice_index(slice.start)
        start = max(start + length, lower) if start < 0 else min(start, upper)

    # Compute stop.
    if slice.stop is None:
        stop = lower if step < 0 else upper
    else:
        stop = evaluate_slice_index(slice.stop)
        stop = max(stop + length, lower) if stop < 0 else min(stop, upper)

    return start, stop, step


# Class providing an __index__ method.  Used for testing slice.indices.


class MyIndexable(object):
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


class SliceTest(unittest.TestCase):
    @pytest.mark.xfail
    def test_constructor(self):
        self.assertRaises(TypeError, Slice)
        self.assertRaises(TypeError, Slice, 1, 2, 3, 4)

    def test_repr(self):
        self.assertEqual(repr(Slice(1, 2, 3))[:33], "<syft.lib.python.Slice object at ")

    def test_hash(self):
        # Verify clearing of SF bug #800796
        self.assertRaises(TypeError, hash, Slice(5))
        with self.assertRaises(TypeError):
            slice(5).__hash__()

    def test_cmp(self):
        s1 = Slice(1, 2, 3)
        s2 = Slice(1, 2, 3)
        s3 = Slice(1, 2, 4)
        self.assertEqual(s1.value, s2.value)
        self.assertNotEqual(s1.value, s3.value)
        self.assertNotEqual(s1, None)
        self.assertNotEqual(s1.value, (1, 2, 3))
        self.assertNotEqual(s1, "")

        class Exc(Exception):
            pass

        class BadCmp(object):
            def __eq__(self, other):
                raise Exc

        s1 = Slice(BadCmp())
        s2 = Slice(BadCmp())
        self.assertEqual(s1, s1)
        self.assertRaises(Exc, lambda: s1.value == s2.value)

        s1 = Slice(1, BadCmp())
        s2 = Slice(1, BadCmp())
        self.assertEqual(s1, s1)
        self.assertRaises(Exc, lambda: s1.value == s2.value)

        s1 = Slice(1, 2, BadCmp())
        s2 = Slice(1, 2, BadCmp())
        self.assertEqual(s1, s1)
        self.assertRaises(Exc, lambda: s1.value == s2.value)

    def test_members(self):
        s = Slice(1)
        self.assertEqual(s.start, None)
        self.assertEqual(s.stop, 1)
        self.assertEqual(s.step, None)

        s = Slice(1, 2)
        self.assertEqual(s.start, 1)
        self.assertEqual(s.stop, 2)
        self.assertEqual(s.step, None)

        s = Slice(1, 2, 3)
        self.assertEqual(s.start, 1)
        self.assertEqual(s.stop, 2)
        self.assertEqual(s.step, 3)

        class AnyClass:
            pass

        obj = AnyClass()
        s = Slice(obj)
        self.assertTrue(s.stop is obj)

    def check_indices(self, slice, length):
        try:
            actual = slice.indices(length)
        except ValueError:
            actual = "valueerror"
        try:
            expected = slice_indices(slice, length)
        except ValueError:
            expected = "valueerror"
        self.assertEqual(actual, expected)

        if length >= 0 and slice.step != 0:
            actual = range(*slice.indices(length))
            expected = range(length)[slice.value]
            self.assertEqual(actual, expected)

    @pytest.mark.slow
    def test_indices(self):
        self.assertEqual(Slice(None).indices(10), (0, 10, 1))
        self.assertEqual(Slice(None, None, 2).indices(10), (0, 10, 2))
        self.assertEqual(Slice(1, None, 2).indices(10), (1, 10, 2))
        self.assertEqual(Slice(None, None, -1).indices(10), (9, -1, -1))
        self.assertEqual(Slice(None, None, -2).indices(10), (9, -1, -2))
        self.assertEqual(Slice(3, None, -2).indices(10), (3, -1, -2))
        # issue 3004 tests
        self.assertEqual(Slice(None, -9).indices(10), (0, 1, 1))
        self.assertEqual(Slice(None, -10).indices(10), (0, 0, 1))
        self.assertEqual(Slice(None, -11).indices(10), (0, 0, 1))
        self.assertEqual(Slice(None, -10, -1).indices(10), (9, 0, -1))
        self.assertEqual(Slice(None, -11, -1).indices(10), (9, -1, -1))
        self.assertEqual(Slice(None, -12, -1).indices(10), (9, -1, -1))
        self.assertEqual(Slice(None, 9).indices(10), (0, 9, 1))
        self.assertEqual(Slice(None, 10).indices(10), (0, 10, 1))
        self.assertEqual(Slice(None, 11).indices(10), (0, 10, 1))
        self.assertEqual(Slice(None, 8, -1).indices(10), (9, 8, -1))
        self.assertEqual(Slice(None, 9, -1).indices(10), (9, 9, -1))
        self.assertEqual(Slice(None, 10, -1).indices(10), (9, 9, -1))

        self.assertEqual(Slice(-100, 100).indices(10), Slice(None).indices(10))
        self.assertEqual(
            Slice(100, -100, -1).indices(10), Slice(None, None, -1).indices(10)
        )
        self.assertEqual(Slice(-100, 100, 2).indices(10), (0, 10, 2))

        self.assertEqual(list(range(10))[:: sys.maxsize - 1], [0])

        # Check a variety of start, stop, step and length values, including
        # values exceeding sys.maxsize (see issue #14794).
        vals = [
            None,
            -(2 ** 100),
            -(2 ** 30),
            -53,
            -7,
            -1,
            0,
            1,
            7,
            53,
            2 ** 30,
            2 ** 100,
        ]
        lengths = [0, 1, 7, 53, 2 ** 30, 2 ** 100]
        for slice_args in itertools.product(vals, repeat=3):
            s = Slice(*slice_args)
            for length in lengths:
                self.check_indices(s, length)
        self.check_indices(Slice(0, 10, 1), -3)

        # Negative length should raise ValueError
        with self.assertRaises(ValueError):
            Slice(None).indices(-1)

        # Zero step should raise ValueError
        with self.assertRaises(ValueError):
            Slice(0, 10, 0).indices(5)

        # Using a start, stop or step or length that can't be interpreted as an
        # integer should give a TypeError ...
        with self.assertRaises(TypeError):
            Slice(0.0, 10, 1).indices(5)
        with self.assertRaises(TypeError):
            Slice(0, 10.0, 1).indices(5)
        with self.assertRaises(TypeError):
            Slice(0, 10, 1.0).indices(5)
        with self.assertRaises(TypeError):
            Slice(0, 10, 1).indices(5.0)

        # ... but it should be fine to use a custom class that provides index.
        self.assertEqual(Slice(0, 10, 1).indices(5), (0, 5, 1))
        self.assertEqual(Slice(MyIndexable(0), 10, 1).indices(5), (0, 5, 1))
        self.assertEqual(Slice(0, MyIndexable(10), 1).indices(5), (0, 5, 1))
        self.assertEqual(Slice(0, 10, MyIndexable(1)).indices(5), (0, 5, 1))
        self.assertEqual(Slice(0, 10, 1).indices(MyIndexable(5)), (0, 5, 1))

    def test_setslice_without_getslice(self):
        tmp = []

        class X(object):
            def __setitem__(self, i, k):
                tmp.append((i, k))

        x = X()
        x[1:2] = 42
        self.assertEqual(tmp, [(Slice(1, 2), 42)])

    def test_cycle(self):
        class myobj:
            pass

        o = myobj()
        o.s = Slice(o)
        w = weakref.ref(o)
        o = None
        support.gc_collect()
        self.assertIsNone(w())

    def test_slice_types(self) -> None:
        py_string = "Python"
        py_list = ["P", "y", "t", "h", "o", "n"]
        py_tuple = ("P", "y", "t", "h", "o", "n")

        sy_string = String(py_string)
        sy_tuple = Tuple(py_tuple)
        sy_list = List(py_list)

        py_slice1 = slice(1)
        sy_slice1 = Slice(1)

        assert py_slice1.start == sy_slice1.start
        assert py_slice1.stop == sy_slice1.stop
        assert py_slice1.step == sy_slice1.step

        assert py_slice1 == sy_slice1

        py_slice2 = slice(1, 2)
        sy_slice2 = Slice(1, 2)

        assert py_slice2 == sy_slice2

        assert py_slice2.start == sy_slice2.start
        assert py_slice2.stop == sy_slice2.stop
        assert py_slice2.step == sy_slice2.step

        py_slice3 = slice(1, 2, -1)
        sy_slice3 = Slice(1, 2, -1)

        assert py_slice3 == sy_slice3

        assert py_slice3.start == sy_slice3.start
        assert py_slice3.stop == sy_slice3.stop
        assert py_slice3.step == sy_slice3.step

        assert sy_string[sy_slice1] == py_string[py_slice1]
        assert sy_string[sy_slice2] == py_string[py_slice2]
        assert sy_string[sy_slice3] == py_string[py_slice3]

        assert sy_tuple[sy_slice1] == py_tuple[py_slice1]
        assert sy_tuple[sy_slice2] == py_tuple[py_slice2]
        assert sy_tuple[sy_slice3] == py_tuple[py_slice3]

        assert sy_list[sy_slice1] == py_list[py_slice1]
        assert sy_list[sy_slice2] == py_list[py_slice2]
        assert sy_list[sy_slice3] == py_list[py_slice3]

        assert sy_list[py_slice3] == py_list[py_slice3]
