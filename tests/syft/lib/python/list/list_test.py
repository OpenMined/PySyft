# flake8: noqa
"""
Tests copied from cpython test suite:
https://github.com/python/cpython/blob/3.9/Lib/test/list_tests.py"""

# stdlib
from functools import cmp_to_key
import sys
from typing import Any
from typing import List as ListT
import unittest

# syft absolute
from syft.lib.python.list import List


class CommonTest(unittest.TestCase):
    def test_init(self) -> None:
        # Iterable arg is optional
        self.assertEqual(List([]), List())

        # Init clears previous values
        a = List([1, 2, 3])
        a.__init__()
        self.assertEqual(a, List([]))

        # Init overwrites previous values
        a = List([1, 2, 3])
        a.__init__([4, 5, 6])
        self.assertEqual(a, List([4, 5, 6]))

        # Mutables always return a new object
        b = List(a)
        self.assertNotEqual(id(a), id(b))
        self.assertEqual(a, b)

    def test_getitem_error(self) -> None:
        a = List([])
        msg = "list indices must be integers or slices"
        with self.assertRaisesRegex(TypeError, msg):
            a["a"]

    def test_setitem_error(self) -> None:
        a = List([])
        msg = "list indices must be integers or slices"
        with self.assertRaisesRegex(TypeError, msg):
            a["a"] = "python"

    def test_repr(self) -> None:
        l0: ListT[int] = []
        l2: ListT[int] = [0, 1, 2]
        a0 = List(l0)
        a2 = List(l2)

        self.assertEqual(str(a0), str(l0))
        self.assertEqual(repr(a0), repr(l0))
        self.assertEqual(repr(a2), repr(l2))
        self.assertEqual(str(a2), "[0, 1, 2]")
        self.assertEqual(repr(a2), "[0, 1, 2]")

        a2.append(a2)
        a2.append(3)
        self.assertEqual(str(a2), "[0, 1, 2, [...], 3]")
        self.assertEqual(repr(a2), "[0, 1, 2, [...], 3]")

    def test_repr_deep(self) -> None:
        a = List([])
        for i in range(sys.getrecursionlimit() + 100):
            a = List([a])
        self.assertRaises(RecursionError, repr, a)

    def test_set_subscript(self) -> None:
        a = List(range(20))
        self.assertRaises(ValueError, a.__setitem__, slice(0, 10, 0), [1, 2, 3])
        self.assertRaises(TypeError, a.__setitem__, slice(0, 10), 1)
        self.assertRaises(ValueError, a.__setitem__, slice(0, 10, 2), [1, 2])
        self.assertRaises(TypeError, a.__getitem__, "x", 1)
        a[slice(2, 10, 3)] = [1, 2, 3]
        self.assertEqual(
            a,
            List(
                [0, 1, 1, 3, 4, 2, 6, 7, 3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            ),
        )

    def test_reversed(self) -> None:
        a = List(range(20))
        r = reversed(a)
        self.assertEqual(list(r), List(range(19, -1, -1)))
        self.assertRaises(StopIteration, next, r)
        self.assertEqual(list(reversed(List())), List())
        # Bug 3689: make sure list-reversed-iterator doesn't have __len__
        self.assertRaises(TypeError, len, reversed([1, 2, 3]))

    def test_setitem(self) -> None:
        a = List([0, 1])
        a[0] = 0
        a[1] = 100
        self.assertEqual(a, List([0, 100]))
        a[-1] = 200
        self.assertEqual(a, List([0, 200]))
        a[-2] = 100
        self.assertEqual(a, List([100, 200]))
        self.assertRaises(IndexError, a.__setitem__, -3, 200)
        self.assertRaises(IndexError, a.__setitem__, 2, 200)

        a = List([])
        self.assertRaises(IndexError, a.__setitem__, 0, 200)
        self.assertRaises(IndexError, a.__setitem__, -1, 200)
        self.assertRaises(TypeError, a.__setitem__)

        a = List([0, 1, 2, 3, 4])
        a[0] = 1
        a[1] = 2
        a[2] = 3
        self.assertEqual(a, List([1, 2, 3, 3, 4]))
        a[0] = 5
        a[1] = 6
        a[2] = 7
        self.assertEqual(a, List([5, 6, 7, 3, 4]))
        a[-2] = 88
        a[-1] = 99
        self.assertEqual(a, List([5, 6, 7, 88, 99]))
        a[-2] = 8
        a[-1] = 9
        self.assertEqual(a, List([5, 6, 7, 8, 9]))

        msg = "list indices must be integers or slices"
        with self.assertRaisesRegex(TypeError, msg):
            a["a"] = "python"

    def test_delitem(self) -> None:
        a = List([0, 1])
        del a[1]
        self.assertEqual(a, [0])
        del a[0]
        self.assertEqual(a, [])

        a = List([0, 1])
        del a[-2]
        self.assertEqual(a, [1])
        del a[-1]
        self.assertEqual(a, [])

        a = List([0, 1])
        self.assertRaises(IndexError, a.__delitem__, -3)
        self.assertRaises(IndexError, a.__delitem__, 2)

        a = List([])
        self.assertRaises(IndexError, a.__delitem__, 0)

        self.assertRaises(TypeError, a.__delitem__)

    def test_setslice(self) -> None:
        l = [0, 1]  # noqa: E741
        a = List(l)

        for i in range(-3, 4):
            a[:i] = l[:i]
            self.assertEqual(a, l)
            a2 = a[:]
            a2[:i] = a[:i]
            self.assertEqual(a2, a)
            a[i:] = l[i:]
            self.assertEqual(a, l)
            a2 = a[:]
            a2[i:] = a[i:]
            self.assertEqual(a2, a)
            for j in range(-3, 4):
                a[i:j] = l[i:j]
                self.assertEqual(a, l)
                a2 = a[:]
                a2[i:j] = a[i:j]
                self.assertEqual(a2, a)

        aa2 = a2[:]
        aa2[:0] = [-2, -1]
        self.assertEqual(aa2, [-2, -1, 0, 1])
        aa2[0:] = []
        self.assertEqual(aa2, [])

        a = List([1, 2, 3, 4, 5])
        a[:-1] = a
        self.assertEqual(a, List([1, 2, 3, 4, 5, 5]))
        a = List([1, 2, 3, 4, 5])
        a[1:] = a
        self.assertEqual(a, List([1, 1, 2, 3, 4, 5]))
        a = List([1, 2, 3, 4, 5])
        a[1:-1] = a
        self.assertEqual(a, List([1, 1, 2, 3, 4, 5, 5]))

        a = List([])
        a[:] = tuple(range(10))
        self.assertEqual(a, List(range(10)))

        self.assertRaises(TypeError, a.__setitem__, slice(0, 1, 5))

        self.assertRaises(TypeError, a.__setitem__)

    def test_delslice(self) -> None:
        a = List([0, 1])
        del a[1:2]
        del a[0:1]
        self.assertEqual(a, List([]))

        a = List([0, 1])
        del a[1:2]
        del a[0:1]
        self.assertEqual(a, List([]))

        a = List([0, 1])
        del a[-2:-1]
        self.assertEqual(a, List([1]))

        a = List([0, 1])
        del a[-2:-1]
        self.assertEqual(a, List([1]))

        a = List([0, 1])
        del a[1:]
        del a[:1]
        self.assertEqual(a, List([]))

        a = List([0, 1])
        del a[1:]
        del a[:1]
        self.assertEqual(a, List([]))

        a = List([0, 1])
        del a[-1:]
        self.assertEqual(a, List([0]))

        a = List([0, 1])
        del a[-1:]
        self.assertEqual(a, List([0]))

        a = List([0, 1])
        del a[:]
        self.assertEqual(a, List([]))

    def test_append(self) -> None:
        a = List([])
        a.append(0)
        a.append(1)
        a.append(2)
        self.assertEqual(a, List([0, 1, 2]))

        self.assertRaises(TypeError, a.append)

    def test_extend(self) -> None:
        a1 = List([0])
        a2 = List((0, 1))
        a = a1[:]
        a.extend(a2)
        self.assertEqual(a, a1 + a2)

        a.extend(List([]))
        self.assertEqual(a, a1 + a2)

        a.extend(a)
        self.assertEqual(a, List([0, 0, 1, 0, 0, 1]))

        a = List("spam")
        a.extend("eggs")
        self.assertEqual(a, list("spameggs"))

        self.assertRaises(TypeError, a.extend, None)
        self.assertRaises(TypeError, a.extend)

        # overflow test. issue1621
        class CustomIter:
            def __iter__(self) -> "CustomIter":
                return self

            def __next__(self) -> Any:
                raise StopIteration

            def __length_hint__(self) -> int:
                return sys.maxsize

        a = List([1, 2, 3, 4])
        a.extend(CustomIter())
        self.assertEqual(a, [1, 2, 3, 4])

    def test_insert(self) -> None:
        a = List([0, 1, 2])
        a.insert(0, -2)
        a.insert(1, -1)
        a.insert(2, 0)
        self.assertEqual(a, [-2, -1, 0, 0, 1, 2])

        b = a[:]
        b.insert(-2, "foo")
        b.insert(-200, "left")
        b.insert(200, "right")
        self.assertEqual(b, List(["left", -2, -1, 0, 0, "foo", 1, 2, "right"]))

        self.assertRaises(TypeError, a.insert)

    def test_pop(self) -> None:
        a = List([-1, 0, 1])
        a.pop()
        self.assertEqual(a, [-1, 0])
        a.pop(0)
        self.assertEqual(a, [0])
        self.assertRaises(IndexError, a.pop, 5)
        a.pop(0)
        self.assertEqual(a, [])
        self.assertRaises(IndexError, a.pop)
        self.assertRaises(TypeError, a.pop, 42, 42)
        a = List([0, 10, 20, 30, 40])

    def test_remove(self) -> None:
        a = List([0, 0, 1])
        a.remove(1)
        self.assertEqual(a, [0, 0])
        a.remove(0)
        self.assertEqual(a, [0])
        a.remove(0)
        self.assertEqual(a, [])

        self.assertRaises(ValueError, a.remove, 0)

        self.assertRaises(TypeError, a.remove)

        class BadExc(Exception):
            pass

        class BadCmp:
            def __eq__(self, other: Any) -> bool:
                if other == 2:
                    raise BadExc()
                return False

        a = List([0, 1, 2, 3])
        self.assertRaises(BadExc, a.remove, BadCmp())

        class BadCmp2:
            def __eq__(self, other: Any) -> bool:
                raise BadExc()

        d = List("abcdefghcij")
        d.remove("c")
        self.assertEqual(d, List("abdefghcij"))
        d.remove("c")
        self.assertEqual(d, List("abdefghij"))
        self.assertRaises(ValueError, d.remove, "c")
        self.assertEqual(d, List("abdefghij"))

        # Handle comparison errors
        d = List(["a", "b", BadCmp2(), "c"])
        e = List(d)
        self.assertRaises(BadExc, d.remove, "c")
        for x, y in zip(d, e):
            # verify that original order and values are retained.
            # we upcast to remove the randomly generated IDs
            if hasattr(x, "upcast") and hasattr(y, "upcast"):
                self.assertIs(x.upcast(), y.upcast())
            else:
                self.assertIs(x, y)

    def test_index(self) -> None:
        u = List([0, 1])
        self.assertEqual(u.index(0), 0)
        self.assertEqual(u.index(1), 1)
        self.assertRaises(ValueError, u.index, 2)

        u = List([-2, -1, 0, 0, 1, 2])
        self.assertEqual(u.count(0), 2)
        self.assertEqual(u.index(0), 2)
        self.assertEqual(u.index(0, 2), 2)
        self.assertEqual(u.index(-2, -10), 0)
        self.assertEqual(u.index(0, 3), 3)
        self.assertEqual(u.index(0, 3, 4), 3)
        self.assertRaises(ValueError, u.index, 2, 0, -10)

        self.assertRaises(TypeError, u.index)

        class BadExc(Exception):
            pass

        class BadCmp:
            def __eq__(self, other: Any) -> bool:
                if other == 2:
                    raise BadExc()
                return False

        a = List([0, 1, 2, 3])
        self.assertRaises(BadExc, a.index, BadCmp())

        a = List([-2, -1, 0, 0, 1, 2])
        self.assertEqual(a.index(0), 2)
        self.assertEqual(a.index(0, 2), 2)
        self.assertEqual(a.index(0, -4), 2)
        self.assertEqual(a.index(-2, -10), 0)
        self.assertEqual(a.index(0, 3), 3)
        self.assertEqual(a.index(0, -3), 3)
        self.assertEqual(a.index(0, 3, 4), 3)
        self.assertEqual(a.index(0, -3, -2), 3)
        self.assertEqual(a.index(0, -4 * sys.maxsize, 4 * sys.maxsize), 2)
        self.assertRaises(ValueError, a.index, 0, 4 * sys.maxsize, -4 * sys.maxsize)
        self.assertRaises(ValueError, a.index, 2, 0, -10)

        a = List([-2, -1, 0, 0, 1, 2])
        a.remove(0)
        self.assertRaises(ValueError, a.index, 2, 0, 4)
        self.assertEqual(a, List([-2, -1, 0, 1, 2]))

        # Test modifying the list during index's iteration
        class EvilCmp:
            def __init__(self, victim: Any) -> None:
                self.victim = victim

            def __eq__(self, other: Any) -> bool:
                del self.victim[:]
                return False

        a = List()
        a[:] = [EvilCmp(a) for _ in range(100)]
        # This used to seg fault before patch #1005778
        self.assertRaises(ValueError, a.index, None)

    def test_reverse(self) -> None:
        u = List([-2, -1, 0, 1, 2])
        u2 = u[:]
        u.reverse()
        self.assertEqual(u, [2, 1, 0, -1, -2])
        u.reverse()
        self.assertEqual(u, u2)

        self.assertRaises(TypeError, u.reverse, 42)

    def test_clear(self) -> None:
        u = List([2, 3, 4])
        u.clear()
        self.assertEqual(u, [])

        u = List([])
        u.clear()
        self.assertEqual(u, [])

        u = List([])
        u.append(1)
        u.clear()
        u.append(2)
        self.assertEqual(u, [2])

        self.assertRaises(TypeError, u.clear, None)

    def test_copy(self) -> None:
        u = List([1, 2, 3])
        v = u.copy()
        self.assertEqual(v, [1, 2, 3])

        u = List([])
        v = u.copy()
        self.assertEqual(v, [])

        # test that it's indeed a copy and not a reference
        u = List(["a", "b"])
        v = u.copy()
        v.append("i")
        self.assertEqual(u, ["a", "b"])
        self.assertEqual(v, u + ["i"])

        # test that it's a shallow, not a deep copy
        u = List([1, 2, [3, 4], 5])
        v = u.copy()
        self.assertEqual(u, v)

        # we upcast to remove the randomly generated IDs
        if hasattr(v[3], "upcast") and hasattr(u[3], "upcast"):
            self.assertIs(v[3].upcast(), u[3].upcast())
        else:
            self.assertIs(v[3], u[3])

        self.assertRaises(TypeError, u.copy, None)

    def test_sort(self) -> None:
        u = List([1, 0])
        u.sort()
        self.assertEqual(u, [0, 1])

        u = List([2, 1, 0, -1, -2])
        u.sort()
        self.assertEqual(u, List([-2, -1, 0, 1, 2]))

        self.assertRaises(TypeError, u.sort, 42, 42)

        def revcmp(a: Any, b: Any) -> int:
            if a == b:
                return 0
            elif a < b:
                return 1
            else:  # a > b
                return -1

        u.sort(key=cmp_to_key(revcmp))
        self.assertEqual(u, List([2, 1, 0, -1, -2]))

        # The following dumps core in unpatched Python 1.5:
        def myComparison(x: Any, y: Any) -> int:
            xmod, ymod = x % 3, y % 7
            if xmod == ymod:
                return 0
            elif xmod < ymod:
                return -1
            else:  # xmod > ymod
                return 1

        z = List(range(12))
        z.sort(key=cmp_to_key(myComparison))

        self.assertRaises(TypeError, z.sort, 2)

        def selfmodifyingComparison(x: Any, y: Any) -> int:
            z.append(1)
            if x == y:
                return 0
            elif x < y:
                return -1
            else:  # x > y
                return 1

        self.assertRaises(ValueError, z.sort, key=cmp_to_key(selfmodifyingComparison))

        self.assertRaises(TypeError, z.sort, 42, 42, 42, 42)

    def test_slice(self) -> None:
        u = List("spam")
        u[:2] = "h"
        self.assertEqual(u, list("ham"))

    def test_iadd(self) -> None:
        u = List([0, 1])
        u += List()
        self.assertEqual(u, List([0, 1]))
        u += List([2, 3])
        self.assertEqual(u, List([0, 1, 2, 3]))
        u += List([4, 5])
        self.assertEqual(u, List([0, 1, 2, 3, 4, 5]))

        u = List("spam")
        u += List("eggs")
        self.assertEqual(u, List("spameggs"))
        u = List([0, 1])

        u2 = u
        u += [2, 3]
        self.assertIs(u, u2)

        u = List("spam")
        u += "eggs"
        self.assertEqual(u, List("spameggs"))

        self.assertRaises(TypeError, u.__iadd__, None)

    def test_imul(self) -> None:
        u = List([0, 1])
        u *= 3
        self.assertEqual(u, List([0, 1, 0, 1, 0, 1]))
        u *= 0
        self.assertEqual(u, List([]))
        s = List([])

        oldid = id(s)
        s *= 10
        self.assertEqual(id(s), oldid)

    def test_extendedslicing(self) -> None:
        #  subscript
        a = List([0, 1, 2, 3, 4])

        #  deletion
        del a[::2]
        self.assertEqual(a, List([1, 3]))
        a = List(range(5))
        del a[1::2]
        self.assertEqual(a, List([0, 2, 4]))
        a = List(range(5))
        del a[1::-2]
        self.assertEqual(a, List([0, 2, 3, 4]))
        a = List(range(10))
        del a[::1000]
        self.assertEqual(a, List([1, 2, 3, 4, 5, 6, 7, 8, 9]))
        #  assignment
        a = List(range(10))
        a[::2] = [-1] * 5
        self.assertEqual(a, List([-1, 1, -1, 3, -1, 5, -1, 7, -1, 9]))
        a = List(range(10))
        a[::-4] = [10] * 3
        self.assertEqual(a, List([0, 10, 2, 3, 4, 10, 6, 7, 8, 10]))
        a = List(range(4))
        a[::-1] = a
        self.assertEqual(a, List([3, 2, 1, 0]))
        a = List(range(10))
        b = a[:]
        c = a[:]
        a[2:3] = List(["two", "elements"])
        b[slice(2, 3)] = List(["two", "elements"])
        c[2:3:] = List(["two", "elements"])
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        a = List(range(10))
        a[::2] = tuple(range(5))
        self.assertEqual(a, List([0, 1, 1, 3, 2, 5, 3, 7, 4, 9]))
        # test issue7788
        a = List(range(10))
        del a[9 :: 1 << 333]  # noqa: E203

    def test_constructor_exception_handling(self) -> None:
        # Bug #1242657
        class F(object):
            def __iter__(self) -> "F":
                raise KeyboardInterrupt

        self.assertRaises(KeyboardInterrupt, list, F())

    def test_exhausted_iterator(self) -> None:
        a = List([1, 2, 3])
        exhit = iter(a)
        empit = iter(a)
        for x in exhit:  # exhaust the iterator
            next(empit)  # not exhausted
        a.append(9)
        self.assertEqual(list(exhit), [])
        self.assertEqual(list(empit), [9])
        self.assertEqual(a, List([1, 2, 3, 9]))
