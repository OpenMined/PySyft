# flake8: noqa
"""
Tests copied from cpython test suite:
https://github.com/python/cpython/blob/3.9/Lib/test/test_dict.py
"""

# stdlib
import collections
import collections.abc
import gc
import pickle
import random
import string
import sys
from test import support
import unittest
import weakref

# third party
import pytest

# syft absolute
from syft.lib.python.dict import Dict
from syft.lib.python.none import SyNone
from syft.lib.python.string import String

# import weakref


class DictTest(unittest.TestCase):
    def test_invalid_keyword_arguments(self):
        class Custom(dict):
            pass

        for invalid in {1: 2}, Custom({1: 2}):
            with self.assertRaises(TypeError):
                dict(**invalid)
            with self.assertRaises(TypeError):
                {}.update(**invalid)

    def test_constructor(self):
        # calling built-in types without argument must return empty
        self.assertEqual(Dict(), {})
        self.assertIsNot(Dict(), {})

    @pytest.mark.slow
    def test_literal_constructor(self):
        # check literal constructor for different sized dicts
        # (to exercise the BUILD_MAP oparg).
        for n in (0, 1, 6, 256, 400):
            items = [
                ("".join(random.sample(string.ascii_letters, 8)), i) for i in range(n)
            ]
            random.shuffle(items)
            formatted_items = (f"{k!r}: {v:d}" for k, v in items)
            dictliteral = "{" + ", ".join(formatted_items) + "}"
            self.assertEqual(eval(dictliteral), dict(items))

    def test_merge_operator(self):

        a = Dict({0: 0, 1: 1, 2: 1})
        b = Dict({1: 1, 2: 2, 3: 3})

        if sys.version_info >= (3, 9):
            c = a.copy()
            c |= b

            self.assertEqual(a | b, Dict({0: 0, 1: 1, 2: 2, 3: 3}))
            self.assertEqual(c, Dict({0: 0, 1: 1, 2: 2, 3: 3}))

            c = b.copy()
            c |= a

            self.assertEqual(b | a, Dict({1: 1, 2: 1, 3: 3, 0: 0}))
            self.assertEqual(c, Dict({1: 1, 2: 1, 3: 3, 0: 0}))

            c = a.copy()
            c |= [(1, 1), (2, 2), (3, 3)]

            self.assertEqual(c, Dict({0: 0, 1: 1, 2: 2, 3: 3}))

            self.assertIs(a.__or__(None), NotImplemented)
            self.assertIs(a.__or__(()), NotImplemented)
            self.assertIs(a.__or__("BAD"), NotImplemented)
            self.assertIs(a.__or__(""), NotImplemented)

            self.assertRaises(TypeError, a.__ior__, None)
            self.assertEqual(a.__ior__(()), {0: 0, 1: 1, 2: 1})
            self.assertRaises(ValueError, a.__ior__, "BAD")
            self.assertEqual(a.__ior__(""), {0: 0, 1: 1, 2: 1})

    def test_bool(self):
        self.assertIs(not {}, True)
        self.assertTrue(Dict({1: 2}))
        self.assertIs(bool(Dict({})), False)
        self.assertIs(bool(Dict({1: 2})), True)

    def test_keys(self):
        d = Dict()
        self.assertEqual(set(d.keys()), set())
        d = {"a": 1, "b": 2}
        k = d.keys()
        self.assertEqual(set(k), {"a", "b"})
        self.assertIn("a", k)
        self.assertIn("b", k)
        self.assertIn("a", d)
        self.assertIn("b", d)
        self.assertRaises(TypeError, d.keys, None)
        self.assertEqual(repr(dict(a=1).keys()), "dict_keys(['a'])")

    def test_values(self):
        d = Dict()
        self.assertEqual(set(d.values()), set())
        d = Dict({1: 2})
        self.assertEqual(set(d.values()), {2})
        self.assertRaises(TypeError, d.values, None)
        self.assertEqual(repr(dict(a=1).values()), "dict_values([1])")

    @pytest.mark.xfail
    def test_items(self):
        # TODO: support this when we have sets:
        d = Dict()
        self.assertEqual(set(d.items()), set())

        d = Dict({1: 2})
        self.assertEqual(set(d.items()), {(1, 2)})
        self.assertRaises(TypeError, d.items, None)
        self.assertEqual(repr(dict(a=1).items()), "dict_items([('a', 1)])")

    def test_contains(self):
        d = Dict()
        self.assertNotIn("a", d)
        self.assertFalse("a" in d)
        self.assertTrue("a" not in d)
        d = Dict({"a": 1, "b": 2})
        self.assertIn("a", d)
        self.assertIn("b", d)
        self.assertNotIn("c", d)

        self.assertRaises(TypeError, d.__contains__)

    def test_len(self):
        d = Dict()
        self.assertEqual(len(d), 0)
        d = Dict({"a": 1, "b": 2})
        self.assertEqual(len(d), 2)

    def test_getitem(self):
        d = Dict({"a": 1, "b": 2})
        self.assertEqual(d["a"], 1)
        self.assertEqual(d["b"], 2)
        d["c"] = 3
        d["a"] = 4
        self.assertEqual(d["c"], 3)
        self.assertEqual(d["a"], 4)
        del d["b"]
        self.assertEqual(d, {"a": 4, "c": 3})

        self.assertRaises(TypeError, d.__getitem__)

        class BadEq(object):
            def __eq__(self, other):
                raise Exc()

            def __hash__(self):
                return 24

        d = Dict()
        d[BadEq()] = 42
        self.assertRaises(KeyError, d.__getitem__, 23)

        class Exc(Exception):
            pass

        class BadHash(object):
            fail = False

            def __hash__(self):
                if self.fail:
                    raise Exc()
                else:
                    return 42

        x = BadHash()
        d[x] = 42
        x.fail = True
        self.assertRaises(Exc, d.__getitem__, x)

    def test_clear(self):
        d = Dict({1: 1, 2: 2, 3: 3})
        d.clear()
        self.assertEqual(d, {})

        self.assertRaises(TypeError, d.clear, None)

    def test_update(self):
        d = Dict()
        d.update({1: 100})
        d.update({2: 20})
        d.update({1: 1, 2: 2, 3: 3})
        self.assertEqual(d, {1: 1, 2: 2, 3: 3})

        d.update()
        self.assertEqual(d, {1: 1, 2: 2, 3: 3})

        self.assertRaises((TypeError, AttributeError), d.update, None)

        class SimpleUserDict:
            def __init__(self):
                self.d = {1: 1, 2: 2, 3: 3}

            def keys(self):
                return self.d.keys()

            def __getitem__(self, i):
                return self.d[i]

        d.clear()
        d.update(SimpleUserDict())
        self.assertEqual(d, {1: 1, 2: 2, 3: 3})

        class Exc(Exception):
            pass

        d.clear()

        class FailingUserDict:
            def keys(self):
                raise Exc

        self.assertRaises(Exc, d.update, FailingUserDict())

        class FailingUserDict:
            def keys(self):
                class BogonIter:
                    def __init__(self):
                        self.i = 1

                    def __iter__(self):
                        return self

                    def __next__(self):
                        if self.i:
                            self.i = 0
                            return "a"
                        raise Exc

                return BogonIter()

            def __getitem__(self, key):
                return key

        self.assertRaises(Exc, d.update, FailingUserDict())

        class FailingUserDict:
            def keys(self):
                class BogonIter:
                    def __init__(self):
                        self.i = ord("a")

                    def __iter__(self):
                        return self

                    def __next__(self):
                        if self.i <= ord("z"):
                            rtn = chr(self.i)
                            self.i += 1
                            return rtn
                        raise StopIteration

                return BogonIter()

            def __getitem__(self, key):
                raise Exc

        self.assertRaises(Exc, d.update, FailingUserDict())

        class badseq(object):
            def __iter__(self):
                return self

            def __next__(self):
                raise Exc()

        self.assertRaises(Exc, {}.update, badseq())

        self.assertRaises(ValueError, {}.update, [(1, 2, 3)])

    def test_fromkeys(self):
        self.assertEqual(dict.fromkeys("abc"), {"a": None, "b": None, "c": None})
        d = Dict()
        self.assertIsNot(d.fromkeys("abc"), d)
        self.assertEqual(d.fromkeys("abc"), {"a": None, "b": None, "c": None})
        self.assertEqual(d.fromkeys((4, 5), 0), {4: 0, 5: 0})
        self.assertEqual(d.fromkeys([]), {})

        def g():
            yield 1

        self.assertEqual(d.fromkeys(g()), {1: None})
        self.assertRaises(TypeError, {}.fromkeys, 3)

        class dictlike(dict):
            pass

        self.assertEqual(dictlike.fromkeys("a"), {"a": None})
        self.assertEqual(dictlike().fromkeys("a"), {"a": None})
        self.assertIsInstance(dictlike.fromkeys("a"), dictlike)
        self.assertIsInstance(dictlike().fromkeys("a"), dictlike)

        class mydict(dict):
            def __new__(cls):
                return Dict()

        ud = mydict.fromkeys("ab")
        self.assertEqual(ud, {"a": None, "b": None})
        self.assertIsInstance(ud, Dict)
        self.assertRaises(TypeError, dict.fromkeys)

        class Exc(Exception):
            pass

        class baddict1(dict):
            def __init__(self):
                raise Exc()

        self.assertRaises(Exc, baddict1.fromkeys, [1])

        class BadSeq(object):
            def __iter__(self):
                return self

            def __next__(self):
                raise Exc()

        self.assertRaises(Exc, dict.fromkeys, BadSeq())

        class baddict2(dict):
            def __setitem__(self, key, value):
                raise Exc()

        self.assertRaises(Exc, baddict2.fromkeys, [1])

        # test fast path for dictionary inputs
        d = dict(zip(range(6), range(6)))
        self.assertEqual(dict.fromkeys(d, 0), dict(zip(range(6), [0] * 6)))

        class baddict3(dict):
            def __new__(cls):
                return d

        d = {i: i for i in range(10)}
        res = d.copy()
        res.update(a=None, b=None, c=None)
        self.assertEqual(baddict3.fromkeys({"a", "b", "c"}), res)

    def test_copy(self):
        d = Dict({1: 1, 2: 2, 3: 3})
        self.assertIsNot(d.copy(), d)
        self.assertEqual(d.copy(), d)
        self.assertEqual(d.copy(), {1: 1, 2: 2, 3: 3})

        copy = d.copy()
        d[4] = 4
        self.assertNotEqual(copy, d)

        self.assertEqual({}.copy(), {})
        self.assertRaises(TypeError, d.copy, None)

    @pytest.mark.slow
    def test_copy_fuzz(self):
        for dict_size in [10, 100, 1000]:  # TODO: 10000, 100000
            dict_size = random.randrange(dict_size // 2, dict_size + dict_size // 2)
            with self.subTest(dict_size=dict_size):
                d = Dict()
                for i in range(dict_size):
                    d[i] = i

                d2 = d.copy()
                self.assertIsNot(d2, d)
                self.assertEqual(d, d2)
                d2["key"] = "value"
                self.assertNotEqual(d, d2)
                self.assertEqual(len(d2), len(d) + 1)

    def test_copy_maintains_tracking(self):
        class A:
            pass

        key = A()

        for d in (Dict(), Dict({"a": 1}), Dict({key: "val"})):
            d2 = d.copy()
            self.assertEqual(gc.is_tracked(d), gc.is_tracked(d2))

    def test_copy_noncompact(self):
        # Dicts don't compact themselves on del/pop operations.
        # Copy will use a slow merging strategy that produces
        # a compacted copy when roughly 33% of dict is a non-used
        # keys-space (to optimize memory footprint).
        # In this test we want to hit the slow/compacting
        # branch of dict.copy() and make sure it works OK.
        d = Dict({k: k for k in range(1000)})
        for k in range(950):
            del d[k]
        d2 = d.copy()
        self.assertEqual(d2, d)

    def test_get(self):
        # We call dict_get because of the conflict with our "get" method
        # from pointers
        d = Dict()
        self.assertIs(d.dict_get("c"), SyNone)
        self.assertEqual(d.dict_get("c", 3), 3)
        d = Dict({"a": 1, "b": 2})
        self.assertIs(d.dict_get("c"), SyNone)
        self.assertEqual(d.dict_get("c", 3), 3)
        self.assertEqual(d.dict_get("a"), 1)
        self.assertEqual(d.dict_get("a", 3), 1)
        self.assertRaises(TypeError, d.get)
        self.assertRaises(TypeError, d.get, None, None, None)

    def test_setdefault(self):
        # dict.setdefault()
        d = Dict()
        self.assertIs(d.setdefault("key0"), SyNone)
        d.setdefault("key0", [])
        self.assertIs(d.setdefault("key0"), SyNone)
        d.setdefault("key", []).append(3)
        self.assertEqual(d["key"][0], 3)
        d.setdefault("key", []).append(4)
        self.assertEqual(len(d["key"]), 2)
        self.assertRaises(TypeError, d.setdefault)

        class Exc(Exception):
            pass

        class BadHash(object):
            fail = False

            def __hash__(self):
                if self.fail:
                    raise Exc()
                else:
                    return 42

        x = BadHash()
        d[x] = 42
        x.fail = True
        self.assertRaises(Exc, d.setdefault, x, [])

    def test_setdefault_atomic(self):
        # Issue #13521: setdefault() calls __hash__ and __eq__ only once.
        class Hashed(object):
            def __init__(self):
                self.hash_count = 0
                self.eq_count = 0

            def __hash__(self):
                self.hash_count += 1
                return 42

            def __eq__(self, other):
                self.eq_count += 1
                return id(self) == id(other)

        hashed1 = Hashed()
        y = {hashed1: 5}
        hashed2 = Hashed()
        y.setdefault(hashed2, [])
        self.assertEqual(hashed1.hash_count, 1)
        self.assertEqual(hashed2.hash_count, 1)
        self.assertEqual(hashed1.eq_count + hashed2.eq_count, 1)

    def test_setitem_atomic_at_resize(self):
        class Hashed(object):
            def __init__(self):
                self.hash_count = 0
                self.eq_count = 0

            def __hash__(self):
                self.hash_count += 1
                return 42

            def __eq__(self, other):
                self.eq_count += 1
                return id(self) == id(other)

        hashed1 = Hashed()
        # 5 items
        y = Dict({hashed1: 5, 0: 0, 1: 1, 2: 2, 3: 3})
        hashed2 = Hashed()
        # 6th item forces a resize
        y[hashed2] = []
        # this is different for UserDict which is 3
        # we are subclassing UserDict so if we match UserDict that should be correct
        # self.assertEqual(hashed1.hash_count, 1)
        self.assertEqual(hashed1.hash_count, 3)
        self.assertEqual(hashed2.hash_count, 1)
        self.assertEqual(hashed1.eq_count + hashed2.eq_count, 1)

    @pytest.mark.slow
    def test_popitem(self):
        # dict.popitem()
        for copymode in -1, +1:
            # -1: b has same structure as a
            # +1: b is a.copy()
            for log2size in range(12):
                size = 2 ** log2size
                a = Dict()
                b = Dict()
                for i in range(size):
                    a[repr(i)] = i
                    if copymode < 0:
                        b[repr(i)] = i
                if copymode > 0:
                    b = a.copy()
                for i in range(size):
                    ka, va = ta = a.popitem()
                    self.assertEqual(va, ka.__int__())
                    kb, vb = tb = b.popitem()
                    self.assertEqual(vb, kb.__int__())
                    self.assertFalse(copymode < 0 and ta != tb)
                self.assertFalse(a)
                self.assertFalse(b)

        d = {}
        self.assertRaises(KeyError, d.popitem)

    def test_pop(self):
        # Tests for pop with specified key
        d = Dict()
        k, v = "abc", "def"
        d[k] = v
        self.assertRaises(KeyError, d.pop, "ghi")

        self.assertEqual(d.pop(k), v)
        self.assertEqual(len(d), 0)

        self.assertRaises(KeyError, d.pop, k)

        self.assertEqual(d.pop(k, v), v)
        d[k] = v
        self.assertEqual(d.pop(k, 1), v)

        self.assertRaises(TypeError, d.pop)

        class Exc(Exception):
            pass

        class BadHash(object):
            fail = False

            def __hash__(self):
                if self.fail:
                    raise Exc()
                else:
                    return 42

        x = BadHash()
        d[x] = 42
        x.fail = True
        self.assertRaises(Exc, d.pop, x)

    def test_mutating_iteration(self):
        # changing dict size during iteration
        d = Dict()
        d[1] = 1
        with self.assertRaises(RuntimeError):
            for i in d:
                d[i + 1] = 1

    def test_mutating_iteration_delete(self):
        # change dict content during iteration
        d = Dict()
        d[0] = 0
        # python 3.8+ raise RuntimeError but older versions do not
        if sys.version_info >= (3, 8):
            with self.assertRaises(RuntimeError):
                for i in d:
                    del d[0]
                    d[0] = 0

    def test_mutating_iteration_delete_over_values(self):
        # change dict content during iteration
        d = Dict()
        d[0] = 0
        # python 3.8+ raise RuntimeError but older versions do not
        if sys.version_info >= (3, 8):
            with self.assertRaises(RuntimeError):
                for i in d.values():
                    del d[0]
                    d[0] = 0

    @pytest.mark.xfail
    def test_mutating_iteration_delete_over_items(self):
        # TODO: proper iterators needed over the views, currently, we convert them to lists
        # change dict content during iteration
        d = Dict()
        d[0] = 0
        if sys.version_info >= (3, 8):
            with self.assertRaises(RuntimeError):
                for i in d.items():
                    del d[0]
                    d[0] = 0

    @pytest.mark.xfail
    def test_mutating_lookup(self):
        # changing dict during a lookup (issue #14417)
        # TODO: investigate this at some point
        class NastyKey:
            mutate_dict = None

            def __init__(self, value):
                self.value = value

            def __hash__(self):
                # hash collision!
                return 1

            def __eq__(self, other):
                if NastyKey.mutate_dict:
                    mydict, key = NastyKey.mutate_dict
                    NastyKey.mutate_dict = None
                    del mydict[key]
                return self.value == other.value

        key1 = NastyKey(1)
        key2 = NastyKey(2)
        d = Dict({key1: 1})
        NastyKey.mutate_dict = (d, key1)
        d[key2] = 2
        self.assertEqual(d, {key2: 2})

    def test_repr(self):
        d = Dict()
        self.assertEqual(repr(d), "{}")
        d[1] = 2
        self.assertEqual(repr(d), "{1: 2}")
        d = Dict()
        d[1] = d
        self.assertEqual(repr(d), "{1: {...}}")

        class Exc(Exception):
            pass

        class BadRepr(object):
            def __repr__(self):
                raise Exc()

        d = Dict({1: BadRepr()})
        self.assertRaises(Exc, repr, d)

    def test_repr_deep(self):
        d = Dict()
        for i in range(sys.getrecursionlimit() + 100):
            d = Dict({1: d})
        self.assertRaises(RecursionError, repr, d)

    def test_eq(self):
        self.assertEqual(Dict(), {})
        self.assertEqual(Dict({1: 2}), {1: 2})

        class Exc(Exception):
            pass

        class BadCmp(object):
            def __eq__(self, other):
                raise Exc()

            def __hash__(self):
                return 1

        d1 = Dict({BadCmp(): 1})
        d2 = Dict({1: 1})

        with self.assertRaises(Exc):
            d1 == d2

    @pytest.mark.xfail
    def test_keys_contained(self):
        self.helper_keys_contained(lambda x: x.keys())
        self.helper_keys_contained(lambda x: x.items())

    @pytest.mark.xfail
    def helper_keys_contained(self, fn):
        # TODO add this when we have set support
        # Test rich comparisons against dict key views, which should behave the
        # same as sets.
        empty = fn(Dict())
        empty2 = fn(Dict())
        smaller = fn(Dict({1: 1, 2: 2}))
        larger = fn(Dict({1: 1, 2: 2, 3: 3}))
        larger2 = fn(Dict({1: 1, 2: 2, 3: 3}))
        larger3 = fn(Dict({4: 1, 2: 2, 3: 3}))

        self.assertTrue(smaller < larger)
        self.assertTrue(smaller <= larger)
        self.assertTrue(larger > smaller)
        self.assertTrue(larger >= smaller)

        self.assertFalse(smaller >= larger)
        self.assertFalse(smaller > larger)
        self.assertFalse(larger <= smaller)
        self.assertFalse(larger < smaller)

        self.assertFalse(smaller < larger3)
        self.assertFalse(smaller <= larger3)
        self.assertFalse(larger3 > smaller)
        self.assertFalse(larger3 >= smaller)

        # Inequality strictness
        self.assertTrue(larger2 >= larger)
        self.assertTrue(larger2 <= larger)
        self.assertFalse(larger2 > larger)
        self.assertFalse(larger2 < larger)

        self.assertTrue(larger == larger2)
        self.assertTrue(smaller != larger)

        # There is an optimization on the zero-element case.
        self.assertTrue(empty == empty2)
        self.assertFalse(empty != empty2)
        self.assertFalse(empty == smaller)
        self.assertTrue(empty != smaller)

        # With the same size, an elementwise compare happens
        self.assertTrue(larger != larger3)
        self.assertFalse(larger == larger3)

    @pytest.mark.xfail
    def test_errors_in_view_containment_check(self):
        # TODO: add support for custom objects
        class C:
            def __eq__(self, other):
                raise RuntimeError

        d1 = Dict({1: C()})
        d2 = Dict({1: C()})
        with self.assertRaises(RuntimeError):
            d1.items() == d2.items()
        with self.assertRaises(RuntimeError):
            d1.items() != d2.items()
        with self.assertRaises(RuntimeError):
            d1.items() <= d2.items()
        with self.assertRaises(RuntimeError):
            d1.items() >= d2.items()

        d3 = Dict({1: C(), 2: C()})
        with self.assertRaises(RuntimeError):
            d2.items() < d3.items()
        with self.assertRaises(RuntimeError):
            d3.items() > d2.items()

    @pytest.mark.xfail
    def test_dictview_set_operations_on_keys(self):
        # TODO add support for sets
        k1 = Dict({1: 1, 2: 2}).keys()
        k2 = Dict({1: 1, 2: 2, 3: 3}).keys()
        k3 = Dict({4: 4}).keys()

        self.assertEqual(k1 - k2, set())
        self.assertEqual(k1 - k3, {1, 2})
        self.assertEqual(k2 - k1, {3})
        self.assertEqual(k3 - k1, {4})
        self.assertEqual(k1 & k2, {1, 2})
        self.assertEqual(k1 & k3, set())
        self.assertEqual(k1 | k2, {1, 2, 3})
        self.assertEqual(k1 ^ k2, {3})
        self.assertEqual(k1 ^ k3, {1, 2, 4})

    @pytest.mark.xfail
    def test_dictview_set_operations_on_items(self):
        # TODO add support for sets
        k1 = Dict({1: 1, 2: 2}).items()
        k2 = Dict({1: 1, 2: 2, 3: 3}).items()
        k3 = Dict({4: 4}).items()

        self.assertEqual(k1 - k2, set())
        self.assertEqual(k1 - k3, {(1, 1), (2, 2)})
        self.assertEqual(k2 - k1, {(3, 3)})
        self.assertEqual(k3 - k1, {(4, 4)})
        self.assertEqual(k1 & k2, {(1, 1), (2, 2)})
        self.assertEqual(k1 & k3, set())
        self.assertEqual(k1 | k2, {(1, 1), (2, 2), (3, 3)})
        self.assertEqual(k1 ^ k2, {(3, 3)})
        self.assertEqual(k1 ^ k3, {(1, 1), (2, 2), (4, 4)})

    @pytest.mark.xfail
    def test_dictview_mixed_set_operations(self):
        # TODO add support for sets
        # Just a few for .keys()
        self.assertTrue(Dict({1: 1}).keys() == {1})
        self.assertEqual(Dict({1: 1}).keys() | {2}, {1, 2})
        # And a few for .items()
        self.assertTrue(Dict({1: 1}).items() == {(1, 1)})

        # This test has been changed to reflect the behavior of UserDict
        self.assertTrue(Dict({(1, 1)}) == {1: 1})

        # UserDict does not support init with set items like:
        # UserDict({2}) so neither do we with Dict
        with pytest.raises(TypeError):
            self.assertEqual(Dict({2}) | Dict({1: 1}).keys(), {1, 2})
            self.assertTrue(Dict({1}) == {1: 1}.keys())
            self.assertEqual(Dict({2}) | Dict({1: 1}).items(), {(1, 1), 2})
            self.assertEqual(Dict({1: 1}).items() | Dict({2}), {(1, 1), 2})

    def test_missing(self):
        # Make sure dict doesn't have a __missing__ method
        self.assertFalse(hasattr(Dict, "__missing__"))
        self.assertFalse(hasattr(Dict(), "__missing__"))
        # Test several cases:
        # (D) subclass defines __missing__ method returning a value
        # (E) subclass defines __missing__ method raising RuntimeError
        # (F) subclass sets __missing__ instance variable (no effect)
        # (G) subclass doesn't define __missing__ at all

        class D(Dict):
            def __missing__(self, key):
                return 42

        d = D({1: 2, 3: 4})
        self.assertEqual(d[1], 2)
        self.assertEqual(d[3], 4)
        self.assertNotIn(2, d)
        self.assertNotIn(2, d.keys())
        self.assertEqual(d[2], 42)

        class E(dict):
            def __missing__(self, key):
                raise RuntimeError(key)

        e = E()
        with self.assertRaises(RuntimeError) as c:
            e[42]
        self.assertEqual(c.exception.args, (42,))

        class F(dict):
            def __init__(self):
                # An instance variable __missing__ should have no effect
                self.__missing__ = lambda key: None

        f = F()
        with self.assertRaises(KeyError) as c:
            f[42]
        self.assertEqual(c.exception.args, (42,))

        class G(dict):
            pass

        g = G()
        with self.assertRaises(KeyError) as c:
            g[42]
        self.assertEqual(c.exception.args, (42,))

    def test_tuple_keyerror(self):
        # SF #1576657
        d = Dict()
        with self.assertRaises(KeyError) as c:
            d[(1,)]
        self.assertEqual(c.exception.args, ((1,),))

    def test_bad_key(self):
        # Dictionary lookups should fail if __eq__() raises an exception.
        class CustomException(Exception):
            pass

        class BadDictKey:
            def __hash__(self):
                return hash(self.__class__)

            def __eq__(self, other):
                if isinstance(other, self.__class__):
                    raise CustomException
                return other

        d = Dict()
        x1 = BadDictKey()
        x2 = BadDictKey()
        d[x1] = 1
        for stmt in [
            "d[x2] = 2",
            "z = d[x2]",
            "x2 in d",
            "d.get(x2)",
            "d.setdefault(x2, 42)",
            "d.pop(x2)",
            "d.update({x2: 2})",
        ]:
            with self.assertRaises(CustomException):
                exec(stmt, locals())

    def test_resize1(self):
        # Dict resizing bug, found by Jack Jansen in 2.2 CVS development.
        # This version got an assert failure in debug build, infinite loop in
        # release build.  Unfortunately, provoking this kind of stuff requires
        # a mix of inserts and deletes hitting exactly the right hash codes in
        # exactly the right order, and I can't think of a randomized approach
        # that would be *likely* to hit a failing case in reasonable time.

        d = Dict()
        for i in range(5):
            d[i] = i
        for i in range(5):
            del d[i]
        for i in range(5, 9):  # i==8 was the problem
            d[i] = i

    def test_resize2(self):
        # Another dict resizing bug (SF bug #1456209).
        # This caused Segmentation faults or Illegal instructions.

        class X(object):
            def __hash__(self):
                return 5

            def __eq__(self, other):
                if resizing:
                    d.clear()
                return False

        d = Dict()
        resizing = False
        d[X()] = 1
        d[X()] = 2
        d[X()] = 3
        d[X()] = 4
        d[X()] = 5
        # now trigger a resize
        resizing = True
        d[9] = 6

    def test_empty_presized_dict_in_freelist(self):
        # Bug #3537: if an empty but presized dict with a size larger
        # than 7 was in the freelist, it triggered an assertion failure
        with self.assertRaises(ZeroDivisionError):
            d = Dict(
                {
                    "a": 1 // 0,
                    "b": None,
                    "c": None,
                    "d": None,
                    "e": None,
                    "f": None,
                    "g": None,
                    "h": None,
                }
            )
            d.clear()

    @pytest.mark.xfail
    @pytest.mark.slow
    def test_container_iterator(self):
        # TODO: make this pass
        # Bug #3680: tp_traverse was not implemented for dictiter and
        # dictview objects.
        class C(object):
            pass

        views = (Dict.items, Dict.values, Dict.keys)
        for v in views:
            obj = C()
            ref = weakref.ref(obj)
            container = {obj: 1}
            obj.v = v(container)
            obj.x = iter(obj.v)
            del obj, container
            gc.collect()
            self.assertIs(ref(), None, "Cycle was not collected")

    def _not_tracked(self, t):
        # Nested containers can take several collections to untrack
        gc.collect()
        gc.collect()
        # UserDict is tracked unlike normal dict so we have to change
        # this test for our Dict
        # self.assertFalse(gc.is_tracked(t), t)
        self.assertTrue(gc.is_tracked(t), t)

    def _tracked(self, t):
        self.assertTrue(gc.is_tracked(t), t)
        gc.collect()
        gc.collect()
        self.assertTrue(gc.is_tracked(t), t)

    @pytest.mark.slow
    @support.cpython_only
    def test_track_literals(self):
        # Test GC-optimization of dict literals
        x, y, z, w = 1.5, "a", (1, None), []

        self._not_tracked(Dict())
        self._not_tracked(Dict({x: (), y: x, z: 1}))
        self._not_tracked(Dict({1: "a", "b": 2}))
        self._not_tracked(Dict({1: 2, (None, True, False, ()): int}))
        self._not_tracked(Dict({1: object()}))

        # Dicts with mutable elements are always tracked, even if those
        # elements are not tracked right now.
        self._tracked(Dict({1: []}))
        self._tracked(Dict({1: ([],)}))
        self._tracked(Dict({1: {}}))
        self._tracked(Dict({1: set()}))

    @pytest.mark.slow
    @support.cpython_only
    def test_track_dynamic(self):
        # Test GC-optimization of dynamically-created dicts
        class MyObject(object):
            pass

        x, y, z, w, o = 1.5, "a", (1, object()), [], MyObject()

        d = Dict()
        self._not_tracked(d)
        d[1] = "a"
        self._not_tracked(d)
        d[y] = 2
        self._not_tracked(d)
        d[z] = 3
        self._not_tracked(d)
        self._not_tracked(d.copy())
        d[4] = w
        self._tracked(d)
        self._tracked(d.copy())
        d[4] = None
        self._not_tracked(d)
        self._not_tracked(d.copy())

        # dd isn't tracked right now, but it may mutate and therefore d
        # which contains it must be tracked.
        d = Dict()
        dd = Dict()
        d[1] = dd
        self._not_tracked(dd)
        self._tracked(d)
        dd[1] = d
        self._tracked(dd)

        d = Dict.fromkeys([x, y, z])
        self._not_tracked(d)
        dd = Dict()
        dd.update(d)
        self._not_tracked(dd)
        d = Dict.fromkeys([x, y, z, o])
        self._tracked(d)
        dd = Dict()
        dd.update(d)
        self._tracked(dd)

        d = Dict(x=x, y=y, z=z)
        self._not_tracked(d)
        d = Dict(x=x, y=y, z=z, w=w)
        self._tracked(d)
        d = Dict()
        d.update(x=x, y=y, z=z)
        self._not_tracked(d)
        d.update(w=w)
        self._tracked(d)

        d = Dict([(x, y), (z, 1)])
        self._not_tracked(d)
        d = Dict([(x, y), (z, w)])
        self._tracked(d)
        d = Dict()
        d.update([(x, y), (z, 1)])
        self._not_tracked(d)
        d.update([(x, y), (z, w)])
        self._tracked(d)

    @support.cpython_only
    def test_track_subtypes(self):
        # Dict subtypes are always tracked
        class MyDict(Dict):
            pass

        self._tracked(MyDict())

    def make_shared_key_dict(self, n):
        class C:
            pass

        dicts = []
        for i in range(n):
            a = C()
            a.x, a.y, a.z = 1, 2, 3
            dicts.append(a.__dict__)

        return dicts

    @support.cpython_only
    def test_splittable_setdefault(self):
        """split table must be combined when setdefault()
        breaks insertion order"""
        a, b = self.make_shared_key_dict(2)

        a["a"] = 1
        size_a = sys.getsizeof(a)
        a["b"] = 2
        b.setdefault("b", 2)
        size_b = sys.getsizeof(b)
        b["a"] = 1

        self.assertGreater(size_b, size_a)
        self.assertEqual(list(a), ["x", "y", "z", "a", "b"])
        self.assertEqual(list(b), ["x", "y", "z", "b", "a"])

    @support.cpython_only
    def test_splittable_del(self):
        """split table must be combined when del d[k]"""
        a, b = self.make_shared_key_dict(2)

        orig_size = sys.getsizeof(a)

        del a["y"]  # split table is combined
        with self.assertRaises(KeyError):
            del a["y"]

        self.assertGreater(sys.getsizeof(a), orig_size)
        self.assertEqual(list(a), ["x", "z"])
        self.assertEqual(list(b), ["x", "y", "z"])

        # Two dicts have different insertion order.
        a["y"] = 42
        self.assertEqual(list(a), ["x", "z", "y"])
        self.assertEqual(list(b), ["x", "y", "z"])

    @support.cpython_only
    def test_splittable_pop(self):
        """split table must be combined when d.pop(k)"""
        a, b = self.make_shared_key_dict(2)

        orig_size = sys.getsizeof(a)

        a.pop("y")  # split table is combined
        with self.assertRaises(KeyError):
            a.pop("y")

        self.assertGreater(sys.getsizeof(a), orig_size)
        self.assertEqual(list(a), ["x", "z"])
        self.assertEqual(list(b), ["x", "y", "z"])

        # Two dicts have different insertion order.
        a["y"] = 42
        self.assertEqual(list(a), ["x", "z", "y"])
        self.assertEqual(list(b), ["x", "y", "z"])

    @support.cpython_only
    def test_splittable_pop_pending(self):
        """pop a pending key in a splitted table should not crash"""
        a, b = self.make_shared_key_dict(2)

        a["a"] = 4
        with self.assertRaises(KeyError):
            b.pop("a")

    @support.cpython_only
    def test_splittable_popitem(self):
        """split table must be combined when d.popitem()"""
        a, b = self.make_shared_key_dict(2)

        orig_size = sys.getsizeof(a)

        item = a.popitem()  # split table is combined
        self.assertEqual(item, ("z", 3))
        with self.assertRaises(KeyError):
            del a["z"]

        self.assertGreater(sys.getsizeof(a), orig_size)
        self.assertEqual(list(a), ["x", "y"])
        self.assertEqual(list(b), ["x", "y", "z"])

    @support.cpython_only
    def test_splittable_setattr_after_pop(self):
        """setattr() must not convert combined table into split table."""
        # Issue 28147
        # third party
        import _testcapi

        class C:
            pass

        a = C()

        a.a = 1
        self.assertTrue(_testcapi.dict_hassplittable(a.__dict__))

        # dict.pop() convert it to combined table
        a.__dict__.pop("a")
        self.assertFalse(_testcapi.dict_hassplittable(a.__dict__))

        # But C should not convert a.__dict__ to split table again.
        a.a = 1
        self.assertFalse(_testcapi.dict_hassplittable(a.__dict__))

        # Same for popitem()
        a = C()
        a.a = 2
        self.assertTrue(_testcapi.dict_hassplittable(a.__dict__))
        a.__dict__.popitem()
        self.assertFalse(_testcapi.dict_hassplittable(a.__dict__))
        a.a = 3
        self.assertFalse(_testcapi.dict_hassplittable(a.__dict__))

    @pytest.mark.xfail
    def test_iterator_pickling(self):
        # set to xfail because we dont really care about pickling
        # see test_valuesiterator_pickling
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            data = Dict({1: "a", 2: "b", 3: "c"})
            it = iter(data)
            d = pickle.dumps(it, proto)
            it = pickle.loads(d)
            self.assertEqual(list(it), list(data))

            it = pickle.loads(d)
            try:
                drop = next(it)
            except StopIteration:
                continue
            d = pickle.dumps(it, proto)
            it = pickle.loads(d)
            del data[drop]
            self.assertEqual(list(it), list(data))

    def test_itemiterator_pickling(self):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            # UserDict fails these tests so our Dict should fail as well
            with pytest.raises(TypeError):
                data = Dict({1: "a", 2: "b", 3: "c"})
                # dictviews aren't picklable, only their iterators
                itorg = iter(data.items())
                d = pickle.dumps(itorg, proto)
                it = pickle.loads(d)
                # note that the type of the unpickled iterator
                # is not necessarily the same as the original.  It is
                # merely an object supporting the iterator protocol, yielding
                # the same objects as the original one.
                # self.assertEqual(type(itorg), type(it))
                self.assertIsInstance(it, collections.abc.Iterator)
                self.assertEqual(Dict(it), data)

                it = pickle.loads(d)
                drop = next(it)
                d = pickle.dumps(it, proto)
                it = pickle.loads(d)
                del data[drop[0]]
                self.assertEqual(Dict(it), data)

    def test_valuesiterator_pickling(self):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            # UserDict fails these tests so our Dict should fail as well
            with pytest.raises(TypeError):
                data = Dict({1: "a", 2: "b", 3: "c"})
                # data.values() isn't picklable, only its iterator
                it = iter(data.values())
                d = pickle.dumps(it, proto)
                it = pickle.loads(d)
                self.assertEqual(list(it), list(data.values()))

                it = pickle.loads(d)
                drop = next(it)
                d = pickle.dumps(it, proto)
                it = pickle.loads(d)
                values = list(it) + [drop]
                self.assertEqual(sorted(values), sorted(list(data.values())))

    def test_reverseiterator_pickling(self):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            # UserDict fails these tests so our Dict should fail as well
            with pytest.raises(TypeError):
                data = Dict({1: "a", 2: "b", 3: "c"})
                it = reversed(data)
                d = pickle.dumps(it, proto)
                it = pickle.loads(d)
                self.assertEqual(list(it), list(reversed(data)))

                it = pickle.loads(d)
                try:
                    drop = next(it)
                except StopIteration:
                    continue
                d = pickle.dumps(it, proto)
                it = pickle.loads(d)
                del data[drop]
                self.assertEqual(list(it), list(reversed(data)))

    def test_reverseitemiterator_pickling(self):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            # UserDict fails these tests so our Dict should fail as well
            with pytest.raises(TypeError):
                data = Dict({1: "a", 2: "b", 3: "c"})
                # dictviews aren't picklable, only their iterators
                itorg = reversed(data.items())
                d = pickle.dumps(itorg, proto)
                it = pickle.loads(d)
                # note that the type of the unpickled iterator
                # is not necessarily the same as the original.  It is
                # merely an object supporting the iterator protocol, yielding
                # the same objects as the original one.
                # self.assertEqual(type(itorg), type(it))
                self.assertIsInstance(it, collections.abc.Iterator)
                self.assertEqual(Dict(it), data)

                it = pickle.loads(d)
                drop = next(it)
                d = pickle.dumps(it, proto)
                it = pickle.loads(d)
                del data[drop[0]]
                self.assertEqual(Dict(it), data)

    def test_reversevaluesiterator_pickling(self):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            # UserDict fails these tests so our Dict should fail as well
            with pytest.raises(TypeError):
                data = Dict({1: "a", 2: "b", 3: "c"})
                # data.values() isn't picklable, only its iterator
                it = reversed(data.values())
                d = pickle.dumps(it, proto)
                it = pickle.loads(d)
                self.assertEqual(list(it), list(reversed(data.values())))

                it = pickle.loads(d)
                drop = next(it)
                d = pickle.dumps(it, proto)
                it = pickle.loads(d)
                values = list(it) + [drop]
                self.assertEqual(sorted(values), sorted(data.values()))

    def test_instance_dict_getattr_str_subclass(self):
        class Foo:
            def __init__(self, msg):
                self.msg = msg

        f = Foo("123")

        class _str(str):
            pass

        self.assertEqual(f.msg, getattr(f, _str("msg")))
        self.assertEqual(f.msg, f.__dict__[_str("msg")])

    def test_object_set_item_single_instance_non_str_key(self):
        class Foo:
            pass

        f = Foo()
        f.__dict__[1] = 1
        f.a = "a"
        self.assertEqual(f.__dict__, {1: 1, "a": "a"})

    def check_reentrant_insertion(self, mutate):
        # This object will trigger mutation of the dict when replaced
        # by another value.  Note this relies on refcounting: the test
        # won't achieve its purpose on fully-GCed Python implementations.
        class Mutating:
            def __del__(self):
                mutate(d)

        d = Dict({k: Mutating() for k in "abcdefghijklmnopqr"})
        for k in list(d):
            d[k] = k

    def test_reentrant_insertion(self):
        # Reentrant insertion shouldn't crash (see issue #22653)
        def mutate(d):
            d["b"] = 5

        self.check_reentrant_insertion(mutate)

        def mutate(d):
            d.update(self.__dict__)
            d.clear()

        self.check_reentrant_insertion(mutate)

        def mutate(d):
            while d:
                d.popitem()

        self.check_reentrant_insertion(mutate)

    @pytest.mark.slow
    def test_merge_and_mutate(self):
        # this fails because it expects a RuntimeError when the keys change, however
        # the test_dictitems_contains_use_after_free expects StopIteration when the
        # keys change?
        class X:
            def __hash__(self):
                return 0

            def __eq__(self, o):
                other.clear()
                return False

        l = [(i, 0) for i in range(1, 1337)]
        other = Dict(l)
        other[X()] = 0
        d = Dict({X(): 0, 1: 1})
        self.assertRaises(RuntimeError, d.update, other)

    @pytest.mark.xfail
    @pytest.mark.slow
    def test_free_after_iterating(self):
        # this seems like a bit of a puzzle
        support.check_free_after_iterating(self, iter, Dict)
        support.check_free_after_iterating(self, lambda d: iter(d.keys()), Dict)
        support.check_free_after_iterating(self, lambda d: iter(d.values()), Dict)
        support.check_free_after_iterating(self, lambda d: iter(d.items()), Dict)

    def test_equal_operator_modifying_operand(self):
        # test fix for seg fault reported in bpo-27945 part 3.
        class X:
            def __del__(self):
                dict_b.clear()

            def __eq__(self, other):
                dict_a.clear()
                return True

            def __hash__(self):
                return 13

        dict_a = Dict({X(): 0})
        dict_b = Dict({X(): X()})
        self.assertTrue(dict_a == dict_b)

        # test fix for seg fault reported in bpo-38588 part 1.
        class Y:
            def __eq__(self, other):
                dict_d.clear()
                return True

        dict_c = Dict({0: Y()})
        dict_d = Dict({0: set()})
        self.assertTrue(dict_c == dict_d)

    def test_fromkeys_operator_modifying_dict_operand(self):
        # test fix for seg fault reported in issue 27945 part 4a.
        class X(int):
            def __hash__(self):
                return 13

            def __eq__(self, other):
                if len(d) > 1:
                    d.clear()
                return False

        d = Dict()  # this is required to exist so that d can be constructed!
        d = Dict({X(1): 1, X(2): 2})
        try:
            dict.fromkeys(d)  # shouldn't crash
        except RuntimeError:  # implementation defined
            pass

    def test_fromkeys_operator_modifying_set_operand(self):
        # test fix for seg fault reported in issue 27945 part 4b.
        class X(int):
            def __hash__(self):
                return 13

            def __eq__(self, other):
                if len(d) > 1:
                    d.clear()
                return False

        d = {}  # this is required to exist so that d can be constructed!
        d = {X(1), X(2)}
        try:
            Dict.fromkeys(d)  # shouldn't crash
        except RuntimeError:  # implementation defined
            pass

    @pytest.mark.xfail
    def test_dictitems_contains_use_after_free(self):
        # this seems like a bit of a puzzle
        # see iterator.py for more details
        class X:
            def __eq__(self, other):
                d.clear()
                return NotImplemented

        d = Dict({0: set()})  # TODO: we should be able to support set
        (0, X()) in d.items()

    def test_dict_contain_use_after_free(self):
        # bpo-40489
        class S(str):
            def __eq__(self, other):
                d.clear()
                return NotImplemented

            def __hash__(self):
                return hash("test")

        d = Dict({S(): "value"})
        self.assertFalse("test" in d)

    def test_init_use_after_free(self):
        class X:
            def __hash__(self):
                pair[:] = []
                return 13

        pair = [X(), 123]
        Dict([pair])

    @pytest.mark.xfail
    def test_oob_indexing_dictiter_iternextitem(self):
        class X(int):
            def __del__(self):
                d.clear()

        d = Dict({i: X(i) for i in range(8)})

        def iter_and_mutate():
            for result in d.items():
                if result[0] == 2:
                    d[2] = None  # free d[2] --> X(2).__del__ was called

        self.assertRaises(RuntimeError, iter_and_mutate)

    def test_reversed(self):
        d = Dict({"a": 1, "b": 2, "foo": 0, "c": 3, "d": 4})
        del d["foo"]
        # UserDict does not support reversed so we do not either
        with pytest.raises(TypeError):
            r = reversed(d)
            self.assertEqual(list(r), list("dcba"))
            self.assertRaises(StopIteration, next, r)

    def test_reverse_iterator_for_empty_dict(self):
        # bpo-38525: revered iterator should work properly

        # empty dict is directly used for reference count test
        # UserDict does not support reversed so we do not either
        with pytest.raises(TypeError):
            self.assertEqual(list(reversed(Dict())), [])
            self.assertEqual(list(reversed(Dict().items())), [])
            self.assertEqual(list(reversed(Dict().values())), [])
            self.assertEqual(list(reversed(Dict().keys())), [])

            # dict() and {} don't trigger the same code path
            self.assertEqual(list(reversed(dict())), [])
            self.assertEqual(list(reversed(dict().items())), [])
            self.assertEqual(list(reversed(dict().values())), [])
            self.assertEqual(list(reversed(dict().keys())), [])

    @pytest.mark.xfail
    def test_reverse_iterator_for_shared_shared_dicts(self):
        # UserDict doesnt support reversed and this causes infinite recursion
        # we will just disable this test
        class A:
            def __init__(self, x, y):
                if x:
                    self.x = x
                if y:
                    self.y = y

        self.assertEqual(list(reversed(A(1, 2).__dict__)), ["y", "x"])
        self.assertEqual(list(reversed(A(1, 0).__dict__)), ["x"])
        self.assertEqual(list(reversed(A(0, 1).__dict__)), ["y"])

    @pytest.mark.xfail
    def test_dict_copy_order(self):
        # bpo-34320
        od = collections.OrderedDict([("a", 1), ("b", 2)])
        od.move_to_end("a")
        expected = list(od.items())

        copy = Dict(od)
        self.assertEqual(list(copy.items()), expected)

        # dict subclass doesn't override __iter__
        class CustomDict(Dict):
            pass

        pairs = [("a", 1), ("b", 2), ("c", 3)]

        d = CustomDict(pairs)
        self.assertEqual(pairs, list(Dict(d).items()))

        # UserDict doesnt support reversed and this causes infinite recursion
        # we will just disable this test
        class CustomReversedDict(dict):
            def keys(self):
                return reversed(list(dict.keys(self)))

            __iter__ = keys

            def items(self):
                return reversed(dict.items(self))

        d = CustomReversedDict(pairs)
        self.assertEqual(pairs[::-1], list(dict(d).items()))

    @support.cpython_only
    def test_dict_items_result_gc(self):
        # bpo-42536: dict.items's tuple-reuse speed trick breaks the GC's
        # assumptions about what can be untracked. Make sure we re-track result
        # tuples whenever we reuse them.
        it = iter(Dict({None: []}).items())
        gc.collect()
        # That GC collection probably untracked the recycled internal result
        # tuple, which is initialized to (None, None). Make sure it's re-tracked
        # when it's mutated and returned from __next__:
        self.assertTrue(gc.is_tracked(next(it)))

    @pytest.mark.xfail
    @support.cpython_only
    def test_dict_items_result_gc_reversed(self):
        # UserDict doesnt support reversed and this causes infinite recursion
        # Same as test_dict_items_result_gc above, but reversed.
        it = reversed(Dict({None: []}).items())
        gc.collect()
        self.assertTrue(gc.is_tracked(next(it)))
