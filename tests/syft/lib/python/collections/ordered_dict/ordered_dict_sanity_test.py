# flake8: noqa
"""
File copied from cpython test suite:
https://github.com/python/cpython/blob/3.9/Lib/test/test_ordered_dict.py

Sanity tests for OrderedDict
"""

# stdlib
from collections.abc import MutableMapping
import copy
import gc
import pickle
from random import randrange
from random import shuffle
import sys
import weakref

# third party
import pytest

# syft absolute
from syft.core.common.uid import UID
from syft.lib.python import SyNone
from syft.lib.python.collections import OrderedDict as SyOrderedDict


def assertEqual(left, right):
    assert left == right


def assertNotEqual(left, right):
    assert left != right


def assertNotIn(key, container):
    assert key not in container


def assertRaises(exc, obj, methodname, *args):
    with pytest.raises(exc) as e_info:
        getattr(obj, methodname)(*args)
    assert str(e_info) != ""


def test_init():
    OrderedDict = SyOrderedDict
    with pytest.raises(TypeError):
        OrderedDict([("a", 1), ("b", 2)], None)  # too many args
    pairs = [("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5)]
    assertEqual(sorted(OrderedDict(dict(pairs)).items()), pairs)  # dict input
    assertEqual(sorted(OrderedDict(**dict(pairs)).items()), pairs)  # kwds input
    assertEqual(list(OrderedDict(pairs).items()), pairs)  # pairs input
    assertEqual(
        list(OrderedDict([("a", 1), ("b", 2), ("c", 9), ("d", 4)], c=3, e=5).items()),
        pairs,
    )  # mixed input

    # make sure no positional args conflict with possible kwdargs
    assertEqual(list(OrderedDict(other=42).items()), [("other", 42)])
    assertRaises(TypeError, OrderedDict, 42)
    assertRaises(TypeError, OrderedDict, (), ())

    # Make sure that direct calls to __init__ do not clear previous contents
    d = OrderedDict([("a", 1), ("b", 2), ("c", 3), ("d", 44), ("e", 55)])
    d.__init__([("e", 5), ("f", 6)], g=7, d=4)
    assertEqual(
        list(d.items()),
        [("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5), ("f", 6), ("g", 7)],
    )


def test_468():
    OrderedDict = SyOrderedDict
    items = [("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5), ("f", 6), ("g", 7)]
    shuffle(items)
    argdict = OrderedDict(items)
    unpacked = {}
    for item in argdict:
        unpacked[str(item)] = argdict[item]
    d = OrderedDict(unpacked)
    assertEqual(list(d.items()), items)


def test_update():
    OrderedDict = SyOrderedDict
    with pytest.raises(TypeError):
        OrderedDict().update([("a", 1), ("b", 2)], None)  # too many args
    pairs = [("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5)]
    od = OrderedDict()
    od.update(dict(pairs))
    assertEqual(sorted(od.items()), pairs)  # dict input
    od = OrderedDict()
    od.update(**dict(pairs))
    assertEqual(sorted(od.items()), pairs)  # kwds input
    od = OrderedDict()
    od.update(pairs)
    assertEqual(list(od.items()), pairs)  # pairs input
    od = OrderedDict()
    od.update([("a", 1), ("b", 2), ("c", 9), ("d", 4)], c=3, e=5)
    assertEqual(list(od.items()), pairs)  # mixed input

    # Issue 9137: Named argument called 'other' or ''
    # shouldn't be treated specially.
    od = OrderedDict()

    od = OrderedDict()
    od.update(other={})
    assertEqual(list(od.items()), [("other", {})])

    # Make sure that direct calls to update do not clear previous contents
    # add that updates items are not moved to the end
    d = OrderedDict([("a", 1), ("b", 2), ("c", 3), ("d", 44), ("e", 55)])
    d.update([("e", 5), ("f", 6)], g=7, d=4)
    assertEqual(
        list(d.items()),
        [("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5), ("f", 6), ("g", 7)],
    )

    pytest.raises(TypeError, OrderedDict().update, 42)
    pytest.raises(TypeError, OrderedDict().update, (), ())

    d = OrderedDict(
        [("a", 1), ("b", 2), ("c", 3), ("d", 44), ("e", 55)],
        _id=UID.from_string(value="{12345678-1234-5678-1234-567812345678}"),
    )
    assert d.id.__eq__(UID.from_string(value="{12345678-1234-5678-1234-567812345678}"))


def test_init_calls():
    calls = []

    class Spam:
        def keys(self):
            calls.append("keys")
            return ()

        def items(self):
            calls.append("items")
            return ()

    SyOrderedDict(Spam())
    assertEqual(calls, ["keys"])


def test_FromKeys():
    OrderedDict = SyOrderedDict
    od = OrderedDict.FromKeys("abc")
    assertEqual(list(od.items()), [(c, None) for c in "abc"])
    od = OrderedDict.FromKeys("abc", value=None)
    assertEqual(list(od.items()), [(c, None) for c in "abc"])
    od = OrderedDict.FromKeys("abc", value=0)
    assertEqual(list(od.items()), [(c, 0) for c in "abc"])


def test_abc():
    OrderedDict = SyOrderedDict
    assert isinstance(OrderedDict(), MutableMapping)
    assert issubclass(OrderedDict, MutableMapping)


def test_clear():
    OrderedDict = SyOrderedDict
    pairs = [("c", 1), ("b", 2), ("a", 3), ("d", 4), ("e", 5), ("f", 6)]
    shuffle(pairs)
    od = OrderedDict(pairs)
    assertEqual(len(od), len(pairs))
    od.clear()
    assertEqual(len(od), 0)


def test_delitem():
    OrderedDict = SyOrderedDict
    pairs = [("c", 1), ("b", 2), ("a", 3), ("d", 4), ("e", 5), ("f", 6)]
    od = OrderedDict(pairs)
    del od["a"]
    assertNotIn("a", od)
    with pytest.raises(KeyError):
        del od["a"]
    assertEqual(list(od.items()), pairs[:2] + pairs[3:])


def test_setitem():
    OrderedDict = SyOrderedDict
    od = OrderedDict([("d", 1), ("b", 2), ("c", 3), ("a", 4), ("e", 5)])
    od["c"] = 10  # existing element
    od["f"] = 20  # new element
    assertEqual(
        list(od.items()), [("d", 1), ("b", 2), ("c", 10), ("a", 4), ("e", 5), ("f", 20)]
    )


def test_iterators():
    OrderedDict = SyOrderedDict
    pairs = [("c", 1), ("b", 2), ("a", 3), ("d", 4), ("e", 5), ("f", 6)]
    shuffle(pairs)
    od = OrderedDict(pairs)
    assertEqual(list(od), [t[0] for t in pairs])
    assertEqual(list(od.keys()), [t[0] for t in pairs])
    assertEqual(list(od.values()), [t[1] for t in pairs])
    assertEqual(list(od.items()), pairs)
    assertEqual(list(reversed(od)), [t[0] for t in reversed(pairs)])
    assertEqual(list(reversed(list(od.keys()))), [t[0] for t in reversed(pairs)])
    assertEqual(list(reversed(list(od.values()))), [t[1] for t in reversed(pairs)])
    assertEqual(list(reversed(list(od.items()))), list(reversed(pairs)))


def test_detect_deletion_during_iteration():
    OrderedDict = SyOrderedDict
    od = OrderedDict.FromKeys("abc")
    it = iter(od)
    key = next(it)
    del od[key]
    with pytest.raises(Exception):
        # Note, the exact exception raised is not guaranteed
        # The only guarantee that the next() will not succeed
        next(it)


def test_sorted_iterators():
    OrderedDict = SyOrderedDict
    with pytest.raises(TypeError):
        OrderedDict([("a", 1), ("b", 2)], None)
    pairs = [("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5)]
    od = OrderedDict(pairs)
    assertEqual(sorted(od), [t[0] for t in pairs])
    assertEqual(sorted(od.keys()), [t[0] for t in pairs])
    assertEqual(sorted(od.values()), [t[1] for t in pairs])
    assertEqual(sorted(od.items()), pairs)
    assertEqual(sorted(reversed(od)), sorted([t[0] for t in reversed(pairs)]))


def test_iterators_empty():
    OrderedDict = SyOrderedDict
    od = OrderedDict()
    empty = []
    assertEqual(list(od), empty)
    assertEqual(list(od.keys()), empty)
    assertEqual(list(od.values()), empty)
    assertEqual(list(od.items()), empty)
    assertEqual(list(reversed(od)), empty)
    assertEqual(list(reversed(list(od.keys()))), empty)
    assertEqual(list(reversed(list(od.values()))), empty)
    assertEqual(list(reversed(list(od.items()))), empty)


def test_popitem():
    OrderedDict = SyOrderedDict
    pairs = [("c", 1), ("b", 2), ("a", 3), ("d", 4), ("e", 5), ("f", 6)]
    shuffle(pairs)
    od = OrderedDict(pairs)
    while pairs:
        assertEqual(od.popitem(), pairs.pop())
    with pytest.raises(KeyError):
        od.popitem()
    assertEqual(len(od), 0)


def test_popitem_last():
    OrderedDict = SyOrderedDict
    pairs = [(i, i) for i in range(30)]

    obj = OrderedDict(pairs)
    for i in range(8):
        obj.popitem(True)
    obj.popitem(True)
    obj.popitem(last=True)
    assertEqual(len(obj), 20)


def test_pop():
    OrderedDict = SyOrderedDict
    pairs = [("c", 1), ("b", 2), ("a", 3), ("d", 4), ("e", 5), ("f", 6)]
    shuffle(pairs)
    od = OrderedDict(pairs)
    shuffle(pairs)
    while pairs:
        k, v = pairs.pop()
        assertEqual(od.pop(k), v)
    with pytest.raises(KeyError):
        od.pop("xyz")
    assertEqual(len(od), 0)
    assertEqual(od.pop(k, 12345), 12345)

    # make sure pop still works when __missing__ is defined
    class Missing(OrderedDict):
        def __missing__(self, key):
            return 0

    m = Missing(a=1)
    assertEqual(m.pop("b", 5), 5)
    assertEqual(m.pop("a", 6), 1)
    assertEqual(m.pop("a", 6), 6)
    assertEqual(m.pop("a", default=6), 6)
    with pytest.raises(KeyError):
        m.pop("a")


def test_equality():
    OrderedDict = SyOrderedDict
    pairs = [("c", 1), ("b", 2), ("a", 3), ("d", 4), ("e", 5), ("f", 6)]
    shuffle(pairs)
    od1 = OrderedDict(pairs)
    od2 = OrderedDict(pairs)
    assertEqual(od1, od2)  # same order implies equality
    pairs = pairs[2:] + pairs[:2]
    od2 = OrderedDict(pairs)
    assertNotEqual(od1, od2)  # different order implies inequality
    # comparison to regular dict is not order sensitive
    assertEqual(od1, dict(od2))
    assertEqual(dict(od2), od1)
    # different length implied inequality
    assertNotEqual(od1, OrderedDict(pairs[:-1]))


def test_copying():
    OrderedDict = SyOrderedDict
    # Check that ordered dicts are copyable, deepcopyable, picklable,
    # and have a repr/eval round-trip
    pairs = [("c", 1), ("b", 2), ("a", 3), ("d", 4), ("e", 5), ("f", 6)]
    od = OrderedDict(pairs)

    def check(dup):
        assert dup is not od
        assertEqual(dup, od)
        assertEqual(list(dup.items()), list(od.items()))
        assertEqual(len(dup), len(od))
        assertEqual(type(dup), type(od))

    check(od.copy())
    check(copy.copy(od))
    check(copy.deepcopy(od))

    check(eval(repr(od)))
    update_test = OrderedDict()
    update_test.update(od)
    check(update_test)
    check(OrderedDict(od))


def test_yaml_linkage():
    OrderedDict = SyOrderedDict
    # Verify that __reduce__ is setup in a way that supports PyYAML's dump() feature.
    # In yaml, lists are native but tuples are not.
    pairs = [("c", 1), ("b", 2), ("a", 3), ("d", 4), ("e", 5), ("f", 6)]
    od = OrderedDict(pairs)
    # yaml.dump(od) -->
    # '!!python/object/apply:__main__.OrderedDict\n- - [a, 1]\n  - [b, 2]\n'
    assert all(type(pair) == list for pair in od.__reduce__()[1])


def test_reduce_not_too_fat():
    OrderedDict = SyOrderedDict
    # do not save instance dictionary if not needed
    pairs = [("c", 1), ("b", 2), ("a", 3), ("d", 4), ("e", 5), ("f", 6)]
    od = OrderedDict(pairs)
    assert isinstance(od.__dict__, dict)

    res = od.__reduce__()[2]
    del res["_id"]
    assertEqual(res, {})

    od.x = 10
    assertEqual(od.__dict__["x"], 10)
    assertEqual(od.__reduce__()[2], {"x": 10})


def test_pickle_recursive():
    OrderedDict = SyOrderedDict
    od = OrderedDict()
    od[1] = od

    # pickle directly pulls the module, so we have to fake it
    for proto in range(-1, pickle.HIGHEST_PROTOCOL + 1):
        dup = pickle.loads(pickle.dumps(od, proto))
        assert dup is not od
        assertEqual(list(dup.keys()), [1])
        assert dup[1], dup


def test_repr():
    OrderedDict = SyOrderedDict
    od = OrderedDict([("c", 1), ("b", 2), ("a", 3), ("d", 4), ("e", 5), ("f", 6)])
    assertEqual(
        repr(od),
        "OrderedDict([('c', 1), ('b', 2), ('a', 3), ('d', 4), ('e', 5), ('f', 6)])",
    )
    assertEqual(eval(repr(od)), od)
    assertEqual(repr(OrderedDict()), "OrderedDict()")


def test_repr_recursive():
    OrderedDict = SyOrderedDict
    # See issue #9826
    od = OrderedDict.FromKeys("abc")
    od["x"] = od
    assertEqual(
        repr(od),
        f"OrderedDict([('a', {repr(SyNone)}), ('b', {repr(SyNone)}), ('c', {repr(SyNone)}), ('x', ...)])",
    )


def test_repr_recursive_values():
    OrderedDict = SyOrderedDict
    od = OrderedDict()
    od[42] = od.values()
    r = repr(od)
    # Cannot perform a stronger test, as the contents of the repr
    # are implementation-dependent.  All we can say is that we
    # want a str result, not an exception of any sort.
    assert isinstance(r, str)
    od[42] = od.items()
    r = repr(od)
    # Again.
    assert isinstance(r, str)


def test_setdefault():
    OrderedDict = SyOrderedDict
    pairs = [("c", 1), ("b", 2), ("a", 3), ("d", 4), ("e", 5), ("f", 6)]
    shuffle(pairs)
    od = OrderedDict(pairs)
    pair_order = list(od.items())
    assertEqual(od.setdefault("a", 10), 3)
    # make sure order didn't change
    assertEqual(list(od.items()), pair_order)
    assertEqual(od.setdefault("x", 10), 10)
    # make sure 'x' is added to the end
    assertEqual(list(od.items())[-1], ("x", 10))
    assertEqual(od.setdefault("g", default=9), 9)

    # make sure setdefault still works when __missing__ is defined
    class Missing(OrderedDict):
        def __missing__(self, key):
            return 0

    assertEqual(Missing().setdefault(5, 9), 9)


def test_reinsert():
    OrderedDict = SyOrderedDict
    # Given insert a, insert b, delete a, re-insert a,
    # verify that a is now later than b.
    od = OrderedDict()
    od["a"] = 1
    od["b"] = 2
    del od["a"]
    assertEqual(list(od.items()), [("b", 2)])
    od["a"] = 1
    assertEqual(list(od.items()), [("b", 2), ("a", 1)])


def test_move_to_end():
    OrderedDict = SyOrderedDict
    od = OrderedDict.FromKeys("abcde")
    assertEqual(list(od), list("abcde"))
    od.move_to_end("c")
    assertEqual(list(od), list("abdec"))
    od.move_to_end("c", 0)
    assertEqual(list(od), list("cabde"))
    od.move_to_end("c", 0)
    assertEqual(list(od), list("cabde"))
    od.move_to_end("e")
    assertEqual(list(od), list("cabde"))
    od.move_to_end("b", last=False)
    assertEqual(list(od), list("bcade"))
    with pytest.raises(KeyError):
        od.move_to_end("x")
    with pytest.raises(KeyError):
        od.move_to_end("x", 0)


def test_move_to_end_issue25406():
    OrderedDict = SyOrderedDict
    od = OrderedDict.FromKeys("abc")
    od.move_to_end("c", last=False)
    assertEqual(list(od), list("cab"))
    od.move_to_end("a", last=False)
    assertEqual(list(od), list("acb"))

    od = OrderedDict.FromKeys("abc")
    od.move_to_end("a")
    assertEqual(list(od), list("bca"))
    od.move_to_end("c")
    assertEqual(list(od), list("bac"))


def test_sizeof():
    OrderedDict = SyOrderedDict
    # Wimpy test: Just verify the reported size is larger than a regular dict
    d = dict(a=1)
    od = OrderedDict(**d)
    assert sys.getsizeof(od) > sys.getsizeof(d)


def test_views():
    OrderedDict = SyOrderedDict
    # See http://bugs.python.org/issue24286
    s = "the quick brown fox jumped over a lazy dog yesterday before dawn".split()
    od = OrderedDict.FromKeys(s)
    assertEqual(list(od.keys()), list(dict(od).keys()))
    assertEqual(list(od.items()), list(dict(od).items()))


def test_override_update():
    OrderedDict = SyOrderedDict
    # Verify that subclasses can override update() without breaking __init__()

    class MyOD(OrderedDict):
        def update(self, *args, **kwds):
            raise Exception()

    items = [("a", 1), ("c", 3), ("b", 2)]
    assertEqual(list(MyOD(items).items()), items)


def test_highly_nested():
    # Issues 25395 and 35983: test that the trashcan mechanism works
    # correctly for OrderedDict: deleting a highly nested OrderDict
    # should not crash Python.
    OrderedDict = SyOrderedDict
    obj = None
    for _ in range(1000):
        obj = OrderedDict([(None, obj)])
    del obj
    gc.collect()


def test_highly_nested_subclass():
    # Issues 25395 and 35983: test that the trashcan mechanism works
    # correctly for OrderedDict: deleting a highly nested OrderDict
    # should not crash Python.
    OrderedDict = SyOrderedDict
    deleted = []

    class MyOD(OrderedDict):
        def __del__(self):
            deleted.append(self.i)

    obj = None
    for i in range(100):
        obj = MyOD([(None, obj)])
        obj.i = i
    del obj
    gc.collect()
    assertEqual(deleted, list(reversed(range(100))))


def test_delitem_hash_collision():
    OrderedDict = SyOrderedDict

    class Key:
        def __init__(self, hash):
            self._hash = hash
            self.value = str(id(self))

        def __hash__(self):
            return self._hash

        def __eq__(self, other):
            try:
                return self.value == other.value
            except AttributeError:
                return False

        def __repr__(self):
            return self.value

    def blocking_hash(hash):
        # See the collision-handling in lookdict (in Objects/dictobject.c).
        MINSIZE = 8
        i = hash & MINSIZE - 1
        return (i << 2) + i + hash + 1

    COLLIDING = 1

    key = Key(COLLIDING)
    colliding = Key(COLLIDING)
    blocking = Key(blocking_hash(COLLIDING))

    od = OrderedDict()
    od[key] = ...
    od[blocking] = ...
    od[colliding] = ...
    od["after"] = ...

    del od[blocking]
    del od[colliding]
    assertEqual(list(od.items()), [(key, ...), ("after", ...)])


def test_issue24347():
    OrderedDict = SyOrderedDict

    class Key:
        def __hash__(self):
            return randrange(100000)

    od = OrderedDict()
    for i in range(100):
        key = Key()
        od[key] = i

    # These should not crash.
    with pytest.raises(RuntimeError):
        list(od.values())
    with pytest.raises(RuntimeError):
        list(od.items())
    with pytest.raises(RuntimeError):
        repr(od)
    with pytest.raises(KeyError):
        od.copy()


def test_issue24348():
    OrderedDict = SyOrderedDict

    class Key:
        def __hash__(self):
            return 1

    od = OrderedDict()
    od[Key()] = 0
    # This should not crash.
    od.popitem()


def test_issue24667():
    """
    dict resizes after a certain number of insertion operations,
    whether or not there were deletions that freed up slots in the
    hash table.  During fast node lookup, OrderedDict must correctly
    respond to all resizes, even if the current "size" is the same
    as the old one.  We verify that here by forcing a dict resize
    on a sparse odict and then perform an operation that should
    trigger an odict resize (e.g. popitem).  One key aspect here is
    that we will keep the size of the odict the same at each popitem
    call.  This verifies that we handled the dict resize properly.
    """
    OrderedDict = SyOrderedDict

    od = OrderedDict()
    for c0 in "0123456789ABCDEF":
        for c1 in "0123456789ABCDEF":
            if len(od) == 4:
                # This should not raise a KeyError.
                od.popitem(last=False)
            key = c0 + c1
            od[key] = key


# Direct use of dict methods


def test_dict_setitem():
    OrderedDict = SyOrderedDict
    od = OrderedDict()
    dict.__setitem__(od, "spam", 1)
    assertNotIn("NULL", repr(od))


def test_dict_delitem():
    OrderedDict = SyOrderedDict
    od = OrderedDict()
    od["spam"] = 1
    od["ham"] = 2
    dict.__delitem__(od, "spam")
    with pytest.raises(RuntimeError):
        repr(od)


def test_dict_clear():
    OrderedDict = SyOrderedDict
    od = OrderedDict()
    od["spam"] = 1
    od["ham"] = 2
    dict.clear(od)
    assertNotIn("NULL", repr(od))


def test_dict_pop():
    OrderedDict = SyOrderedDict
    od = OrderedDict()
    od["spam"] = 1
    od["ham"] = 2
    dict.pop(od, "spam")
    with pytest.raises(RuntimeError):
        repr(od)


def test_dict_popitem():
    OrderedDict = SyOrderedDict
    od = OrderedDict()
    od["spam"] = 1
    od["ham"] = 2
    dict.popitem(od)
    with pytest.raises(RuntimeError):
        repr(od)


def test_dict_setdefault():
    OrderedDict = SyOrderedDict
    od = OrderedDict()
    dict.setdefault(od, "spam", 1)
    assertNotIn("NULL", repr(od))


def test_dict_update():
    OrderedDict = SyOrderedDict
    od = OrderedDict()
    dict.update(od, [("spam", 1)])
    assertNotIn("NULL", repr(od))


def test_reference_loop():
    # Issue 25935
    OrderedDict = SyOrderedDict

    class A:
        od = OrderedDict()

    A.od[A] = None
    r = weakref.ref(A)
    del A
    gc.collect()
    assert r() is None


def test_ordered_dict_items_result_gc():
    # bpo-42536: OrderedDict.items's tuple-reuse speed trick breaks the GC's
    # assumptions about what can be untracked. Make sure we re-track result
    # tuples whenever we reuse them.
    it = iter(SyOrderedDict({None: []}).items())
    gc.collect()
    # That GC collection probably untracked the recycled internal result
    # tuple, which is initialized to (None, None). Make sure it's re-tracked
    # when it's mutated and returned from __next__:
    assert gc.is_tracked(next(it))


def test_key_change_during_iteration():
    OrderedDict = SyOrderedDict

    od = OrderedDict.FromKeys("abcde")
    assertEqual(list(od), list("abcde"))
    with pytest.raises(RuntimeError):
        for i, k in enumerate(od):
            od.move_to_end(k)
            assert i < 5
    with pytest.raises(RuntimeError):
        for k in od:
            od["f"] = None
    with pytest.raises(RuntimeError):
        for k in od:
            del od["c"]
    assertEqual(list(od), list("bdeaf"))


def test_weakref_list_is_not_traversed():
    # Check that the weakref list is not traversed when collecting
    # OrderedDict objects. See bpo-39778 for more information.

    gc.collect()

    x = SyOrderedDict()
    x.cycle = x

    cycle = []
    cycle.append(cycle)

    x_ref = weakref.ref(x)
    cycle.append(x_ref)

    del x, cycle, x_ref

    gc.collect()
