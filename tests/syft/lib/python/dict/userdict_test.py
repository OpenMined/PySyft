# Check every path through every method of UserDict

# stdlib
# from test import mapping_tests
import unittest

# syft absolute
from syft.lib.python.dict import Dict

d0 = {}
d1 = {"one": 1}
d2 = {"one": 1, "two": 2}
d3 = {"one": 1, "two": 3, "three": 5}
d4 = {"one": None, "two": None}
d5 = {"one": 1, "two": 1}


# class UserDictTest(mapping_tests.TestHashMappingProtocol):
class UserDictTest:
    type2test = Dict

    def test_all(self):
        # Test constructors
        u = Dict()
        u0 = Dict(d0)
        u1 = Dict(d1)
        u2 = Dict(d2)

        uu = Dict(u)
        uu0 = Dict(u0)
        uu1 = Dict(u1)
        uu2 = Dict(u2)

        # keyword arg constructor
        self.assertEqual(Dict(one=1, two=2), d2)
        # item sequence constructor
        self.assertEqual(Dict([("one", 1), ("two", 2)]), d2)
        with self.assertWarnsRegex(DeprecationWarning, "'dict'"):
            left = Dict(dict=[("one", 1), ("two", 2)])
            right = d2
            print("left", left, type(left))
            print("right", right, type(right))
            self.assertEqual(Dict(dict=[("one", 1), ("two", 2)]), d2)
        # both together
        self.assertEqual(Dict([("one", 1), ("two", 2)], two=3, three=5), d3)

        # alternate constructor
        self.assertEqual(Dict.fromkeys("one two".split()), d4)
        self.assertEqual(Dict().fromkeys("one two".split()), d4)
        self.assertEqual(Dict.fromkeys("one two".split(), 1), d5)
        self.assertEqual(Dict().fromkeys("one two".split(), 1), d5)
        self.assertTrue(u1.fromkeys("one two".split()) is not u1)
        self.assertIsInstance(u1.fromkeys("one two".split()), Dict)
        self.assertIsInstance(u2.fromkeys("one two".split()), Dict)

        # Test __repr__
        self.assertEqual(str(u0), str(d0))
        self.assertEqual(repr(u1), repr(d1))
        self.assertIn(repr(u2), ("{'one': 1, 'two': 2}", "{'two': 2, 'one': 1}"))

        # Test rich comparison and __len__
        all = [d0, d1, d2, u, u0, u1, u2, uu, uu0, uu1, uu2]
        for a in all:
            for b in all:
                self.assertEqual(a == b, len(a) == len(b))

        # Test __getitem__
        self.assertEqual(u2["one"], 1)
        self.assertRaises(KeyError, u1.__getitem__, "two")

        # Test __setitem__
        u3 = Dict(u2)
        u3["two"] = 2
        u3["three"] = 3

        # Test __delitem__
        del u3["three"]
        self.assertRaises(KeyError, u3.__delitem__, "three")

        # Test clear
        u3.clear()
        self.assertEqual(u3, {})

        # Test copy()
        u2a = u2.copy()
        self.assertEqual(u2a, u2)
        u2b = Dict(x=42, y=23)
        u2c = u2b.copy()  # making a copy of a UserDict is special cased
        self.assertEqual(u2b, u2c)

        class MyUserDict(Dict):
            def display(self):
                print(self)

        m2 = MyUserDict(u2)
        m2a = m2.copy()
        self.assertEqual(m2a, m2)

        # SF bug #476616 -- copy() of UserDict subclass shared data
        m2["foo"] = "bar"
        self.assertNotEqual(m2a, m2)

        # Test keys, items, values
        self.assertEqual(sorted(u2.keys()), sorted(d2.keys()))
        self.assertEqual(sorted(u2.items()), sorted(d2.items()))
        self.assertEqual(sorted(u2.values()), sorted(d2.values()))

        # Test "in".
        for i in u2.keys():
            self.assertIn(i, u2)
            self.assertEqual(i in u1, i in d1)
            self.assertEqual(i in u0, i in d0)

        # Test update
        t = Dict()
        t.update(u2)
        self.assertEqual(t, u2)

        # Test get
        for i in u2.keys():
            self.assertEqual(u2.get(i), u2[i])
            self.assertEqual(u1.get(i), d1.get(i))
            self.assertEqual(u0.get(i), d0.get(i))

        # Test "in" iteration.
        for i in range(20):
            u2[i] = str(i)
        ikeys = []
        for k in u2:
            ikeys.append(k)
        keys = u2.keys()
        self.assertEqual(set(ikeys), set(keys))

        # Test setdefault
        t = Dict()
        self.assertEqual(t.setdefault("x", 42), 42)
        self.assertIn("x", t)
        self.assertEqual(t.setdefault("x", 23), 42)

        # Test pop
        t = Dict(x=42)
        self.assertEqual(t.pop("x"), 42)
        self.assertRaises(KeyError, t.pop, "x")
        self.assertEqual(t.pop("x", 1), 1)
        t["x"] = 42
        self.assertEqual(t.pop("x", 1), 42)

        # Test popitem
        t = Dict(x=42)
        self.assertEqual(t.popitem(), ("x", 42))
        self.assertRaises(KeyError, t.popitem)

    def test_init(self):
        for kw in "self", "other", "iterable":
            self.assertEqual(list(Dict(**{kw: 42}).items()), [(kw, 42)])

        a = list(Dict({}, dict=42).items())
        b = [("dict", 42)]
        print("left", a, type(a))
        print("right", b, type(b))
        self.assertEqual(list(Dict({}, dict=42).items()), [("dict", 42)])
        self.assertEqual(list(Dict({}, dict=None).items()), [("dict", None)])
        with self.assertWarnsRegex(DeprecationWarning, "'dict'"):
            self.assertEqual(list(Dict(dict={"a": 42}).items()), [("a", 42)])
        self.assertRaises(TypeError, Dict, 42)
        self.assertRaises(TypeError, Dict, (), ())
        self.assertRaises(TypeError, Dict.__init__)

    def test_update(self):
        for kw in "self", "dict", "other", "iterable":
            d = Dict()
            d.update(**{kw: 42})
            self.assertEqual(list(d.items()), [(kw, 42)])
        self.assertRaises(TypeError, Dict().update, 42)
        self.assertRaises(TypeError, Dict().update, {}, {})
        self.assertRaises(TypeError, Dict.update)

    def test_missing(self):
        # Make sure UserDict doesn't have a __missing__ method
        self.assertEqual(hasattr(Dict, "__missing__"), False)
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

        class E(Dict):
            def __missing__(self, key):
                raise RuntimeError(key)

        e = E()
        try:
            e[42]
        except RuntimeError as err:
            self.assertEqual(err.args, (42,))
        else:
            self.fail("e[42] didn't raise RuntimeError")

        class F(Dict):
            def __init__(self):
                # An instance variable __missing__ should have no effect
                self.__missing__ = lambda key: None
                Dict.__init__(self)

        f = F()
        try:
            f[42]
        except KeyError as err:
            self.assertEqual(err.args, (42,))
        else:
            self.fail("f[42] didn't raise KeyError")

        class G(Dict):
            pass

        g = G()
        try:
            g[42]
        except KeyError as err:
            self.assertEqual(err.args, (42,))
        else:
            self.fail("g[42] didn't raise KeyError")


if __name__ == "__main__":
    unittest.main()
