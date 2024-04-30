# stdlib
import json
import os
from unittest import TestCase

# third party
from mongomock.helpers import get_value_by_dot
from mongomock.helpers import hashdict
from mongomock.helpers import parse_uri
from mongomock.helpers import print_deprecation_warning
from mongomock.helpers import set_value_by_dot


class HashdictTest(TestCase):
    def test__hashdict(self):
        """Make sure hashdict can be used as a key for a dict"""
        h = {}
        _id = hashdict({"a": 1})
        h[_id] = "foo"
        self.assertEqual(h[_id], "foo")
        _id = hashdict({"a": {"foo": 2}})
        h[_id] = "foo"
        self.assertEqual(h[_id], "foo")
        _id = hashdict({"a": {"foo": {"bar": 3}}})
        h[_id] = "foo"
        self.assertEqual(h[_id], "foo")
        _id = hashdict({hashdict({"a": "3"}): {"foo": 2}})
        h[_id] = "foo"
        self.assertEqual(h[_id], "foo")

        with self.assertRaises(TypeError):
            _id["a"] = 2
        with self.assertRaises(TypeError):
            del _id["a"]
        with self.assertRaises(TypeError):
            _id.clear()
        with self.assertRaises(TypeError):
            _id.pop("a")
        with self.assertRaises(TypeError):
            _id.popitem("a")
        with self.assertRaises(TypeError):
            _id.setdefault("c", 3)
        with self.assertRaises(TypeError):
            _id.update({"b": 2, "c": 4})

        self.assertEqual(
            hashdict({"a": 1, "b": 3, "c": 4}),
            hashdict({"a": 1, "b": 2}) + hashdict({"b": 3, "c": 4}),
        )

        self.assertEqual("hashdict(a=1, b=2)", repr(hashdict({"a": 1, "b": 2})))


class TestDeprecationWarning(TestCase):
    def test__deprecation_warning(self):
        # ensure this doesn't throw an exception
        print_deprecation_warning("aaa", "bbb")


class TestAllUriScenarios(TestCase):
    pass


_URI_SPEC_TEST_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    os.path.join("connection_string", "test"),
)


def create_uri_spec_tests():
    """Use json specifications in `_TEST_PATH` to generate uri spec tests.

    This is a simplified version from the PyMongo "test/test_uri_spec.py". It
    is modified to disregard warnings and only check that valid uri's are valid
    with the correct database.
    """

    def create_uri_spec_test(scenario_def):
        def run_scenario(self):
            self.assertTrue(scenario_def["tests"], "tests cannot be empty")
            for test in scenario_def["tests"]:
                dsc = test["description"]

                error = False

                try:
                    dbase = parse_uri(test["uri"])["database"]
                except Exception as e:
                    print(e)
                    error = True

                self.assertEqual(not error, test["valid"], "Test failure '%s'" % dsc)

                # Compare auth options.
                auth = test["auth"]
                if auth is not None:
                    expected_dbase = auth.pop("db")  # db == database
                    # Special case for PyMongo's collection parsing
                    if expected_dbase and "." in expected_dbase:
                        expected_dbase, _ = expected_dbase.split(".", 1)
                    self.assertEqual(
                        expected_dbase,
                        dbase,
                        "Expected %s but got %s" % (expected_dbase, dbase),
                    )

        return run_scenario

    for dirpath, _, filenames in os.walk(_URI_SPEC_TEST_PATH):
        dirname = os.path.split(dirpath)
        dirname = os.path.split(dirname[-2])[-1] + "_" + dirname[-1]

        for filename in filenames:
            with open(os.path.join(dirpath, filename)) as scenario_stream:
                scenario_def = json.load(scenario_stream)
            # Construct test from scenario.
            new_test = create_uri_spec_test(scenario_def)
            test_name = "test_%s_%s" % (dirname, os.path.splitext(filename)[0])
            new_test.__name__ = test_name
            setattr(TestAllUriScenarios, new_test.__name__, new_test)


create_uri_spec_tests()


class ValueByDotTest(TestCase):
    def test__get_value_by_dot_missing_key(self):
        """Test get_value_by_dot raises KeyError when looking for a missing key"""
        for doc, key in (
            ({}, "a"),
            ({"a": 1}, "b"),
            ({"a": 1}, "a.b"),
            ({"a": {"b": 1}}, "a.b.c"),
            ({"a": {"b": 1}}, "a.c"),
            ({"a": [{"b": 1}]}, "a.b"),
            ({"a": [{"b": 1}]}, "a.1.b"),
        ):
            self.assertRaises(KeyError, get_value_by_dot, doc, key)

    def test__get_value_by_dot_find_key(self):
        """Test get_value_by_dot when key can be found"""
        for doc, key, expected in (
            ({"a": 1}, "a", 1),
            ({"a": {"b": 1}}, "a", {"b": 1}),
            ({"a": {"b": 1}}, "a.b", 1),
            ({"a": [{"b": 1}]}, "a.0.b", 1),
        ):
            found = get_value_by_dot(doc, key)
            self.assertEqual(found, expected)

    def test__set_value_by_dot(self):
        """Test set_value_by_dot"""
        for doc, key, expected in (
            ({}, "a", {"a": 42}),
            ({"a": 1}, "a", {"a": 42}),
            ({"a": {"b": 1}}, "a", {"a": 42}),
            ({"a": {"b": 1}}, "a.b", {"a": {"b": 42}}),
            ({"a": [{"b": 1}]}, "a.0", {"a": [42]}),
            ({"a": [{"b": 1}]}, "a.0.b", {"a": [{"b": 42}]}),
        ):
            ret = set_value_by_dot(doc, key, 42)
            assert ret is doc
            self.assertEqual(ret, expected)

    def test__set_value_by_dot_bad_key(self):
        """Test set_value_by_dot when key has an invalid parent"""
        for doc, key in (
            ({}, "a.b"),
            ({"a": 1}, "a.b"),
            ({"a": {"b": 1}}, "a.b.c"),
            ({"a": [{"b": 1}]}, "a.1.b"),
            ({"a": [{"b": 1}]}, "a.1"),
        ):
            self.assertRaises(KeyError, set_value_by_dot, doc, key, 42)
