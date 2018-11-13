from syft.serde import _simplify
from unittest import TestCase


class TupleSerde(TestCase):
    def test_tuple_simplify(self):
        input = ("hello", "world")
        target = ("hello", "world")
        assert _simplify(input) == target

    def test_list_simplify(self):
        input = ["hello", "world"]
        target = ["hello", "world"]
        assert _simplify(input) == target

    def test_set_simplify(self):
        input = set(["hello", "world"])
        target = set(["hello", "world"])
        assert _simplify(input) == target

    def test_float_simplify(self):
        input = 5.6
        target = 5.6
        assert _simplify(input) == target

    def test_int_simplify(self):
        input = 5
        target = 5
        assert _simplify(input) == target

    def test_string_simplify(self):
        input = "hello"
        target = "hello"
        assert _simplify(input) == target
