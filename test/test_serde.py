import syft
from unittest import TestCase


class TupleSerde(TestCase):
    def test_tuple_serialize(self):
        input = ("hello", "world")
        target = ("hello", "world")
        assert syft.serde._simplify(input) == target

    def test_list_serialize(self):
        input = ["hello", "world"]
        target = ["hello", "world"]
        assert syft.serde._simplify(input) == target

    def test_set_serialize(self):
        input = set(["hello", "world"])
        target = set(["hello", "world"])
        assert syft.serde._simplify(input) == target
