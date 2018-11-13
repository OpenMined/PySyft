import syft
from unittest import TestCase

class TupleSerde(TestCase):

    def test_tuple_serialize(self):
        input = ('hello', 'world')
        target = ['hello', 'world']
        assert syft.serde.simplify(input) == target
