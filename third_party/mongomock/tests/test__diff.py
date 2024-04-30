# stdlib
from unittest import TestCase

# third party
from tests.diff import diff


class DiffTest(TestCase):
    def test__assert_no_diff(self):
        for obj in [
            1,
            "string",
            {"complex": {"object": {"with": ["lists"]}}},
        ]:
            self.assertEqual(diff(obj, obj), [])

    def test__diff_values(self):
        self._assert_entire_diff(1, 2)
        self._assert_entire_diff("a", "b")

    def test__diff_sequences(self):
        self._assert_entire_diff([], [1, 2, 3])

    def test__composite_diff(self):
        a = {"a": {"b": [1, 2, 3]}}
        b = {"a": {"b": [1, 6, 3]}}
        [(path, x, y)] = diff(a, b)
        self.assertEqual(path, ["a", "b", 1])
        self.assertEqual(x, 2)
        self.assertEqual(y, 6)

    def _assert_entire_diff(self, a, b):
        [(_, x, y)] = diff(a, b)
        self.assertEqual(x, a)
        self.assertEqual(y, b)
