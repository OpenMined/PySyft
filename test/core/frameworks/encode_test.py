from unittest import TestCase

from syft import VirtualWorker
from syft.core.frameworks.encode import PythonEncoder
import syft as sy

from syft.core.frameworks.numpy import array


class TestPythonEncoder(TestCase):
    def setUp(self):
        self.cut = PythonEncoder()

    def test_encode_int(self):
        # Given
        obj = 12
        expected = {"mode": "subscribe", "obj": obj}

        # When
        result = self.cut.encode(obj)

        # Then
        self.assertDictEqual(expected, result)

    def test_encode_float(self):
        # Given
        obj = 12.12
        expected = {"mode": "subscribe", "obj": obj}

        # When
        result = self.cut.encode(obj)

        # Then
        self.assertDictEqual(expected, result)

    def test_encode_str(self):
        # Given
        obj = "I am a scientist - I seek to understand me"
        expected = {"mode": "subscribe", "obj": obj}

        # When
        result = self.cut.encode(obj)

        # Then
        self.assertDictEqual(expected, result)

    def test_encode_dictionary(self):
        # Given
        obj = {"key1": 1, "key2": "some_value"}
        expected = {"mode": "subscribe", "obj": obj}

        # When
        result = self.cut.encode(obj)

        # Then
        self.assertDictEqual(expected, result)

    def test_encode_list(self):
        # Given
        obj = [1, "some_value"]
        expected = {"mode": "subscribe", "obj": obj}

        # When
        result = self.cut.encode(obj)

        # Then
        self.assertDictEqual(expected, result)

    def test_encode_tuple(self):
        # Given
        obj = (1, 2)
        expected = {"mode": "subscribe", "obj": {"__tuple__": [1, 2]}}

        # When
        result = self.cut.encode(obj)

        # Then
        self.assertDictEqual(expected, result)

    def test_encode_set(self):
        # Given
        obj = {1, 2}
        expected = {"mode": "subscribe", "obj": {"__set__": [1, 2]}}

        # When
        result = self.cut.encode(obj)

        # Then
        self.assertDictEqual(expected, result)

    def test_encode_bytearray(self):
        # Given
        obj = bytearray([0x13, 0x00, 0x00, 0x00, 0x08, 0x00])
        expected = {
            "mode": "subscribe",
            "obj": {"__bytearray__": [0x13, 0x00, 0x00, 0x00, 0x08, 0x00]},
        }

        # When
        result = self.cut.encode(obj)

        # Then
        self.assertDictEqual(expected, result)

    def test_encode_range(self):
        # Given
        obj = range(1, 5)
        expected = {"mode": "subscribe", "obj": {"__range__": [1, 2, 3, 4]}}

        # When
        result = self.cut.encode(obj)

        # Then
        self.assertDictEqual(expected, result)

    def test_encode_SyftTensor(self):
        # Given
        owner = VirtualWorker(id="me")
        obj = sy._LocalTensor(owner=owner, id="id", torch_type="syft.FloatTensor")
        expected = {
            "mode": "subscribe",
            "obj": {
                "___LocalTensor__": {
                    "id": "id",
                    "owner": "me",
                    "torch_type": "syft.FloatTensor",
                }
            },
        }

        # When
        result = self.cut.encode(obj)

        # Then
        self.assertEqual(expected, result)

    def test_encode_slice(self):
        # Given
        obj = slice(1, 5, 2)
        expected = {"mode": "subscribe", "obj": {"__slice__": {"args": [1, 5, 2]}}}

        # When
        result = self.cut.encode(obj)

        # Then
        self.assertEqual(expected, result)

    def test_encode_ellipsis(self):
        # Given
        obj = ...
        expected = {"mode": "subscribe", "obj": "..."}

        # When
        result = self.cut.encode(obj)

        # Then
        self.assertEqual(expected, result)

    def test_encode_generator(self):
        def _firstn(n):
            num = 0
            while num < n:
                yield num
                num += 1

        # Given
        obj = _firstn(2)
        expected = {"mode": "subscribe", "obj": []}

        # When
        result = self.cut.encode(obj)

        # Then
        self.assertEqual(expected, result)

    def test_encode_variable(self):
        # Given
        data = array([1, 2], id=123).torch()
        obj = sy.Variable(data, requires_grad=False)
        obj.child = None
        expected = {
            "mode": "subscribe",
            "obj": {
                "__Variable__": {
                    "child": {
                        "___LocalTensor__": {
                            "id": 76_308_044_977,
                            "owner": "me",
                            "torch_type": "Variable",
                        }
                    },
                    "data": {
                        "__FloatTensor__": {
                            "child": {
                                "___LocalTensor__": {
                                    "id": 32_071_180_896,
                                    "owner": "me",
                                    "torch_type": "FloatTensor",
                                }
                            },
                            "data": [],
                            "torch_type": "FloatTensor",
                        }
                    },
                    "grad": {
                        "__Variable__": {
                            "child": {
                                "___LocalTensor__": {
                                    "id": 77_824_091_007,
                                    "owner": "me",
                                    "torch_type": "Variable",
                                }
                            },
                            "data": {
                                "__FloatTensor__": {
                                    "child": {
                                        "___LocalTensor__": {
                                            "id": 32_100_939_892,
                                            "owner": "me",
                                            "torch_type": "FloatTensor",
                                        }
                                    },
                                    "data": [],
                                    "torch_type": "FloatTensor",
                                }
                            },
                            "requires_grad": False,
                            "torch_type": "Variable",
                        }
                    },
                    "requires_grad": False,
                    "torch_type": "Variable",
                }
            },
        }

        # When
        result = self.cut.encode(obj)

        # Then
        # Reducing the scope of the test as we cannot keep control on certain values (i.e. `id`)
        self.assertEqual(
            expected["obj"]["__Variable__"]["torch_type"],
            result["obj"]["__Variable__"]["torch_type"],
        )

    def test_encode_tensor(self):
        # Given
        obj = array([1, 2], id=123).torch()
        expected = {
            "mode": "subscribe",
            "obj": {
                "__FloatTensor__": {
                    "child": {
                        "___LocalTensor__": {
                            "id": 73_108_883_601,
                            "owner": "me",
                            "torch_type": "FloatTensor",
                        }
                    },
                    "data": [],
                    "torch_type": "FloatTensor",
                }
            },
        }

        # When
        result = self.cut.encode(obj)

        # Then
        self.assertEqual(expected["mode"], result["mode"])
        self.assertEqual(
            expected["obj"]["__FloatTensor__"]["torch_type"],
            result["obj"]["__FloatTensor__"]["torch_type"],
        )

    def test_encode_nparray(self):
        # Given
        obj = array([1, 2], id=123)
        expected = {
            "mode": "subscribe",
            "obj": {"data": [], "id": 123, "owner": "me", "type": "numpy.array"},
        }

        # When
        result = self.cut.encode(obj)

        # Then
        self.assertDictEqual(expected, result)

    def test_encode_socketWorker(self):
        # Given
        obj = sy.SocketWorker()
        expected = {"mode": "subscribe", "obj": {"__worker__": 0}}

        # When
        result = self.cut.encode(obj)

        # Then
        self.assertDictEqual(expected, result)

    def test_encode_virtualWorker(self):
        # Given
        obj = sy.VirtualWorker()
        expected = {"mode": "subscribe", "obj": {"__worker__": 0}}

        # When
        result = self.cut.encode(obj)

        # Then
        self.assertDictEqual(expected, result)

    def test_encode_unhandled_type(self):
        # Given
        class MyClass:
            pass

        obj = MyClass()

        # When
        # Then
        self.assertRaises(ValueError, lambda: self.cut.encode(obj))

    def test_encode_non_private_local(self):
        # Given
        obj = array([1, 2], id=123)
        expected = {
            "mode": "acquire",
            "obj": {"data": [1, 2], "id": 123, "owner": "me", "type": "numpy.array"},
        }

        # When
        result = self.cut.encode(obj, private_local=False)

        # Then
        self.assertDictEqual(expected, result)

    def test_encode_retrieve_pointers(self):
        # Given
        obj = array([1, 2], id=123)
        expected = (
            {
                "mode": "subscribe",
                "obj": {"data": [], "id": 123, "owner": "me", "type": "numpy.array"},
            },
            [],
        )

        # When
        result = self.cut.encode(obj, retrieve_pointers=True)

        # Then
        self.assertTupleEqual(expected, result)
