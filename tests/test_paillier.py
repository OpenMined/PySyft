from syft.he.paillier import KeyPair, PaillierTensor
from syft.he.keys import Paillier
from syft import TensorBase
import unittest
import numpy as np
import syft as sy


# Here's our "unit tests".
class DimTests(unittest.TestCase):
    def test_dim_one(self):
        p, s = KeyPair().generate()
        x = PaillierTensor(p, np.array([1, 2, 3, 4, 5.]))
        self.assertTrue(x.dim() == 1)


class DotTests(unittest.TestCase):
    def test_dot_product(self):

        pk, sk = Paillier()

        x = pk.ones(10)
        y = sy.ones(10)

        out1 = y.dot(x).decrypt(sk)
        out2 = x.dot(y).decrypt(sk)

        self.assertEqual(out1, 10)
        self.assertEqual(out2, 10)


class AddTests(unittest.TestCase):

    def test_simple(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([1, 2, 3, 4, 5.]))
        x2 = PaillierTensor(p, np.array([3, 4, 5, 6, 7.]))

        y = (x + x2).decrypt(s)
        self.assertTrue(y == np.array([4., 6., 8., 10., 12.]))

    def test_simple_reversed(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([1, 2, 3, 4, 5.]))
        x2 = PaillierTensor(p, np.array([3, 4, 5, 6, 7.]))

        y = (x2 + x).decrypt(s)
        self.assertTrue(y == np.array([4., 6., 8., 10., 12.]))

    def test_scalar_in_place(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([1, 2, 3, 4, 5.]))

        x += 1
        self.assertTrue(s.decrypt(x) == np.array([2., 3., 4., 5., 6.]))

    def test_in_place(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([1, 2, 3, 4, 5.]))
        x2 = PaillierTensor(p, np.array([3, 4, 5, 6, 7.]))

        x += x2
        self.assertTrue(s.decrypt(x) == np.array([4., 6., 8., 10., 12.]))

    def test_in_place_reversed(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([1, 2, 3, 4, 5.]))
        x2 = PaillierTensor(p, np.array([3, 4, 5, 6, 7.]))

        x2 += x
        self.assertTrue(s.decrypt(x2) == np.array([4., 6., 8., 10., 12.]))

    def test_scalar(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([1, 2, 3, 4, 5.]))

        y = x + 40

        self.assertTrue(s.decrypt(y) == np.array([41., 42., 43., 44., 45.]))

    def test_in_place_plain_text(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([1, 2, 3, 4, 5.]))
        x2 = TensorBase(np.array([3, 4, 5, 6, 7.]))

        x += x2
        self.assertTrue(s.decrypt(x) == np.array([4., 6., 8., 10., 12.]))

    def test_add_depth(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([1, 2, 3, 4, 5.]))
        x2 = TensorBase(np.array([3, 4, 5, 6, 7.]))

        x += x2
        self.assertEqual(x._add_depth, 1)


class MulTests(unittest.TestCase):

    def test_basic(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([1, 2, 3, 4, 5.]))
        x2 = TensorBase(np.array([3, 4, 5, 6, 7.]))

        y = x * x2
        self.assertTrue(y.decrypt(s) == np.array([3., 8., 15., 24., 35.]))

    def test_basic_reversed(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([1, 2, 3, 4, 5.]))
        x2 = TensorBase(np.array([3, 4, 5, 6, 7.]))

        y = x2 * x
        self.assertTrue(y.decrypt(s) == np.array([3., 8., 15., 24., 35.]))

    def test_inline(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([1, 2, 3, 4, 5.]))
        x2 = TensorBase(np.array([3, 4, 5, 6, 7.]))

        x *= x2
        self.assertTrue(x.decrypt(s) == np.array([3., 8., 15., 24., 35.]))

    def test_inline_reversed(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([1, 2, 3, 4, 5.]))
        x2 = TensorBase(np.array([3, 4, 5, 6, 7.]))

        x2 *= x
        self.assertTrue(x2.decrypt(s) == np.array([3., 8., 15., 24., 35.]))

    def test_scalar(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([1, 2, 3, 4, 5.]))

        x *= 2
        self.assertTrue(s.decrypt(x) == np.array([2., 4., 6., 8., 10.]))

    def test_mul_depth(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([1, 2, 3, 4, 5.]))
        x2 = TensorBase(np.array([3, 4, 5, 6, 7.]))

        x *= x2
        self.assertEqual(x._mul_depth, 1)


class DivTests(unittest.TestCase):

    def test_basic(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([3., 8., 15., 24., 35.]))
        x2 = TensorBase(np.array([3, 4, 5, 6, 7.]))

        y = x / x2
        print(y.decrypt(s))
        self.assertTrue(y.decrypt(s) == np.array([1., 2., 3., 4., 5.]))

    def test_inline(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([3., 8., 15., 24., 35.]))
        x2 = TensorBase(np.array([3, 4, 5, 6, 7.]))

        x /= x2
        self.assertTrue(x.decrypt(s) == np.array([1., 2., 3., 4., 5.]))

    def test_scalar(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([2., 4., 6., 8., 10.]))

        x /= 2
        self.assertTrue(s.decrypt(x) == np.array([1, 2, 3, 4, 5.]))


class SerializeTest(unittest.TestCase):

    def test_serialize(self):
        pubkey, seckey = KeyPair().generate()

        pk_serialized = pubkey.serialize()
        sk_serialized = seckey.serialize()

        pubkey2, seckey2 = KeyPair().deserialize(pk_serialized,
                                                 sk_serialized)
        self.assertTrue(pubkey.pk == pubkey2.pk and
                        seckey.sk == seckey2.sk)
