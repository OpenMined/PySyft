from syft.he.paillier import KeyPair, PaillierTensor
from syft.he.keys import Paillier
from syft import TensorBase
import unittest
import numpy as np
import syft as sy


# Here's our "unit tests".
class DimTests(unittest.TestCase):
    def testDimOne(self):
        p, s = KeyPair().generate()
        x = PaillierTensor(p, np.array([1, 2, 3, 4, 5.]))
        self.assertTrue(x.dim() == 1)


class DotTests(unittest.TestCase):
    def testDotProduct(self):

        pk, sk = Paillier()

        x = pk.ones(10)
        y = sy.ones(10)

        out1 = y.dot(x).decrypt(sk)
        out2 = x.dot(y).decrypt(sk)

        self.assertEqual(out1, 10)
        self.assertEqual(out2, 10)


class AddTests(unittest.TestCase):

    def testSimple(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([1, 2, 3, 4, 5.]))
        x2 = PaillierTensor(p, np.array([3, 4, 5, 6, 7.]))

        y = (x + x2).decrypt(s)
        self.assertTrue(y == np.array([4., 6., 8., 10., 12.]))

    def testSimpleReversed(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([1, 2, 3, 4, 5.]))
        x2 = PaillierTensor(p, np.array([3, 4, 5, 6, 7.]))

        y = (x2 + x).decrypt(s)
        self.assertTrue(y == np.array([4., 6., 8., 10., 12.]))

    def testScalarInplace(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([1, 2, 3, 4, 5.]))

        x += 1
        self.assertTrue(s.decrypt(x) == np.array([2., 3., 4., 5., 6.]))

    def testInplace(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([1, 2, 3, 4, 5.]))
        x2 = PaillierTensor(p, np.array([3, 4, 5, 6, 7.]))

        x += x2
        self.assertTrue(s.decrypt(x) == np.array([4., 6., 8., 10., 12.]))

    def testInplaceReversed(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([1, 2, 3, 4, 5.]))
        x2 = PaillierTensor(p, np.array([3, 4, 5, 6, 7.]))

        x2 += x
        self.assertTrue(s.decrypt(x2) == np.array([4., 6., 8., 10., 12.]))

    def testScalar(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([1, 2, 3, 4, 5.]))

        y = x + 40

        self.assertTrue(s.decrypt(y) == np.array([41., 42., 43., 44., 45.]))

    def testInplacePlaintext(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([1, 2, 3, 4, 5.]))
        x2 = TensorBase(np.array([3, 4, 5, 6, 7.]))

        x += x2
        self.assertTrue(s.decrypt(x) == np.array([4., 6., 8., 10., 12.]))

    def testAddDepth(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([1, 2, 3, 4, 5.]))
        x2 = TensorBase(np.array([3, 4, 5, 6, 7.]))

        x += x2
        self.assertEqual(x._add_depth, 1)


class MulTests(unittest.TestCase):

    def testBasic(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([1, 2, 3, 4, 5.]))
        x2 = TensorBase(np.array([3, 4, 5, 6, 7.]))

        y = x * x2
        self.assertTrue(y.decrypt(s) == np.array([3., 8., 15., 24., 35.]))

    def testBasicReversed(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([1, 2, 3, 4, 5.]))
        x2 = TensorBase(np.array([3, 4, 5, 6, 7.]))

        y = x2 * x
        self.assertTrue(y.decrypt(s) == np.array([3., 8., 15., 24., 35.]))

    def testInline(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([1, 2, 3, 4, 5.]))
        x2 = TensorBase(np.array([3, 4, 5, 6, 7.]))

        x *= x2
        self.assertTrue(x.decrypt(s) == np.array([3., 8., 15., 24., 35.]))

    def testInlineReversed(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([1, 2, 3, 4, 5.]))
        x2 = TensorBase(np.array([3, 4, 5, 6, 7.]))

        x2 *= x
        self.assertTrue(x2.decrypt(s) == np.array([3., 8., 15., 24., 35.]))

    def testScalar(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([1, 2, 3, 4, 5.]))

        x *= 2
        self.assertTrue(s.decrypt(x) == np.array([2., 4., 6., 8., 10.]))

    def testMulDepth(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([1, 2, 3, 4, 5.]))
        x2 = TensorBase(np.array([3, 4, 5, 6, 7.]))

        x *= x2
        self.assertEqual(x._mul_depth, 1)


class DivTests(unittest.TestCase):

    def testBasic(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([3., 8., 15., 24., 35.]))
        x2 = TensorBase(np.array([3, 4, 5, 6, 7.]))

        y = x / x2
        print(y.decrypt(s))
        self.assertTrue(y.decrypt(s) == np.array([1., 2., 3., 4., 5.]))

    def testInline(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([3., 8., 15., 24., 35.]))
        x2 = TensorBase(np.array([3, 4, 5, 6, 7.]))

        x /= x2
        self.assertTrue(x.decrypt(s) == np.array([1., 2., 3., 4., 5.]))

    def testScalar(self):
        p, s = KeyPair().generate()

        x = PaillierTensor(p, np.array([2., 4., 6., 8., 10.]))

        x /= 2
        self.assertTrue(s.decrypt(x) == np.array([1, 2, 3, 4, 5.]))
