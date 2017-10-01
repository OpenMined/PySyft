import unittest

import numpy as np

import syft
from syft import TensorBase


class ConvenienceTests(unittest.TestCase):
    def testZeros(self):
        self.assertTrue((syft.zeros(5).data == np.zeros(5)).all())

    def testOnes(self):
        self.assertTrue((syft.ones(5).data == np.ones(5)).all())

    def testRand(self):
        np.random.seed(0)
        x = syft.rand(5).data
        np.random.seed(0)
        y = np.random.rand(5)
        self.assertTrue((x == y).all())


class DotTests(unittest.TestCase):
    def testDotInt(self):
        t1 = TensorBase(np.array([1, 2, 3]))
        t2 = TensorBase(np.array([4, 5, 6]))
        self.assertEqual(syft.dot(t1, t2), 32)

    def testDotFloat(self):
        t1 = TensorBase(np.array([1.3, 2.5, 3.7]))
        t2 = TensorBase(np.array([4.9, 5.8, 6.5]))
        self.assertEqual(syft.dot(t1, t2), 44.92)


class CeilTests(unittest.TestCase):
    def testCeil(self):
        t1 = TensorBase(np.array([[2.3, 4.1], [7.4, 8.3]]))
        self.assertTrue(syft.equal(syft.ceil(t1), TensorBase([[3., 5.],
                                                              [8., 9.]])))


class FloorTests(unittest.TestCase):
    def testFloor(self):
        t1 = TensorBase(np.array([[2.3, 4.1], [7.4, 8.3]]))
        self.assertTrue(syft.equal(syft.math.floor(t1), TensorBase([[2., 4.],
                                                                    [7., 8.]])))


class CumsumTests(unittest.TestCase):
    def testCumsum(self):
        t1 = TensorBase(np.array([1, 2, 3]))
        self.assertTrue(syft.equal(syft.cumsum(t1), TensorBase([1, 3, 6])))


class CumprodTests(unittest.TestCase):
    """Cumultative Product test"""

    def testCumprod(self):
        t1 = TensorBase(np.array([1, 2, 3]))
        self.assertTrue(syft.equal(syft.cumprod(t1), TensorBase([1, 2, 6])))


class SigmoidTests(unittest.TestCase):
    """Sigmoid Test"""

    def testSigmoid(self):
        t1 = TensorBase(np.array([1.2, 3.3, 4]))
        self.assertTrue(syft.equal(syft.math.sigmoid(t1), TensorBase(
            [0.76852478, 0.96442881, 0.98201379])))


class MatmulTests(unittest.TestCase):
    """Matmul Tests"""

    def testMatmul1DInt(self):
        t1 = TensorBase(np.array([1, 2, 3]))
        t2 = TensorBase(np.array([4, 5, 6]))
        self.assertEqual(syft.matmul(t1, t2), syft.dot(t1, t2))

    def testMatmul1DFloat(self):
        t1 = TensorBase(np.array([1.3, 2.5, 3.7]))
        t2 = TensorBase(np.array([4.9, 5.8, 6.5]))
        self.assertEqual(syft.matmul(t1, t2), syft.dot(t1, t2))

    def testMatmul2DIdentity(self):
        t1 = TensorBase(np.array([[1, 0],
                                  [0, 1]]))
        t2 = TensorBase(np.array([[5.8, 6.5],
                                  [7.8, 8.9]]))
        self.assertTrue(syft.equal(syft.matmul(t1, t2), [[5.8, 6.5],
                                                         [7.8, 8.9]]))

    def testMatmul2DInt(self):
        t1 = TensorBase(np.array([[1, 2],
                                  [3, 4]]))
        t2 = TensorBase(np.array([[5, 6],
                                  [7, 8]]))
        self.assertTrue(syft.equal(syft.matmul(t1, t2), [[19, 22],
                                                         [43, 50]]))

    def testMatmul2DFloat(self):
        t1 = TensorBase(np.array([[1.3, 2.5],
                                  [3.4, 4.5]]))
        t2 = TensorBase(np.array([[5.8, 6.5],
                                  [7.8, 8.9]]))
        self.assertTrue(syft.equal(syft.matmul(t1, t2), [[27.04, 30.7],
                                                         [54.82, 62.15]]))


class admmTests(unittest.TestCase):
    def testaddmm1d(self):
        t1 = TensorBase(np.array([1, 2, 3]))
        t2 = TensorBase(np.array([2, 3, 4]))
        mat = TensorBase(np.array([5]))
        out = syft.addmm(t1, t2, mat, beta=2, alpha=2)
        self.assertTrue(np.array_equal(out.data, [50]))

    def testaddmm2d(self):
        t1 = TensorBase(np.array([[1, 2], [1, 2]]))
        t2 = TensorBase(np.array([[1, 2], [1, 2]]))
        mat = TensorBase(np.array([[2, 3], [3, 4]]))
        out = syft.addmm(t1, t2, mat, beta=2, alpha=2)
        self.assertTrue(np.array_equal(out.data, [[10, 18], [12, 20]]))


class addcmulTests(unittest.TestCase):
    def testaddcmul1d(self):
        t1 = TensorBase(np.array([1, 2, 3]))
        t2 = TensorBase(np.array([2, 3, 4]))
        mat = TensorBase(np.array([5]))
        out = syft.addcmul(t1, t2, mat, value=2)
        self.assertTrue(np.array_equal(out.data, [9, 17, 29]))

    def testaddcmul2d(self):
        t1 = TensorBase(np.array([[1, 2], [1, 2]]))
        t2 = TensorBase(np.array([[1, 2], [1, 2]]))
        mat = TensorBase(np.array([[2, 3], [3, 4]]))
        out = syft.addcmul(t1, t2, mat, value=2)
        self.assertTrue(np.array_equal(out.data, [[4, 11], [5, 12]]))


class addcdivTests(unittest.TestCase):
    def testaddcdiv1d(self):
        t1 = TensorBase(np.array([1, 2, 3]))
        t2 = TensorBase(np.array([2, 5, 4]))
        mat = TensorBase(np.array([5]))
        out = syft.addcdiv(t1, t2, mat, value=2)
        self.assertTrue(np.array_equal(out.data, [6., 5.8, 6.5]))

    def testaddcdiv2d(self):
        t1 = TensorBase(np.array([[1, 2], [1, 2]]))
        t2 = TensorBase(np.array([[1, 2], [1, 2]]))
        mat = TensorBase(np.array([[2, 3], [3, 4]]))
        out = syft.addcdiv(t1, t2, mat, value=2)
        self.assertTrue(np.array_equal(out.data, [[4., 5.], [5., 6.]]))


class addmv(unittest.TestCase):
    def testaddmv(self):
        t1 = TensorBase(np.array([1, 2]))
        vec = TensorBase(np.array([1, 2, 3, 4]))
        mat = TensorBase(np.array([[2, 3, 3, 4], [5, 6, 6, 7]]))
        out = syft.addmv(t1, mat, vec, beta=2, alpha=2)
        self.assertTrue(np.array_equal(out.data, [68, 130]))


class addbmmTests(unittest.TestCase):
    def testaddbmm(self):
        t1 = TensorBase(np.array([[[3, 4], [5, 6]], [[7, 8], [1, 2]]]))
        t2 = TensorBase(np.array([[[3, 5], [5, 7]], [[7, 9], [1, 3]]]))
        mat = TensorBase(np.array([[2, 3], [3, 4]]))
        out = syft.addbmm(t1, t2, mat, beta=2, alpha=2)
        self.assertTrue(np.array_equal(out.data, [[176, 266], [114, 172]]))


class baddbmmTests(unittest.TestCase):
    def testbaddbmm(self):
        t1 = TensorBase(np.array([[[3, 4], [5, 6]], [[7, 8], [1, 2]]]))
        t2 = TensorBase(np.array([[[3, 5], [5, 7]], [[7, 9], [1, 3]]]))
        mat = TensorBase(np.array([[[2, 3], [3, 4]], [[4, 5], [5, 6]]]))
        out = syft.baddbmm(t1, t2, mat, beta=2, alpha=2)
        self.assertTrue(np.array_equal(out.data, [[[62, 92], [96, 142]],
                                                  [[122, 184], [28, 42]]]))


class transposeTests(unittest.TestCase):
    def testTranspose(self):
        t1 = TensorBase(np.array([[[3, 4], [5, 6]], [[7, 8], [1, 2]]]))
        out1 = syft.transpose(t1, 0, 1)
        self.assertTrue(np.array_equal(out1.data, np.array([[[3, 4], [7, 8]],
                                                            [[5, 6], [1, 2]]])))
        out2 = syft.transpose(t1, 0, 2)
        self.assertTrue(np.array_equal(out2.data, np.array([[[3, 7], [5, 1]],
                                                            [[4, 8], [6, 2]]])))
        out3 = syft.transpose(t1, 1, 2)
        self.assertTrue(np.array_equal(out3.data, np.array([[[3, 5], [4, 6]],
                                                            [[7, 1], [8, 2]]])))


class unsqueezeTests(unittest.TestCase):
    def testUnsqueeze(self):
        t1 = TensorBase(np.arange(3 * 4 * 5).reshape((3, 4, 5)))
        for i in range(len(t1.data.shape)):
            out = syft.unsqueeze(t1, i)
            expected_shape = list(t1.data.shape)
            expected_shape.insert(i, 1)

            self.assertTrue(np.array_equal(out.data.shape, expected_shape))


class mmtest(unittest.TestCase):
    def testmm1d(self):
        t1 = TensorBase(np.array([2, 3, 4]))
        t2 = TensorBase(np.array([3, 4, 5]))
        out = syft.mm(t1, t2)
        self.assertTrue(np.alltrue(out.data == [38]))

    def testmm2d(self):
        t1 = TensorBase(np.array([[1, 2], [1, 2]]))
        t2 = TensorBase(np.array([[2, 3], [2, 3]]))
        out = syft.mm(t1, t2)
        self.assertTrue(np.alltrue(out.data == [[6, 9], [6, 9]]))

    def testmm3d(self):
        t1 = TensorBase(np.array([[1, 2], [2, 3], [3, 4]]))
        t2 = TensorBase(np.array([[1, 2, 3], [2, 3, 4]]))
        out = syft.mm(t1, t2)
        self.assertTrue(np.alltrue(out.data == [[5, 8, 11], [8, 13, 18], [11, 18, 25]]))
