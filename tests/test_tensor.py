from syft import TensorBase
import syft
import unittest
import numpy as np
import math


# Here's our "unit tests".
class DimTests(unittest.TestCase):
    def testDimOne(self):
        t = TensorBase(np.array([1, 2, 3]))
        self.assertTrue(t.dim() == 1)

    def testView(self):
        t = TensorBase([1.0, 2.0, 3.0])
        self.assertTrue(syft.equal(t.view(-1, 1), TensorBase(np.array([[1], [2], [3]]))))

    def testAsView(self):
        t = TensorBase(np.array([1.0, 2.0, 3.0]))
        t1 = t.view([-1, 1])
        print(t.data.dtype)
        self.assertTrue(syft.equal(t.view_as(t1), TensorBase(np.array([[1.0], [2.0], [3.0]]))))


class AddTests(unittest.TestCase):
    def testSimple(self):
        t = TensorBase(np.array([1, 2, 3]))
        self.assertTrue(syft.equal(t + np.array([1, 2, 3]), [2, 4, 6]))

    def testInplace(self):
        t = TensorBase(np.array([1, 2, 3]))
        t += np.array([1, 2, 3])
        self.assertTrue(syft.equal(t.data, [2, 4, 6]))

    def testScalar(self):
        t = TensorBase(np.array([1, 2, 3]))
        self.assertTrue(syft.equal(t + 2, [3, 4, 5]))


class CeilTests(unittest.TestCase):
    def testCeil(self):
        t = TensorBase(np.array([1.4, 2.7, 6.2]))
        tdash = t.ceil()
        self.assertTrue(syft.equal(tdash.data, TensorBase([2, 3, 7])))
        self.assertTrue(syft.equal(t.data, TensorBase([1.4, 2.7, 6.2])))

    def testCeil_(self):
        t = TensorBase(np.array([1.4, 2.7, 6.2]))
        self.assertTrue(syft.equal(t.ceil_(), [2, 3, 7]))
        self.assertTrue(syft.equal(t.data, [2, 3, 7]))


class ZeroTests(unittest.TestCase):
    def testZero(self):
        t = TensorBase(np.array([13, 42, 1024]))
        self.assertTrue(syft.equal(t.zero_(), [0, 0, 0]))

        t = TensorBase(np.array([13.1, 42.2, 1024.4]))
        self.assertTrue(syft.equal(t.zero_(), [0.0, 0.0, 0.0]))

    def testCeil(self):
        t = TensorBase(np.array([1.4, 2.7, 6.2]))
        self.assertTrue(syft.equal(t.ceil(), TensorBase([2, 3, 7])))
        self.assertTrue(syft.equal(t.data, TensorBase([1.4, 2.7, 6.2])))


class FloorTests(unittest.TestCase):
    def testFloor_(self):
        t = TensorBase(np.array([1.4, 2.7, 6.2]))
        self.assertTrue(syft.equal(t.floor_(), [1., 2., 6.]))
        self.assertTrue(syft.equal(t.data, [1., 2., 6.]))


class SubTests(unittest.TestCase):
    def testSimple(self):
        t = TensorBase(np.array([1, 2, 3]))
        self.assertTrue(syft.equal(t - np.array([1, 2, 3]), [0, 0, 0]))

    def testInplace(self):
        t = TensorBase(np.array([1, 2, 3]))
        t -= np.array([1, 2, 3])
        self.assertTrue(syft.equal(t.data, [0, 0, 0]))

    def testScalar(self):
        t = TensorBase(np.array([1, 2, 3]))
        self.assertTrue(syft.equal(t - 1, [0, 1, 2]))


class MultTests(unittest.TestCase):
    def testSimple(self):
        t = TensorBase(np.array([1, 2, 3]))
        self.assertTrue(syft.equal(t * np.array([1, 2, 3]), [1, 4, 9]))

    def testInplace(self):
        t = TensorBase(np.array([1, 2, 3]))
        t *= np.array([1, 2, 3])
        self.assertTrue(syft.equal(t.data, [1, 4, 9]))

    def testScalar(self):
        t = TensorBase(np.array([1, 2, 3]))
        self.assertTrue(syft.equal(t * 2, [2, 4, 6]))


class DivTests(unittest.TestCase):
    def testSimple(self):
        t = TensorBase(np.array([2, 4, 8]))
        self.assertTrue(syft.equal(t / np.array([2, 2, 2]), [1, 2, 4]))

    def testInplace(self):
        t = TensorBase(np.array([2, 4, 8]))
        t /= np.array([2, 2, 2])
        self.assertTrue(syft.equal(t.data, [1, 2, 4]))

        t = TensorBase(np.array([1, 7, 11]))
        t /= np.array([3, 2, 9])
        self.assertTrue(syft.equal(t, [1 / 3, 7 / 2, 11 / 9]))

    def testScalar(self):
        t = TensorBase(np.array([2, 4, 6]))
        self.assertTrue(syft.equal(t / 2, [1, 2, 3]))


class AbsTests(unittest.TestCase):
    def testabs(self):
        t = TensorBase(np.array([-1, -2, 3]))
        self.assertTrue(np.array_equal(t.abs(), [1, 2, 3]))

    def testabs_(self):
        t = TensorBase(np.array([-1, -2, 3]))
        self.assertTrue(np.array_equal(t.abs_(), t.data))


class ShapeTests(unittest.TestCase):
    def testShape(self):
        t = TensorBase(np.array([[0, 1], [0, 5]]))
        self.assertTrue(syft.equal(t.shape(), (2, 2)))


class SqrtTests(unittest.TestCase):
    def testSqrt(self):
        t = TensorBase(np.array([[0, 4], [9, 16]]))

        self.assertTrue(syft.equal(t.sqrt(), ([[0, 2], [3, 4]])))

    def testSqrt_(self):
        t = TensorBase(np.array([[0, 4], [9, 16]]))
        t.sqrt_()
        self.assertTrue(syft.equal(t, ([[0, 2], [3, 4]])))


class SumTests(unittest.TestCase):
    def testDimNoneInt(self):
        t = TensorBase(np.array([1, 2, 3]))
        self.assertTrue(syft.equal(t.sum(), 6))

    def testDimIsNotNoneInt(self):
        t = TensorBase(np.array([[0, 1], [0, 5]]))
        self.assertTrue(syft.equal(t.sum(dim=1), [1, 5]))


class EqualTests(unittest.TestCase):
    def testEqual(self):
        t1 = TensorBase(np.array([1.2, 2, 3]))
        t2 = TensorBase(np.array([1.2, 2, 3]))
        self.assertTrue(syft.equal(t1, t2))

    def testEqOp(self):
        t1 = TensorBase(np.array([1, 2.4, 3]))
        t2 = TensorBase(np.array([1, 2.4, 3]))
        self.assertTrue(t1 == t2)

    def testNotEqual(self):
        t1 = TensorBase(np.array([1, 2, 3]))
        t2 = TensorBase(np.array([1, 4, 3]))
        self.assertFalse(syft.equal(t1, t2))

    def testIneqOp(self):
        t1 = TensorBase(np.array([1, 2, 3]))
        t2 = TensorBase(np.array([1, 4, 5]))
        self.assertTrue(t1 != t2)


class IndexTests(unittest.TestCase):
    def testIndexing(self):
        t1 = TensorBase(np.array([1.2, 2, 3]))
        self.assertEqual(1.2, t1[0])
        self.assertEqual(3, t1[-1])


class sigmoidTests(unittest.TestCase):
    def testSigmoid(self):
        t1 = TensorBase(np.array([1.2, 3.3, 4]))
        self.assertTrue(syft.equal(t1.sigmoid_(), TensorBase(
            [0.76852478, 0.96442881, 0.98201379])))


class addmm(unittest.TestCase):
    def testaddmm1d(self):
        t1 = TensorBase(np.array([1, 2, 3]))
        t2 = TensorBase(np.array([2, 3, 4]))
        mat = TensorBase(np.array([5]))
        out = t1.addmm(t2, mat, beta=2, alpha=2)
        self.assertTrue(np.array_equal(out.data, [50]))

    def testaddmm2d(self):
        t1 = TensorBase(np.array([[1, 2], [1, 2]]))
        t2 = TensorBase(np.array([[1, 2], [1, 2]]))
        mat = TensorBase(np.array([[2, 3], [3, 4]]))
        out = t1.addmm(t2, mat, beta=2, alpha=2)
        self.assertTrue(np.array_equal(out.data, [[10, 18], [12, 20]]))

    def testaddmm_1d(self):
        t1 = TensorBase(np.array([1, 2, 3]))
        t2 = TensorBase(np.array([2, 3, 4]))
        mat = TensorBase(np.array([5]))
        t1.addmm_(t2, mat, beta=2, alpha=2)
        self.assertTrue(np.array_equal(t1.data, [50]))

    def testaddmm_2d(self):
        t1 = TensorBase(np.array([[1, 2], [1, 2]]))
        t2 = TensorBase(np.array([[1, 2], [1, 2]]))
        mat = TensorBase(np.array([[2, 3], [3, 4]]))
        t1.addmm_(t2, mat, beta=2, alpha=2)
        self.assertTrue(np.array_equal(t1.data, [[10, 18], [12, 20]]))


class addcmulTests(unittest.TestCase):
    def testaddcmul1d(self):
        t1 = TensorBase(np.array([1, 2, 3]))
        t2 = TensorBase(np.array([2, 3, 4]))
        mat = TensorBase(np.array([5]))
        out = t1.addcmul(t2, mat, value=2)
        self.assertTrue(np.array_equal(out.data, [9, 17, 29]))

    def testaddcmul2d(self):
        t1 = TensorBase(np.array([[1, 2], [1, 2]]))
        t2 = TensorBase(np.array([[1, 2], [1, 2]]))
        mat = TensorBase(np.array([[2, 3], [3, 4]]))
        out = t1.addcmul(t2, mat, value=2)
        self.assertTrue(np.array_equal(out.data, [[4, 11], [5, 12]]))

    def testaddcmul_1d(self):
        t1 = TensorBase(np.array([1, 2, 3]))
        t2 = TensorBase(np.array([2, 3, 4]))
        mat = TensorBase(np.array([5]))
        t1.addcmul_(t2, mat, value=2)
        self.assertTrue(np.array_equal(t1.data, [9, 17, 29]))

    def testaddcmul_2d(self):
        t1 = TensorBase(np.array([[1, 2], [1, 2]]))
        t2 = TensorBase(np.array([[1, 2], [1, 2]]))
        mat = TensorBase(np.array([[2, 3], [3, 4]]))
        t1.addcmul_(t2, mat, value=2)
        self.assertTrue(np.array_equal(t1.data, [[4, 11], [5, 12]]))


class addcdivTests(unittest.TestCase):
    def testaddcdiv1d(self):
        t1 = TensorBase(np.array([1, 2, 3]))
        t2 = TensorBase(np.array([2, 5, 4]))
        mat = TensorBase(np.array([5]))
        out = t1.addcdiv(t2, mat, value=2)
        self.assertTrue(np.array_equal(out.data, [6., 5.8, 6.5]))

    def testaddcdiv2d(self):
        t1 = TensorBase(np.array([[1, 2], [1, 2]]))
        t2 = TensorBase(np.array([[1, 2], [1, 2]]))
        mat = TensorBase(np.array([[2, 3], [3, 4]]))
        out = t1.addcdiv(t2, mat, value=2)
        self.assertTrue(np.array_equal(out.data, [[4., 5.], [5., 6.]]))

    def testaddcdiv_1d(self):
        t1 = TensorBase(np.array([1, 2, 3]))
        t2 = TensorBase(np.array([2, 5, 4]))
        mat = TensorBase(np.array([5]))
        t1.addcdiv_(t2, mat, value=2)
        self.assertTrue(np.array_equal(t1.data, [6., 5.8, 6.5]))

    def testaddcdiv_2d(self):
        t1 = TensorBase(np.array([[1, 2], [1, 2]]))
        t2 = TensorBase(np.array([[1, 2], [1, 2]]))
        mat = TensorBase(np.array([[2, 3], [3, 4]]))
        t1.addcdiv_(t2, mat, value=2)
        self.assertTrue(np.array_equal(t1.data, [[4., 5.], [5., 6.]]))


class addmvTests(unittest.TestCase):
    def testaddmv(self):
        t1 = TensorBase(np.array([1, 2]))
        vec = TensorBase(np.array([1, 2, 3, 4]))
        mat = TensorBase(np.array([[2, 3, 3, 4], [5, 6, 6, 7]]))
        out = t1.addmv(mat, vec, beta=2, alpha=2)
        self.assertTrue(np.array_equal(out.data, [68, 130]))

    def testaddmv_(self):
        t1 = TensorBase(np.array([1, 2]))
        vec = TensorBase(np.array([1, 2, 3, 4]))
        mat = TensorBase(np.array([[2, 3, 3, 4], [5, 6, 6, 7]]))
        t1.addmv_(mat, vec, beta=2, alpha=2)
        self.assertTrue(np.array_equal(t1.data, [68, 130]))


class addbmmTests(unittest.TestCase):
    def testaddbmm(self):
        t1 = TensorBase(np.array([[[3, 4], [5, 6]], [[7, 8], [1, 2]]]))
        t2 = TensorBase(np.array([[[3, 5], [5, 7]], [[7, 9], [1, 3]]]))
        mat = TensorBase(np.array([[2, 3], [3, 4]]))
        out = t1.addbmm(t2, mat, beta=2, alpha=2)
        self.assertTrue(np.array_equal(out.data, [[176, 266], [114, 172]]))

    def testaddbmm_(self):
        t1 = TensorBase(np.array([[[3, 4], [5, 6]], [[7, 8], [1, 2]]]))
        t2 = TensorBase(np.array([[[3, 5], [5, 7]], [[7, 9], [1, 3]]]))
        mat = TensorBase(np.array([[2, 3], [3, 4]]))
        t1.addbmm_(t2, mat, beta=2, alpha=2)
        self.assertTrue(np.array_equal(t1.data, [[176, 266], [114, 172]]))


class baddbmmTests(unittest.TestCase):
    def testbaddbmm(self):
        t1 = TensorBase(np.array([[[3, 4], [5, 6]], [[7, 8], [1, 2]]]))
        t2 = TensorBase(np.array([[[3, 5], [5, 7]], [[7, 9], [1, 3]]]))
        mat = TensorBase(np.array([[[2, 3], [3, 4]], [[4, 5], [5, 6]]]))
        out = t1.baddbmm(t2, mat, beta=2, alpha=2)
        self.assertTrue(np.array_equal(out.data, [[[62, 92], [96, 142]],
                                                  [[122, 184], [28, 42]]]))

    def testbaddbmm_(self):
        t1 = TensorBase(np.array([[[3, 4], [5, 6]], [[7, 8], [1, 2]]]))
        t2 = TensorBase(np.array([[[3, 5], [5, 7]], [[7, 9], [1, 3]]]))
        mat = TensorBase(np.array([[[2, 3], [3, 4]], [[4, 5], [5, 6]]]))
        t1.baddbmm_(t2, mat, beta=2, alpha=2)
        self.assertTrue(np.array_equal(t1.data, [[[62, 92], [96, 142]],
                                                 [[122, 184], [28, 42]]]))


class transposeTests(unittest.TestCase):
    def testTranspose(self):
        t1 = TensorBase(np.array([[[3, 4], [5, 6]], [[7, 8], [1, 2]]]))
        out1 = t1.transpose(0, 1)
        self.assertTrue(np.array_equal(out1.data, [[[3, 4], [7, 8]],
                                                   [[5, 6], [1, 2]]]))
        out2 = t1.transpose(0, 2)
        self.assertTrue(np.array_equal(out2.data, [[[3, 7], [5, 1]],
                                                   [[4, 8], [6, 2]]]))
        out3 = t1.transpose(1, 2)
        self.assertTrue(np.array_equal(out3.data, [[[3, 5], [4, 6]],
                                                   [[7, 1], [8, 2]]]))

    def testTranspose_(self):
        t1 = TensorBase(np.array([[[3, 4], [5, 6]], [[7, 8], [1, 2]]]))
        t1.transpose_(0, 1)
        self.assertTrue(np.array_equal(t1.data, [[[3, 4], [7, 8]],
                                                 [[5, 6], [1, 2]]]))
        t2 = TensorBase(np.array([[[3, 4], [5, 6]], [[7, 8], [1, 2]]]))
        t2.transpose_(0, 2)
        self.assertTrue(np.array_equal(t2.data, [[[3, 7], [5, 1]],
                                                 [[4, 8], [6, 2]]]))
        t3 = TensorBase(np.array([[[3, 4], [5, 6]], [[7, 8], [1, 2]]]))
        t3.transpose_(1, 2)
        self.assertTrue(np.array_equal(t3.data, [[[3, 5], [4, 6]],
                                                 [[7, 1], [8, 2]]]))

    def testT(self):
        t1 = TensorBase(np.array([[[3, 4], [5, 6]], [[7, 8], [1, 2]]]))
        out1 = t1.t()
        self.assertTrue(np.array_equal(out1.data, [[[3, 4], [7, 8]],
                                                   [[5, 6], [1, 2]]]))

    def testT_(self):
        t1 = TensorBase(np.array([[[3, 4], [5, 6]], [[7, 8], [1, 2]]]))
        t1.transpose_(0, 1)
        self.assertTrue(np.array_equal(t1.data, [[[3, 4], [7, 8]],
                                                 [[5, 6], [1, 2]]]))


class unsqueezeTests(unittest.TestCase):
    def testUnsqueeze(self):
        t1 = TensorBase(np.arange(3 * 4 * 5).reshape((3, 4, 5)))
        for i in range(len(t1.data.shape)):
            out = t1.unsqueeze(i)
            expected_shape = list(t1.data.shape)
            expected_shape.insert(i, 1)
            self.assertTrue(np.array_equal(out.data.shape, expected_shape))

    def testUnsqueeze_(self):
        test_shape = (3, 4, 5)
        for i in range(len(test_shape)):
            t1 = TensorBase(np.arange(3 * 4 * 5).reshape(test_shape))
            expected_shape = list(t1.data.shape)
            expected_shape.insert(i, 1)
            t1.unsqueeze_(i)
            self.assertTrue(np.array_equal(t1.data.shape, expected_shape))


class expTests(unittest.TestCase):
    def testexp(self):
        t3 = TensorBase(np.array([[[1, 3], [3, 5]],
                                  [[5, 7], [9, 1]]]))
        out = t3.exp()
        self.assertTrue(np.allclose(out.data, [[[2.71828183e+00, 2.00855369e+01], [2.00855369e+01, 1.48413159e+02]],
                                               [[1.48413159e+02, 1.09663316e+03], [8.10308393e+03, 2.71828183e+00]]]))

    def testexp_(self):
        t3 = TensorBase(np.array([[[1, 3], [3, 5]],
                                  [[5, 7], [9, 1]]]))
        t3.exp_()
        self.assertTrue(np.allclose(t3.data, [[[2.71828183e+00, 2.00855369e+01], [2.00855369e+01, 1.48413159e+02]],
                                              [[1.48413159e+02, 1.09663316e+03], [8.10308393e+03, 2.71828183e+00]]]))


class fracTests(unittest.TestCase):
    def testfrac(self):
        t3 = TensorBase(np.array([1.23, 4.56, 7.89]))
        out = t3.frac()
        self.assertTrue(np.allclose(out.data, [0.23, 0.56, 0.89]))

    def testfrac_(self):
        t3 = TensorBase(np.array([1.23, 4.56, 7.89]))
        t3.frac_()
        self.assertTrue(np.allclose(t3.data, [0.23, 0.56, 0.89]))


class rsqrtTests(unittest.TestCase):
    def testrsqrt(self):
        t1 = TensorBase(np.array([2, 3, 4]))
        out = t1.rsqrt()
        self.assertTrue(np.allclose(out.data, [0.70710678, 0.57735027, 0.5]))

    def testrsqrt_(self):
        t1 = TensorBase(np.array([2, 3, 4]))
        t1.rsqrt_()
        self.assertTrue(np.allclose(t1.data, [0.70710678, 0.57735027, 0.5]))


class numpyTests(unittest.TestCase):
    def testnumpy(self):
        t1 = TensorBase(np.array([[1, 2], [3, 4]]))
        self.assertTrue(np.array_equal(t1.to_numpy(), np.array([[1, 2], [3, 4]])))


class reciprocalTests(unittest.TestCase):
    def testreciprocal(self):
        t1 = TensorBase(np.array([2, 3, 4]))
        out = t1.reciprocal()
        self.assertTrue(np.allclose(out.data, [0.5, 0.33333333, 0.25]))

    def testrsqrt_(self):
        t1 = TensorBase(np.array([2, 3, 4]))
        t1.reciprocal_()
        self.assertTrue(np.allclose(t1.data, [0.5, 0.33333333, 0.25]))


class logTests(unittest.TestCase):
    def testLog(self):
        t1 = TensorBase(np.array([math.exp(1), math.exp(2), math.exp(3)]))
        self.assertTrue(np.array_equal((t1.log()).data, [1., 2., 3.]))

    def testLog_(self):
        t1 = TensorBase(np.array([math.exp(1), math.exp(2), math.exp(3)]))
        self.assertTrue(np.array_equal((t1.log_()).data, [1., 2., 3.]))

    def testLog1p(self):
        t1 = TensorBase(np.array([1, 2, 3]))
        self.assertTrue(np.allclose((t1.log1p()).data, [0.69314718, 1.09861229, 1.38629436]))

    def testLog1p_(self):
        t1 = TensorBase(np.array([1, 2, 3]))
        self.assertTrue(np.allclose((t1.log1p_()).data, [0.69314718, 1.09861229, 1.38629436]))


class clampTests(unittest.TestCase):
    def testClampInt(self):
        t1 = TensorBase(np.arange(10))
        t2 = t1.clamp(minimum=2, maximum=7)
        expected_tensor = TensorBase(np.array([2, 2, 2, 3, 4, 5, 6, 7, 7, 7]))
        self.assertEqual(t2, expected_tensor)

    def testClampFloat(self):
        t1 = TensorBase(np.arange(1, step=0.1))
        t2 = t1.clamp(minimum=0.2, maximum=0.7)
        expected_tensor = TensorBase(np.array([0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.7]))
        self.assertEqual(t2, expected_tensor)

    def testClampIntInPlace(self):
        t1 = TensorBase(np.arange(10))
        t1.clamp_(minimum=2, maximum=7)
        expected_tensor = TensorBase(np.array([2, 2, 2, 3, 4, 5, 6, 7, 7, 7]))
        self.assertEqual(t1, expected_tensor)

    def testClampFloatInPlace(self):
        t1 = TensorBase(np.arange(1, step=0.1))
        t1.clamp_(minimum=0.2, maximum=0.7)
        expected_tensor = TensorBase(np.array([0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.7]))
        self.assertEqual(t1, expected_tensor)


class uniformTests(unittest.TestCase):
    def testUniform(self):
        t1 = TensorBase(np.zeros(4))
        out = t1.uniform(low=0, high=3)
        self.assertTrue(np.all(out.data > 0) and np.all(out.data < 3))

    def testUniform_(self):
        t1 = TensorBase(np.zeros(4))
        t1.uniform_(low=0, high=3)
        self.assertTrue(np.all(t1.data > 0) and np.all(t1.data < 3))


class fillTests(unittest.TestCase):
    def testFill_(self):
        t1 = TensorBase(np.array([1, 2, 3, 4]))
        t1.fill_(5)
        self.assertTrue(np.all(t1.data == 5))


class topkTests(unittest.TestCase):
    def testTopK(self):
        t1 = TensorBase(np.array([[900, 800, 1000, 2000, 5, 10, 20, 40, 50], [10, 11, 12, 13, 5, 6, 7, 8, 9], [30, 40, 50, 10, 8, 1, 2, 3, 4]]))
        t2 = t1.topk(3, largest=True)
        self.assertTrue(np.array_equal(t2.data, np.array([[900, 1000, 2000], [11, 12, 13], [30, 40, 50]])))


class tolistTests(unittest.TestCase):
    def testToList(self):
        t1 = TensorBase(np.array([200, 300, 90, 100, 600]))
        t1_list = t1.tolist()
        self.assertTrue(isinstance(t1_list, list))


class traceTests(unittest.TestCase):
    def testTrace(self):
        t1 = TensorBase(np.arange(1, 10).reshape(3, 3))
        self.assertTrue(np.equal(t1.trace().data, 15))


class roundTests(unittest.TestCase):
    def testRound(self):
        t1 = TensorBase(np.array([10.4, 9.6, 100.12, 4.0]))
        t2 = t1.round(0)
        self.assertTrue(np.array_equal(t2.data, np.array([10., 10., 100., 4.])))

    def testRound_(self):
        t1 = TensorBase(np.array([10.4, 9.6, 100.12, 4.0]))
        t1.round_(0)
        self.assertTrue(np.array_equal(t1.data, np.array([10., 10., 100., 4.])))


class repeatTests(unittest.TestCase):
    def testRepeat(self):
        t1 = TensorBase(np.array([1, 2, 3]))
        t2 = t1.repeat(reps=(4, 2))
        self.assertTrue(np.array_equal(t2.data, np.array([[1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3]])))


class powTests(unittest.TestCase):
    def testPow(self):
        t1 = TensorBase(np.array([2, 4, 6]))
        t2 = t1.pow(2)
        self.assertTrue(np.array_equal(t2.data, np.array([4, 16, 36])))

    def testPow_(self):
        t1 = TensorBase(np.array([2, 4, 6]))
        t1.pow_(2)
        self.assertTrue(np.array_equal(t1.data, np.array([4, 16, 36])))


class prodTests(unittest.TestCase):
    def testProd(self):
        t1 = TensorBase(np.array([1, 2, 3]))
        t2 = t1.prod()
        self.assertTrue(np.equal(t2.data, 6))


class randomTests(unittest.TestCase):
    def testRandom_(self):
        np.random.seed(0)
        t1 = TensorBase(np.zeros(4))
        t1.random_(low=0, high=5, size=4)
        self.assertTrue(np.array_equal(t1.data, np.array([4, 0, 3, 3])))


class nonzeroTests(unittest.TestCase):
    def testNonZero(self):
        t1 = TensorBase(np.array([[1, 0, 0], [0, 2, 5]]))
        t2 = t1.nonzero()
        self.assertTrue(np.array_equal(t2.data, np.array([[0, 1, 1], [0, 1, 2]])))


if __name__ == "__main__":
    unittest.main()
