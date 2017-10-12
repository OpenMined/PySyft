from syft import TensorBase
import syft
import unittest
from syft import tensor
import numpy as np
import math
import pytest


# Here's our "unit tests".
class DimTests(unittest.TestCase):
    def test_dim_one(self):
        t = TensorBase(np.array([1, 2, 3]))
        self.assertTrue(t.dim() == 1)

    def test_view(self):
        t = TensorBase([1.0, 2.0, 3.0])
        self.assertTrue(syft.equal(t.view(-1, 1), TensorBase(np.array([[1], [2], [3]]))))

    def test_as_view(self):
        t = TensorBase(np.array([1.0, 2.0, 3.0]))
        t1 = t.view([-1, 1])
        print(t.data.dtype)
        self.assertTrue(syft.equal(t.view_as(t1), TensorBase(np.array([[1.0], [2.0], [3.0]]))))

    def test_resize(self):
        t = TensorBase(np.array([1.0, 2.0, 3.0]))
        t.resize_([1, 2])
        self.assertEqual(t.data.shape, (1, 2))

    def test_resize_as(self):
        t = TensorBase(np.array([1, 2, 3]))
        t2 = TensorBase(np.array([[1], [2]]))
        t.resize_as_(t2)
        self.assertEqual(t.data.shape, (2, 1))


class DiagTests(unittest.TestCase):
    def test_one_dim_tensor_main_diag(self):
        t = TensorBase(np.array([1, 2, 3]))
        tdiag = t.diag()
        self.assertTrue(syft.equal(tdiag.data, TensorBase(np.array([[1, 0, 0],
                                                                    [0, 2, 0],
                                                                    [0, 0, 3]]))))

    def test_one_dim_tensor_upper_diag(self):
        t = TensorBase(np.array([1, 2, 3]))
        tdiag = t.diag(1)
        self.assertTrue(syft.equal(tdiag.data, TensorBase(np.array([[0, 1, 0, 0],
                                                                    [0, 0, 2, 0],
                                                                    [0, 0, 0, 3],
                                                                    [0, 0, 0, 0]]))))

    def test_one_dim_tensor_below_diag(self):
        t = TensorBase(np.array([1, 2, 3]))
        tdiag = t.diag(-1)
        self.assertTrue(syft.equal(tdiag.data, TensorBase(np.array([[0, 0, 0, 0],
                                                                    [1, 0, 0, 0],
                                                                    [0, 2, 0, 0],
                                                                    [0, 0, 3, 0]]))))

    def test_two_dim_tensor_main_diag(self):
        t = TensorBase(np.array([[0, 1], [2, 3]]))
        tdiag = t.diag()
        self.assertTrue(syft.equal(tdiag.data, TensorBase(np.array([0, 3]))))

    def test_two_dim_tensor_upper_diag(self):
        t = TensorBase(np.array([[0, 1], [2, 3]]))
        tdiag = t.diag(1)
        self.assertTrue(syft.equal(tdiag.data, TensorBase(np.array([1]))))

    def test_two_dim_tensor_below_diag(self):
        t = TensorBase(np.array([[0, 1], [2, 3]]))
        tdiag = t.diag(-1)
        self.assertTrue(syft.equal(tdiag.data, TensorBase(np.array([2]))))


class AddTests(unittest.TestCase):
    def test_simple(self):
        t = TensorBase(np.array([1, 2, 3]))
        self.assertTrue(syft.equal(t + np.array([1, 2, 3]), [2, 4, 6]))

    def test_inplace(self):
        t = TensorBase(np.array([1, 2, 3]))
        t += np.array([1, 2, 3])
        self.assertTrue(syft.equal(t.data, [2, 4, 6]))

    def test_scalar(self):
        t = TensorBase(np.array([1, 2, 3]))
        self.assertTrue(syft.equal(t + 2, [3, 4, 5]))


class CeilTests(unittest.TestCase):
    def test_ceil(self):
        t = TensorBase(np.array([1.4, 2.7, 6.2]))
        tdash = t.ceil()
        self.assertTrue(syft.equal(tdash.data, TensorBase([2, 3, 7])))
        self.assertTrue(syft.equal(t.data, TensorBase([1.4, 2.7, 6.2])))

    def test_ceil_(self):
        t = TensorBase(np.array([1.4, 2.7, 6.2]))
        self.assertTrue(syft.equal(t.ceil_(), [2, 3, 7]))
        self.assertTrue(syft.equal(t.data, [2, 3, 7]))


class ZeroTests(unittest.TestCase):
    def test_zero(self):
        t = TensorBase(np.array([13, 42, 1024]))
        self.assertTrue(syft.equal(t.zero_(), TensorBase([0, 0, 0])))

        t = TensorBase(np.array([13.1, 42.2, 1024.4]))
        self.assertTrue(syft.equal(t.zero_(), TensorBase([0.0, 0.0, 0.0])))


class FloorTests(unittest.TestCase):
    def test_floor_(self):
        t = TensorBase(np.array([1.4, 2.7, 6.2]))
        self.assertTrue(syft.equal(t.floor_(), [1., 2., 6.]))
        self.assertTrue(syft.equal(t.data, [1., 2., 6.]))


class SubTests(unittest.TestCase):
    def test_simple(self):
        t = TensorBase(np.array([1, 2, 3]))
        self.assertTrue(syft.equal(t - np.array([1, 2, 3]), [0, 0, 0]))

    def test_inplace(self):
        t = TensorBase(np.array([1, 2, 3]))
        t -= np.array([1, 2, 3])
        self.assertTrue(syft.equal(t.data, [0, 0, 0]))

    def test_scalar(self):
        t = TensorBase(np.array([1, 2, 3]))
        self.assertTrue(syft.equal(t - 1, [0, 1, 2]))


class MaxTests(unittest.TestCase):
    def test_no_dim(self):
        t = TensorBase(np.array([[0.77937768, 0.51022484, 0.49155195, 0.02769902], [0.03777148, 0.13020167, 0.02155692, 0.69574893]]))
        self.assertTrue(t.max() == 0.77937768)

    def test_axis(self):
        t = TensorBase(np.array([[0.77937768, 0.51022484, 0.49155195, 0.02769902], [0.03777148, 0.13020167, 0.02155692, 0.69574893]]))
        result = t.max(axis=1)
        self.assertTrue(syft.equal(result, [0.77937768, 0.69574893]))
        result = t.max(axis=0)
        self.assertTrue(syft.equal(result, [0.77937768, 0.51022484, 0.49155195, 0.69574893]))


class MultTests(unittest.TestCase):
    def test_simple(self):
        t = TensorBase(np.array([1, 2, 3]))
        self.assertTrue(syft.equal(t * np.array([1, 2, 3]), [1, 4, 9]))

    def test_inplace(self):
        t = TensorBase(np.array([1, 2, 3]))
        t *= np.array([1, 2, 3])
        self.assertTrue(syft.equal(t.data, [1, 4, 9]))

    def test_scalar(self):
        t = TensorBase(np.array([1, 2, 3]))
        self.assertTrue(syft.equal(t * 2, [2, 4, 6]))


class DivTests(unittest.TestCase):
    def test_simple(self):
        t = TensorBase(np.array([2, 4, 8]))
        self.assertTrue(syft.equal(t / np.array([2, 2, 2]), [1, 2, 4]))

    def test_inplace(self):
        t = TensorBase(np.array([2, 4, 8]))
        t /= np.array([2, 2, 2])
        self.assertTrue(syft.equal(t.data, [1, 2, 4]))

        t = TensorBase(np.array([1, 7, 11]))
        t /= np.array([3, 2, 9])
        self.assertTrue(syft.equal(t, [1 / 3, 7 / 2, 11 / 9]))

    def test_scalar(self):
        t = TensorBase(np.array([2, 4, 6]))
        self.assertTrue(syft.equal(t / 2, [1, 2, 3]))


class AbsTests(unittest.TestCase):
    def test_abs(self):
        t = TensorBase(np.array([-1, -2, 3]))
        self.assertTrue(np.array_equal(t.abs(), TensorBase([1, 2, 3])))

    def test_abs_(self):
        t = TensorBase(np.array([-1, -2, 3]))
        self.assertTrue(np.array_equal(t.abs_(), TensorBase([1, 2, 3])))


class ShapeTests(unittest.TestCase):
    def test_shape(self):
        t = TensorBase(np.array([[0, 1], [0, 5]]))
        self.assertTrue(syft.equal(t.shape(), (2, 2)))


class SqrtTests(unittest.TestCase):
    def test_sqrt(self):
        t = TensorBase(np.array([[0, 4], [9, 16]]))

        self.assertTrue(syft.equal(t.sqrt(), TensorBase([[0, 2], [3, 4]])))

    def test_sqrt_(self):
        t = TensorBase(np.array([[0, 4], [9, 16]]))
        t.sqrt_()
        self.assertTrue(syft.equal(t, TensorBase([[0, 2], [3, 4]])))


class SumTests(unittest.TestCase):
    def test_dim_none_int(self):
        t = TensorBase(np.array([1, 2, 3]))
        self.assertTrue(syft.equal(t.sum(), 6))

    def test_dim_is_not_none_int(self):
        t = TensorBase(np.array([[0, 1], [0, 5]]))
        self.assertTrue(syft.equal(t.sum(dim=1), [1, 5]))


class EqualTests(unittest.TestCase):
    def test_equal(self):
        t1 = TensorBase(np.array([1.2, 2, 3]))
        t2 = TensorBase(np.array([1.2, 2, 3]))
        self.assertTrue(syft.equal(t1, t2))

    def test_equal_operation(self):
        t1 = TensorBase(np.array([1, 2.4, 3]))
        t2 = TensorBase(np.array([1, 2.4, 3]))
        self.assertTrue(t1 == t2)

    def test_not_equal(self):
        t1 = TensorBase(np.array([1, 2, 3]))
        t2 = TensorBase(np.array([1, 4, 3]))
        self.assertFalse(syft.equal(t1, t2))

    def test_shape_not_equal(self):
        t1 = TensorBase(np.array([1, 2]))
        t2 = TensorBase(np.array([1, 4, 3]))
        self.assertFalse(syft.equal(t1, t2))

    def test_inequality_operation(self):
        t1 = TensorBase(np.array([1, 2, 3]))
        t2 = TensorBase(np.array([1, 4, 5]))
        self.assertTrue(t1 != t2)

    def test_shape_inequality_operation(self):
        t1 = TensorBase(np.array([1, 2]))
        t2 = TensorBase(np.array([1, 4, 5]))
        self.assertTrue(t1 != t2)


class sigmoidTests(unittest.TestCase):
    def test_sigmoid(self):
        t1 = TensorBase(np.array([1.2, 3.3, 4]))
        self.assertTrue(syft.equal(t1.sigmoid_(), TensorBase(
            [0.76852478, 0.96442881, 0.98201379])))


class addmm(unittest.TestCase):
    def test_addmm_1d(self):
        t1 = TensorBase(np.array([1, 2, 3]))
        t2 = TensorBase(np.array([2, 3, 4]))
        mat = TensorBase(np.array([5]))
        out = t1.addmm(t2, mat, beta=2, alpha=2)
        self.assertTrue(np.array_equal(out.data, [50]))

    def test_addmm_2d(self):
        t1 = TensorBase(np.array([[1, 2], [1, 2]]))
        t2 = TensorBase(np.array([[1, 2], [1, 2]]))
        mat = TensorBase(np.array([[2, 3], [3, 4]]))
        out = t1.addmm(t2, mat, beta=2, alpha=2)
        self.assertTrue(np.array_equal(out.data, [[10, 18], [12, 20]]))

    def test_addmm__1d(self):
        t1 = TensorBase(np.array([1, 2, 3]))
        t2 = TensorBase(np.array([2, 3, 4]))
        mat = TensorBase(np.array([5]))
        t1.addmm_(t2, mat, beta=2, alpha=2)
        self.assertTrue(np.array_equal(t1.data, [50]))

    def test_addmm__2d(self):
        t1 = TensorBase(np.array([[1, 2], [1, 2]]))
        t2 = TensorBase(np.array([[1, 2], [1, 2]]))
        mat = TensorBase(np.array([[2, 3], [3, 4]]))
        t1.addmm_(t2, mat, beta=2, alpha=2)
        self.assertTrue(np.array_equal(t1.data, [[10, 18], [12, 20]]))


class addcmulTests(unittest.TestCase):
    def test_addcmul_1d(self):
        t1 = TensorBase(np.array([1, 2, 3]))
        t2 = TensorBase(np.array([2, 3, 4]))
        mat = TensorBase(np.array([5]))
        out = t1.addcmul(t2, mat, value=2)
        self.assertTrue(np.array_equal(out.data, [9, 17, 29]))

    def test_addcmul_2d(self):
        t1 = TensorBase(np.array([[1, 2], [1, 2]]))
        t2 = TensorBase(np.array([[1, 2], [1, 2]]))
        mat = TensorBase(np.array([[2, 3], [3, 4]]))
        out = t1.addcmul(t2, mat, value=2)
        self.assertTrue(np.array_equal(out.data, [[4, 11], [5, 12]]))

    def test_addcmul__1d(self):
        t1 = TensorBase(np.array([1, 2, 3]))
        t2 = TensorBase(np.array([2, 3, 4]))
        mat = TensorBase(np.array([5]))
        t1.addcmul_(t2, mat, value=2)
        self.assertTrue(np.array_equal(t1.data, [9, 17, 29]))

    def test_addcmul__2d(self):
        t1 = TensorBase(np.array([[1, 2], [1, 2]]))
        t2 = TensorBase(np.array([[1, 2], [1, 2]]))
        mat = TensorBase(np.array([[2, 3], [3, 4]]))
        t1.addcmul_(t2, mat, value=2)
        self.assertTrue(np.array_equal(t1.data, [[4, 11], [5, 12]]))


class addcdivTests(unittest.TestCase):
    def test_addcdiv_1d(self):
        t1 = TensorBase(np.array([1, 2, 3]))
        t2 = TensorBase(np.array([2, 5, 4]))
        mat = TensorBase(np.array([5]))
        out = t1.addcdiv(t2, mat, value=2)
        self.assertTrue(np.array_equal(out.data, [6., 5.8, 6.5]))

    def test_addcdiv_2d(self):
        t1 = TensorBase(np.array([[1, 2], [1, 2]]))
        t2 = TensorBase(np.array([[1, 2], [1, 2]]))
        mat = TensorBase(np.array([[2, 3], [3, 4]]))
        out = t1.addcdiv(t2, mat, value=2)
        self.assertTrue(np.array_equal(out.data, [[4., 5.], [5., 6.]]))

    def test_addcdiv__1d(self):
        t1 = TensorBase(np.array([1, 2, 3]))
        t2 = TensorBase(np.array([2, 5, 4]))
        mat = TensorBase(np.array([5]))
        t1.addcdiv_(t2, mat, value=2)
        self.assertTrue(np.array_equal(t1.data, [6., 5.8, 6.5]))

    def test_addcdiv__2d(self):
        t1 = TensorBase(np.array([[1, 2], [1, 2]]))
        t2 = TensorBase(np.array([[1, 2], [1, 2]]))
        mat = TensorBase(np.array([[2, 3], [3, 4]]))
        t1.addcdiv_(t2, mat, value=2)
        self.assertTrue(np.array_equal(t1.data, [[4., 5.], [5., 6.]]))


class addmvTests(unittest.TestCase):
    def test_addmv(self):
        t1 = TensorBase(np.array([1, 2]))
        vec = TensorBase(np.array([1, 2, 3, 4]))
        mat = TensorBase(np.array([[2, 3, 3, 4], [5, 6, 6, 7]]))
        out = t1.addmv(mat, vec, beta=2, alpha=2)
        self.assertTrue(np.array_equal(out.data, [68, 130]))

    def test_addmv_(self):
        t1 = TensorBase(np.array([1, 2]))
        vec = TensorBase(np.array([1, 2, 3, 4]))
        mat = TensorBase(np.array([[2, 3, 3, 4], [5, 6, 6, 7]]))
        t1.addmv_(mat, vec, beta=2, alpha=2)
        self.assertTrue(np.array_equal(t1.data, [68, 130]))


class bmmTests(unittest.TestCase):
    def test_bmm_size(self):
        t1 = TensorBase(np.random.rand(4, 3, 2))
        t2 = TensorBase(np.random.rand(4, 2, 1))
        out = t1.bmm(t2)
        self.assertTupleEqual(out.shape(), (4, 3, 1))

    def test_bmm(self):
        t1 = TensorBase(np.array([[[3, 1]], [[1, 2]]]))
        t2 = TensorBase(np.array([[[1], [3]], [[4], [8]]]))
        out = t1.bmm(t2)
        test_result = np.array([[[6]], [[20]]])
        self.assertTrue(np.array_equal(out.data, test_result))


class addbmmTests(unittest.TestCase):
    def test_addbmm(self):
        t1 = TensorBase(np.array([[[3, 4], [5, 6]], [[7, 8], [1, 2]]]))
        t2 = TensorBase(np.array([[[3, 5], [5, 7]], [[7, 9], [1, 3]]]))
        mat = TensorBase(np.array([[2, 3], [3, 4]]))
        out = t1.addbmm(t2, mat, beta=2, alpha=2)
        self.assertTrue(np.array_equal(out.data, [[176, 266], [114, 172]]))

    def test_addbmm_(self):
        t1 = TensorBase(np.array([[[3, 4], [5, 6]], [[7, 8], [1, 2]]]))
        t2 = TensorBase(np.array([[[3, 5], [5, 7]], [[7, 9], [1, 3]]]))
        mat = TensorBase(np.array([[2, 3], [3, 4]]))
        t1.addbmm_(t2, mat, beta=2, alpha=2)
        self.assertTrue(np.array_equal(t1.data, [[176, 266], [114, 172]]))


class baddbmmTests(unittest.TestCase):
    def test_baddbmm(self):
        t1 = TensorBase(np.array([[[3, 4], [5, 6]], [[7, 8], [1, 2]]]))
        t2 = TensorBase(np.array([[[3, 5], [5, 7]], [[7, 9], [1, 3]]]))
        mat = TensorBase(np.array([[[2, 3], [3, 4]], [[4, 5], [5, 6]]]))
        out = t1.baddbmm(t2, mat, beta=2, alpha=2)
        self.assertTrue(np.array_equal(out.data, [[[62, 92], [96, 142]],
                                                  [[122, 184], [28, 42]]]))

    def test_baddbmm_(self):
        t1 = TensorBase(np.array([[[3, 4], [5, 6]], [[7, 8], [1, 2]]]))
        t2 = TensorBase(np.array([[[3, 5], [5, 7]], [[7, 9], [1, 3]]]))
        mat = TensorBase(np.array([[[2, 3], [3, 4]], [[4, 5], [5, 6]]]))
        t1.baddbmm_(t2, mat, beta=2, alpha=2)
        self.assertTrue(np.array_equal(t1.data, [[[62, 92], [96, 142]],
                                                 [[122, 184], [28, 42]]]))


class PermuteTests(unittest.TestCase):
    # WARNING: don't change the name of this function
    def dest3d(self):
        t = TensorBase(np.ones((2, 3, 5)))
        tdash = t.permute((2, 0, 1))
        self.assertTrue(tdash.data.shape == [5, 2, 3])
        self.assertTrue(t.data.shape == [2, 3, 5])


class transposeTests(unittest.TestCase):
    def test_transpose(self):
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

    def test_transpose_(self):
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

    def test_t(self):
        t1 = TensorBase(np.array([[[3, 4], [5, 6]], [[7, 8], [1, 2]]]))
        out1 = t1.t()
        self.assertTrue(np.array_equal(out1.data, [[[3, 4], [7, 8]],
                                                   [[5, 6], [1, 2]]]))


class unsqueezeTests(unittest.TestCase):
    def test_unsqueeze(self):
        t1 = TensorBase(np.arange(3 * 4 * 5).reshape((3, 4, 5)))
        for i in range(len(t1.data.shape)):
            out = t1.unsqueeze(i)
            expected_shape = list(t1.data.shape)
            expected_shape.insert(i, 1)
            self.assertTrue(np.array_equal(out.data.shape, expected_shape))

    def test_unsqueeze_(self):
        test_shape = (3, 4, 5)
        for i in range(len(test_shape)):
            t1 = TensorBase(np.arange(3 * 4 * 5).reshape(test_shape))
            expected_shape = list(t1.data.shape)
            expected_shape.insert(i, 1)
            t1.unsqueeze_(i)
            self.assertTrue(np.array_equal(t1.data.shape, expected_shape))


class expTests(unittest.TestCase):
    def test_exp(self):
        t3 = TensorBase(np.array([[[1, 3], [3, 5]],
                                  [[5, 7], [9, 1]]]))
        out = t3.exp()
        self.assertTrue(np.allclose(out.data, [[[2.71828183e+00, 2.00855369e+01], [2.00855369e+01, 1.48413159e+02]],
                                               [[1.48413159e+02, 1.09663316e+03], [8.10308393e+03, 2.71828183e+00]]]))

    def test_exp_(self):
        t3 = TensorBase(np.array([[[1, 3], [3, 5]],
                                  [[5, 7], [9, 1]]]))
        t3.exp_()
        self.assertTrue(np.allclose(t3.data, [[[2.71828183e+00, 2.00855369e+01], [2.00855369e+01, 1.48413159e+02]],
                                              [[1.48413159e+02, 1.09663316e+03], [8.10308393e+03, 2.71828183e+00]]]))


class fracTests(unittest.TestCase):
    def test_frac(self):
        t3 = TensorBase(np.array([1.23, 4.56, 7.89]))
        out = t3.frac()
        self.assertTrue(np.allclose(out.data, [0.23, 0.56, 0.89]))

    def test_frac_(self):
        t3 = TensorBase(np.array([1.23, 4.56, 7.89]))
        t3.frac_()
        self.assertTrue(np.allclose(t3.data, [0.23, 0.56, 0.89]))


class rsqrtTests(unittest.TestCase):
    def test_rsqrt(self):
        t1 = TensorBase(np.array([2, 3, 4]))
        out = t1.rsqrt()
        self.assertTrue(np.allclose(out.data, [0.70710678, 0.57735027, 0.5]))

    def test_rsqrt_(self):
        t1 = TensorBase(np.array([2, 3, 4]))
        t1.rsqrt_()
        self.assertTrue(np.allclose(t1.data, [0.70710678, 0.57735027, 0.5]))


class signTests(unittest.TestCase):
    def test_sign(self):
        t1 = TensorBase(np.array([1, 2, -1, -2]))
        out = t1.sign()
        self.assertTrue(np.array_equal(out.data, [1, 1, -1, -1]))

    def test_sign_(self):
        t1 = TensorBase(np.array([1, 2, -1, -2]))
        t1.sign_()
        self.assertTrue(np.array_equal(t1.data, [1, 1, -1, -1]))


class numpyTests(unittest.TestCase):
    def test_numpy(self):
        t1 = TensorBase(np.array([[1, 2], [3, 4]]))
        self.assertTrue(np.array_equal(t1.to_numpy(), np.array([[1, 2], [3, 4]])))


class reciprocalTests(unittest.TestCase):
    def test_reciprocal(self):
        t1 = TensorBase(np.array([2, 3, 4]))
        out = t1.reciprocal()
        self.assertTrue(np.allclose(out.data, [0.5, 0.33333333, 0.25]))

    def test_reciprocal_(self):
        t1 = TensorBase(np.array([2, 3, 4]))
        t1.reciprocal_()
        self.assertTrue(np.allclose(t1.data, [0.5, 0.33333333, 0.25]))


class logTests(unittest.TestCase):
    def test_log(self):
        t1 = TensorBase(np.array([math.exp(1), math.exp(2), math.exp(3)]))
        self.assertTrue(np.array_equal((t1.log()).data, [1., 2., 3.]))

    def test_log_(self):
        t1 = TensorBase(np.array([math.exp(1), math.exp(2), math.exp(3)]))
        self.assertTrue(np.array_equal((t1.log_()).data, [1., 2., 3.]))

    def test_log_1p(self):
        t1 = TensorBase(np.array([1, 2, 3]))
        self.assertTrue(np.allclose((t1.log1p()).data, [0.69314718, 1.09861229, 1.38629436]))

    def test_log_1p_(self):
        t1 = TensorBase(np.array([1, 2, 3]))
        self.assertTrue(np.allclose((t1.log1p_()).data, [0.69314718, 1.09861229, 1.38629436]))


class clampTests(unittest.TestCase):
    def test_clamp_int(self):
        t1 = TensorBase(np.arange(10))
        t2 = t1.clamp(minimum=2, maximum=7)
        expected_tensor = TensorBase(np.array([2, 2, 2, 3, 4, 5, 6, 7, 7, 7]))
        self.assertEqual(t2, expected_tensor)

    def test_clamp_float(self):
        t1 = TensorBase(np.arange(1, step=0.1))
        t2 = t1.clamp(minimum=0.2, maximum=0.7)
        expected_tensor = TensorBase(np.array([0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.7]))
        self.assertEqual(t2, expected_tensor)

    def test_clamp_int_in_place(self):
        t1 = TensorBase(np.arange(10))
        t1.clamp_(minimum=2, maximum=7)
        expected_tensor = TensorBase(np.array([2, 2, 2, 3, 4, 5, 6, 7, 7, 7]))
        self.assertEqual(t1, expected_tensor)

    def test_clamp_float_in_place(self):
        t1 = TensorBase(np.arange(1, step=0.1))
        t1.clamp_(minimum=0.2, maximum=0.7)
        expected_tensor = TensorBase(np.array([0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.7]))
        self.assertEqual(t1, expected_tensor)


class cloneTests(unittest.TestCase):
    def test_clone(self):
        t1 = TensorBase(np.random.randint(0, 10, size=(5, 10)))
        t2 = t1.clone()
        self.assertEqual(t1, t2)
        self.assertIsNot(t1, t2)


class chunkTests(unittest.TestCase):
    def test_chunk(self):
        t1 = TensorBase(np.random.randint(0, 10, size=(5, 10)))
        t2, t3 = t1.chunk(2, 0)
        self.assertNotEqual(t2.shape(), t3.shape())

    def test_chunk_same_size(self):
        t1 = TensorBase(np.random.randint(0, 10, size=(4, 10)))
        t2, t3 = t1.chunk(2, 0, same_size=True)
        self.assertEqual(t2.shape(), t3.shape())


class inequalityTest(unittest.TestCase):
    # WARNING: this is an ovverriden function from unittest.Testcase.
    # never change the name of this function.
    def setUp(self):
        self.a1 = np.array([-2, -1, 0, 1, 2])
        self.a2 = np.array([-4, -1, 5, 2, 2])

        self.t1 = TensorBase(self.a1)
        self.t2 = TensorBase(self.a2)

        self.enc = TensorBase(self.a1, encrypted=True)


class gtTests(inequalityTest):
    def test_gt_with_tensor(self):
        self.assertEqual(self.t1.gt(self.t2), self.a1 > self.a2)

    def test_gt_with_number(self):
        self.assertEqual(self.t1.gt(1), self.a1 > 1)

    def test_gt__in_place_with_tensor(self):
        self.t1.gt_(self.t2)
        self.assertEqual(self.t1, self.a1 > self.a2)

    def test_gt__in_place_with_number(self):
        self.t1.gt_(1)
        self.assertEqual(self.t1, self.a1 > 1)

    def test_gt_with_encrypted(self):
        res = self.t1.gt(self.enc)
        self.assertEqual(res, NotImplemented)

        res = self.enc.gt(self.t1)
        self.assertEqual(res, NotImplemented)


class geTests(inequalityTest):
    def test_ge_with_tensor(self):
        self.assertEqual(self.t1.ge(self.t2), self.a1 >= self.a2)

    def test_ge_with_number(self):
        self.assertEqual(self.t1.ge(1), self.a1 >= 1)

    def test_ge__in_place_with_tensor(self):
        self.t1.ge_(self.t2)
        self.assertEqual(self.t1, self.a1 >= self.a2)

    def test_ge__in_place_with_number(self):
        self.t1.ge_(1)
        self.assertEqual(self.t1, self.a1 >= 1)

    def test_ge_with_encrypted(self):
        res = self.t1.ge(self.enc)
        self.assertEqual(res, NotImplemented)

        res = self.enc.ge(self.t1)
        self.assertEqual(res, NotImplemented)


class ltTests(inequalityTest):
    def test_lt_with_tensor(self):
        self.assertEqual(self.t1.lt(self.t2), self.a1 < self.a2)

    def test_lt_with_number(self):
        self.assertEqual(self.t1.lt(1), self.a1 < 1)

    def test_lt__in_place_with_tensor(self):
        self.t1.lt_(self.t2)
        self.assertEqual(self.t1, self.a1 < self.a2)

    def test_lt__in_place_with_number(self):
        self.t1.lt_(1)
        self.assertEqual(self.t1, self.a1 < 1)

    def test_lt_with_encrypted(self):
        res = self.t1.lt(self.enc)
        self.assertEqual(res, NotImplemented)

        res = self.enc.lt(self.t1)
        self.assertEqual(res, NotImplemented)


class leTests(inequalityTest):
    def test_le_with_tensor(self):
        self.assertEqual(self.t1.le(self.t2), self.a1 <= self.a2)

    def test_le_with_number(self):
        self.assertEqual(self.t1.le(1), self.a1 <= 1)

    def test_le__in_place_with_tensor(self):
        self.t1.le_(self.t2)
        self.assertEqual(self.t1, self.a1 <= self.a2)

    def test_le__in_place_with_number(self):
        self.t1.le_(1)
        self.assertEqual(self.t1, self.a1 <= 1)

    def test_le_with_encrypted(self):
        res = self.t1.le(self.enc)
        self.assertEqual(res, NotImplemented)

        res = self.enc.le(self.t1)
        self.assertEqual(res, NotImplemented)


class bernoulliTests(unittest.TestCase):
    def test_bernoulli(self):
        p = TensorBase(np.random.uniform(size=(3, 2)))
        t1 = TensorBase(np.zeros((5, 5)))
        t2 = t1.bernoulli(p)
        self.assertTupleEqual((3, 2), t2.shape())
        self.assertTrue(np.all(t2.data >= 0) and np.all(t2.data <= 1))

    def test_bernoulli_(self):
        p = TensorBase(np.random.uniform(size=(3, 2)))
        t1 = TensorBase(np.zeros((5, 5)))
        t1.bernoulli_(p)
        self.assertTupleEqual((3, 2), t1.shape())
        self.assertTrue(np.all(t1.data >= 0) and np.all(t1.data <= 1))


class cauchyTests(unittest.TestCase):
    def test_cauchy_(self):
        t = TensorBase(np.zeros([1, 2, 3, 4]))
        t.cauchy_()
        self.assertTupleEqual((1, 2, 3, 4), t.shape())
        self.assertTrue(np.all(t.data != 0))


class uniformTests(unittest.TestCase):
    def test_uniform(self):
        t1 = TensorBase(np.zeros(4))
        out = t1.uniform(low=0, high=3)
        self.assertTrue(np.all(out.data > 0) and np.all(out.data < 3))

    def test_uniform_(self):
        t1 = TensorBase(np.zeros(4))
        t1.uniform_(low=0, high=3)
        self.assertTrue(np.all(t1.data > 0) and np.all(t1.data < 3))


class geometricTests(unittest.TestCase):
    def test_geometric_(self):
        t1 = TensorBase(np.zeros((4, 4)))
        out = t1.geometric_(p=0.5)
        self.assertTupleEqual(t1.data.shape, out.data.shape)
        self.assertTrue(np.all(out.data > 0))


class normalTests(unittest.TestCase):
    def test_normal_(self):
        t = TensorBase(np.zeros([1, 2, 3, 4]))
        t.normal_(mu=0, sigma=1)
        self.assertTupleEqual((1, 2, 3, 4), t.shape())
        self.assertTrue(np.all(t.data != 0))

    def test_normal(self):
        t = TensorBase(np.zeros([1, 2, 3, 4]))
        t1 = t.normal(mu=0, sigma=1)
        self.assertTrue(np.array_equal(t.data, np.zeros([1, 2, 3, 4])))
        self.assertTupleEqual((1, 2, 3, 4), t1.shape())
        self.assertTrue(np.all(t1.data != 0))


class fillTests(unittest.TestCase):
    def test_fill_(self):
        t1 = TensorBase(np.array([1, 2, 3, 4]))
        t1.fill_(5)
        self.assertTrue(np.all(t1.data == 5))


class topkTests(unittest.TestCase):
    def test_topK(self):
        t1 = TensorBase(np.array([[900, 800, 1000, 2000, 5, 10, 20, 40, 50], [10, 11, 12, 13, 5, 6, 7, 8, 9], [30, 40, 50, 10, 8, 1, 2, 3, 4]]))
        t2 = t1.topk(3, largest=True)
        self.assertTrue(np.array_equal(t2.data, np.array([[900, 1000, 2000], [11, 12, 13], [30, 40, 50]])))


class tolistTests(unittest.TestCase):
    def test_to_list(self):
        t1 = TensorBase(np.array([200, 300, 90, 100, 600]))
        t1_list = t1.tolist()
        self.assertTrue(isinstance(t1_list, list))


class traceTests(unittest.TestCase):
    def test_trace(self):
        t1 = TensorBase(np.arange(1, 10).reshape(3, 3))
        self.assertTrue(np.equal(t1.trace().data, 15))


class roundTests(unittest.TestCase):
    def test_round(self):
        t1 = TensorBase(np.array([10.4, 9.6, 100.12, 4.0]))
        t2 = t1.round(0)
        self.assertTrue(np.array_equal(t2.data, np.array([10., 10., 100., 4.])))

    def test_round_(self):
        t1 = TensorBase(np.array([10.4, 9.6, 100.12, 4.0]))
        t1.round_(0)
        self.assertTrue(np.array_equal(t1.data, np.array([10., 10., 100., 4.])))


class repeatTests(unittest.TestCase):
    def test_repeat(self):
        t1 = TensorBase(np.array([1, 2, 3]))
        t2 = t1.repeat(reps=(4, 2))
        self.assertTrue(np.array_equal(t2.data, np.array([[1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3]])))


class powTests(unittest.TestCase):
    def test_pow(self):
        t1 = TensorBase(np.array([2, 4, 6]))
        t2 = t1.pow(2)
        self.assertTrue(np.array_equal(t2.data, np.array([4, 16, 36])))

    def test_pow_(self):
        t1 = TensorBase(np.array([2, 4, 6]))
        t1.pow_(2)
        self.assertTrue(np.array_equal(t1.data, np.array([4, 16, 36])))


class negTests(unittest.TestCase):
    def test_neg(self):
        # int
        t1 = TensorBase(np.array([[-0, 1, -2], [0, -1, 2]]))
        t2 = t1.neg()
        self.assertTrue(np.array_equal(t1.data, np.array([[0, 1, -2], [0, -1, 2]])))
        self.assertTrue(np.array_equal(t2.data, np.array([[0, -1, 2], [0, 1, -2]])))
        # float
        t3 = TensorBase(np.array([[-0.0, 1.5, -2.5], [0.0, -1.5, 2.5]]))
        t4 = t3.neg()
        self.assertTrue(np.array_equal(t3.data, np.array([[0.0, 1.5, -2.5], [0.0, -1.5, 2.5]])))
        self.assertTrue(np.array_equal(t4.data, np.array([[0.0, -1.5, 2.5], [0.0, 1.5, -2.5]])))

    def test_neg_(self):
        # int
        t1 = TensorBase(np.array([[-0, 1, -2], [0, -1, 2]]))
        t1.neg_()
        self.assertTrue(np.array_equal(t1.data, np.array([[0, -1, 2], [0, 1, -2]])))
        # float
        t2 = TensorBase(np.array([[-0.0, 1.5, -2.5], [0.0, -1.5, 2.5]]))
        t2.neg_()
        self.assertTrue(np.array_equal(t2.data, np.array([[0.0, -1.5, 2.5], [0.0, 1.5, -2.5]])))


class tanhTests(unittest.TestCase):
    def test_tanh_(self):
        # int
        t1 = TensorBase(np.array([[-0, 1, -2], [0, -1, 2]]))
        t1.tanh_()
        self.assertTrue(np.array_equal(t1.data, np.tanh(np.array([[0, 1, -2], [0, -1, 2]]))))
        # float
        t1 = TensorBase(np.array([[-0.0, 1.5, -2.5], [0.0, -1.5, 2.5]]))
        t1.tanh_()
        self.assertTrue(np.array_equal(t1.data, np.tanh(np.array([[0.0, 1.5, -2.5], [0.0, -1.5, 2.5]]))))


class prodTests(unittest.TestCase):
    def test_prod(self):
        t1 = TensorBase(np.array([1, 2, 3]))
        t2 = t1.prod()
        self.assertTrue(np.equal(t2.data, 6))


class randomTests(unittest.TestCase):
    def test_random_(self):
        np.random.seed(0)
        t1 = TensorBase(np.zeros(4))
        t1.random_(low=0, high=5, size=4)
        self.assertTrue(np.array_equal(t1.data, np.array([4, 0, 3, 3])))


class nonzeroTests(unittest.TestCase):
    def test_non_zero(self):
        t1 = TensorBase(np.array([[1, 0, 0], [0, 2, 5]]))
        t2 = t1.nonzero()
        self.assertTrue(np.array_equal(t2.data, np.array([[0, 1, 1], [0, 1, 2]])))


class cumprodTest(unittest.TestCase):
    def test_cumprod(self):
        t1 = TensorBase(np.array([[1, 2, 3], [4, 5, 6]]))
        t2 = TensorBase(np.array([[1.0, 2.0, 3.0], [4.0, 10.0, 18.0]]))
        t3 = TensorBase(np.array([[1, 2, 6], [4, 20, 120]]))
        self.assertTrue(np.equal(t1.cumprod(dim=0), t2).all())
        self.assertTrue(np.equal(t1.cumprod(dim=1), t3).all())

    def test_cumprod_(self):
        t1 = TensorBase(np.array([[1, 2, 3], [4, 5, 6]]))
        t2 = TensorBase(np.array([[1.0, 2.0, 3.0], [4.0, 10.0, 18.0]]))
        t3 = TensorBase(np.array([[1, 2, 6], [4, 20, 120]]))
        self.assertTrue(np.equal(t1.cumprod_(dim=0), t2).all())
        t1 = TensorBase(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        self.assertTrue(np.equal(t1.cumprod_(dim=1), t3).all())


class splitTests(unittest.TestCase):
    def test_split(self):
        t1 = TensorBase(np.arange(8.0))
        t2 = t1.split(4)
        self.assertTrue(np.array_equal(t2, tuple((np.array([0., 1.]), np.array([2., 3.]), np.array([4., 5.]), np.array([6., 7.])))))


class squeezeTests(unittest.TestCase):
    def test_squeeze(self):
        t1 = TensorBase(np.zeros((2, 1, 2, 1, 2)))
        t2 = t1.squeeze()
        self.assertTrue(np.array_equal(t2.data, np.array([[[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]]])))


class expandAsTests(unittest.TestCase):
    def test_expand_as(self):
        t1 = TensorBase(np.array([[1], [2], [3]]))
        t2 = TensorBase(np.zeros((3, 4)))
        t3 = t1.expand_as(t2)
        self.assertTrue(np.array_equal(t2.data.shape, t3.data.shape))


class meanTests(unittest.TestCase):
    def test_mean(self):
        t1 = TensorBase(np.arange(8).reshape(2, 2, 2))
        t2 = t1.mean(1, True)
        self.assertTrue(np.array_equal(t2.data, np.array([[[1., 2.]], [[5., 6.]]])))


class notEqualTests(unittest.TestCase):
    def test_ne(self):
        t1 = TensorBase([1, 2, 3, 4])
        t2 = TensorBase([1., 2., 3., 5.])
        self.assertEqual(t1.ne(t2), TensorBase([1, 1, 1, 0]))

    def test_ne_(self):
        t1 = TensorBase([1, 2, 3, 4])
        t2 = TensorBase([1., 2., 3., 5.])
        t1.ne_(t2)
        self.assertTrue(syft.equal(t1, TensorBase([1, 1, 1, 0])))


class IndexTests(unittest.TestCase):
    def test_indexing(self):
        t1 = TensorBase(np.array([1.2, 2, 3]))
        self.assertEqual(1.2, t1[0])
        self.assertEqual(3, t1[-1])

    def test_index(self):
        t = TensorBase(np.array([1, 2, 3.5, 4, 5, 6, 3.5]))
        expected1 = TensorBase(np.array(2))
        expected2 = TensorBase(np.array(3.5))
        expected3 = TensorBase(np.array([4, 5, 6]))

        self.assertEqual(expected1, t.index(1))
        self.assertEqual(expected2, t.index(2))
        self.assertEqual(expected2, t.index(-1))
        self.assertEqual(expected3, t.index(slice(3, 6)))
        with pytest.raises(ValueError):
            t.index(3.5)

    def test_index_slice_notation(self):
        t1 = TensorBase(np.array([1, 2, 3, 4]))
        expected1 = TensorBase(np.array([2, 3, 4]))
        expected2 = type(t1[1:])
        expected3 = 1

        # Do not use "t.index" form in following test
        self.assertEqual(expected1, t1[1:])
        self.assertEqual(expected2, TensorBase)
        self.assertEqual(expected3, t1[0])

    def test_index_add_(self):
        t1 = TensorBase(np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1]]))
        t2 = TensorBase(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

        expected_0 = TensorBase(np.array([[1, 2, 3], [8, 9, 10], [5, 6, 7]]))
        t1.index_add_(0, [0, 2, 1], t2)
        self.assertEqual(expected_0, t1)

        t1 = TensorBase(np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1]]))
        expected_1 = TensorBase(np.array([[1, 3, 2], [5, 7, 6], [8, 10, 9]]))
        t1.index_add_(1, [0, 2, 1], t2)
        self.assertEqual(expected_1, t1)

        with pytest.raises(TypeError):
            t1.index_add_(0, [1.0, 2, 2], t2)
        with pytest.raises(IndexError):
            t1.index_add_(0, [0, 1, 2], TensorBase([1, 2]))
        with pytest.raises(ValueError):
            t1.index_add_(0, [0, 1], t2)
        with pytest.raises(ValueError):
            t1.index_add_(0, [0, 1, 5], t2)
        with pytest.raises(IndexError):
            t1.index_add_(4, [0, 1, 2], t2)

    def test_index_copy_(self):
        t1 = TensorBase(np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1]]))
        t2 = TensorBase(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        expected_0 = TensorBase(np.array([[1, 2, 3], [7, 8, 9], [4, 5, 6]]))
        t1.index_copy_(0, [0, 2, 1], t2)
        self.assertEqual(expected_0, t1)

        t1 = TensorBase(np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1]]))
        expected_1 = TensorBase(np.array([[3, 1, 2], [6, 4, 5], [9, 7, 8]]))
        t1.index_copy_(1, [2, 0, 1], t2)
        self.assertEqual(expected_1, t1)

        with pytest.raises(TypeError):
            t1.index_copy_(0, [1.0, 2, 2], t2)
        with pytest.raises(IndexError):
            t1.index_copy_(0, [0, 1, 2], TensorBase([1, 2]))
        with pytest.raises(ValueError):
            t1.index_copy_(0, [0, 1], t2)
        with pytest.raises(ValueError):
            t1.index_copy_(0, [0, 1, 5], t2)
        with pytest.raises(IndexError):
            t1.index_copy_(4, [0, 1, 2], t2)

    def test_index_fill_(self):
        t1 = TensorBase(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        expected_0 = TensorBase(np.array([[1, 1, 1], [1, 1, 1], [7, 8, 9]]))
        t1.index_fill_(0, [0, 1], 1)
        self.assertEqual(expected_0, t1)

        t1 = TensorBase(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        expected_1 = TensorBase(np.array([[-2, 2, -2], [-2, 5, -2], [-2, 8, -2]]))
        t1.index_fill_(1, [0, 2], -2)
        self.assertEqual(expected_1, t1)

        with pytest.raises(TypeError):
            t1.index_fill_(0, [1.0, 2, 2], 1)
        with pytest.raises(ValueError):
            t1.index_fill_(0, [0, 1, 5], 1)
        with pytest.raises(IndexError):
            t1.index_fill_(4, [0, 1, 2], 1)

    def test_index_select(self):
        t = TensorBase(np.reshape(np.arange(0, 2 * 3 * 4), (2, 3, 4)))
        idx = np.array([1, 0])
        dim = 2
        result = t.index_select(dim=dim, index=idx)
        expected = np.array([[[1, 0], [5, 4], [9, 8]], [[13, 12], [17, 16], [21, 20]]])
        self.assertTrue(np.array_equal(result.data, expected))


class gatherTests(unittest.TestCase):
    def test_gather_numerical_1(self):
        t = TensorBase(np.array([[65, 17], [14, 25], [76, 22]]))
        idx = TensorBase(np.array([[0], [1], [0]]))
        dim = 1
        result = t.gather(dim=dim, index=idx)
        self.assertTrue(np.array_equal(result.data, np.array([[65], [25], [76]])))

    def test_gather_numerical_2(self):
        t = TensorBase(np.array([[47, 74, 44], [56, 9, 37]]))
        idx = TensorBase(np.array([[0, 0, 1], [1, 1, 0], [0, 1, 0]]))
        dim = 0
        result = t.gather(dim=dim, index=idx)
        expexted = [[47, 74, 37], [56, 9, 44.], [47, 9, 44]]
        self.assertTrue(np.array_equal(result.data, np.array(expexted)))


class scatterTests(unittest.TestCase):
    def test_scatter_numerical_0(self):
        t = TensorBase(np.zeros((3, 5)))
        idx = TensorBase(np.array([[0, 0, 0, 0, 0]]))
        src = 1.0
        dim = 0
        t.scatter_(dim=dim, index=idx, src=src)
        self.assertTrue(np.array_equal(t.data, np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])))

    def test_scatter_numerical_1(self):
        t = TensorBase(np.zeros((3, 5)))
        idx = TensorBase(np.array([[0], [0], [0]]))
        src = 1.0
        dim = 1
        t.scatter_(dim=dim, index=idx, src=src)
        self.assertTrue(np.array_equal(t.data, np.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]])))

    def test_scatter_numerical_2(self):
        t = TensorBase(np.zeros((3, 5)))
        idx = TensorBase(np.array([[0], [0], [0]]))
        src = 1.0
        dim = -1
        t.scatter_(dim=dim, index=idx, src=src)
        self.assertTrue(np.array_equal(t.data, np.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]])))

    def test_scatter_numerical_3(self):
        t = TensorBase(np.zeros((3, 5)))
        idx = TensorBase(np.array([[0, 0, 0, 0, 0]]))
        src = TensorBase(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]))
        dim = 0
        t.scatter_(dim=dim, index=idx, src=src)
        self.assertTrue(np.array_equal(t.data, np.array([[1, 2, 3, 4, 5], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])))

    def test_scatter_numerical_4(self):
        t = TensorBase(np.zeros((3, 5)))
        idx = TensorBase(np.array([[0, 0, 0, 0, 0]]))
        src = TensorBase(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]))
        dim = -2
        t.scatter_(dim=dim, index=idx, src=src)
        self.assertTrue(np.array_equal(t.data, np.array([[1, 2, 3, 4, 5], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])))

    def test_scatter_numerical_5(self):
        t = TensorBase(np.zeros((3, 5)))
        idx = TensorBase(np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]))
        src = TensorBase(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]))
        dim = 0
        t.scatter_(dim=dim, index=idx, src=src)
        self.assertTrue(np.array_equal(t.data, np.array([[6, 7, 8, 9, 10], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])))

    def test_scatter_numerical_6(self):
        t = TensorBase(np.zeros((3, 4, 5)))
        idx = [[[3, 0, 1, 1, 2], [0, 3, 3, 3, 3]], [[2, 0, 0, 0, 0], [2, 1, 0, 2, 0]],
               [[0, 0, 1, 0, 2], [1, 3, 2, 2, 2]]]
        src = [[[7, 84, 99, 71, 44], [79, 57, 2, 37, 62]], [[31, 44, 43, 54, 56], [72, 52, 21, 89, 95]],
               [[5, 3, 99, 4, 52], [32, 88, 58, 62, 9]]]
        dim = 1
        t.scatter_(dim=dim, index=idx, src=src)
        expected = [[[79, 84, 0, 0, 0], [0, 0, 99, 71, 0], [0, 0, 0, 0, 44], [7, 57, 2, 37, 62]],
                    [[0, 44, 21, 54, 95], [0, 52, 0, 0, 0], [72, 0, 0, 89, 0], [0, 0, 0, 0, 0]],
                    [[5, 3, 0, 4, 0], [32, 0, 99, 0, 0], [0, 0, 58, 62, 9], [0, 88, 0, 0, 0]]]
        self.assertTrue(np.array_equal(t.data, np.array(expected)))

    def test_scatter_index_type(self):
        t = TensorBase(np.zeros((3, 5)))
        idx = TensorBase(np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]))
        src = TensorBase(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]))
        dim = 0
        with self.assertRaises(Exception):
            t.scatter_(dim=dim, index=idx, src=src)

    def test_scatter_index_out_of_range(self):
        t = TensorBase(np.zeros((3, 5)))
        idx = TensorBase(np.array([[5, 0, 0, 0, 0]]))
        src = TensorBase(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]))
        dim = 0
        with self.assertRaises(Exception):
            t.scatter_(dim=dim, index=idx, src=src)

    def test_scatter_dim_out_Of_range(self):
        t = TensorBase(np.zeros((3, 5)))
        idx = TensorBase(np.array([[0, 0, 0, 0, 0]]))
        src = TensorBase(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]))
        dim = 4
        with self.assertRaises(Exception):
            t.scatter_(dim=dim, index=idx, src=src)

    def test_scatter_index_src_dimension_mismatch(self):
        t = TensorBase(np.zeros((3, 5)))
        idx = TensorBase(np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]))
        src = TensorBase(np.array([[1, 2, 3, 4, 5]]))
        dim = 1
        with self.assertRaises(Exception):
            t.scatter_(dim=dim, index=idx, src=src)


class remainderTests(unittest.TestCase):
    def tes_remainder(self):
        t = TensorBase([[-2, -3], [4, 1]])
        result = t.remainder(1.5)
        self.assertTrue(np.array_equal(result.data, np.array([[1, 0], [1, 1]])))

    def test_remainder_broadcasting(self):
        t = TensorBase([[-2, -3], [4, 1]])
        result = t.remainder([2, -3])
        self.assertTrue(np.array_equal(result.data, np.array([[0, 0], [0, -2]])))

    def test_remainder_(self):
        t = TensorBase([[-2, -3], [4, 1]])
        t.remainder_(2)
        self.assertTrue(np.array_equal(t.data, np.array([[0, 1], [0, 1]])))


class testMv(unittest.TestCase):
    def test_mv(self):
        mat = TensorBase([[1, 2, 3], [2, 3, 4], [4, 5, 6]])
        vector = TensorBase([1, 2, 3])
        self.assertEqual(tensor.mv(mat, vector), TensorBase([14, 20, 32]))

    def test_mv_tensor(self):
        mat = TensorBase([[1, 2, 3], [1, 2, 3]])
        vec = TensorBase([1, 2, 3])
        self.assertEqual(mat.mv(vec), TensorBase([14, 14]))


class TestNarrow(unittest.TestCase):
    def test_narrow_int(self):
        mat = TensorBase(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        dim, start, length = 0, 0, 2
        self.assertEqual(mat.narrow(dim, start, length), TensorBase(np.array([[1, 2, 3], [4, 5, 6]])))

    def test_narrow_float(self):
        mat = TensorBase(np.array([[1.1, 2.1, 3.1], [4.2, 5.2, 6.2], [7.3, 8.3, 9.3]]))
        dim, start, length = 1, 1, 2
        self.assertEqual(mat.narrow(dim, start, length), TensorBase(np.array([[2.1, 3.1], [5.2, 6.2], [8.3, 9.3]])))


class masked_scatter_Tests(unittest.TestCase):
    def test_masked_scatter_1(self):
        t = TensorBase(np.ones((2, 3)))
        source = TensorBase([1, 2, 3, 4, 5, 6])
        mask = TensorBase([[0, 1, 0], [1, 0, 1]])
        t.masked_scatter_(mask, source)
        self.assertTrue(np.array_equal(t, TensorBase([[1, 1, 1], [2, 1, 3]])))

    def test_masked_scatter_braodcasting_1(self):
        t = TensorBase(np.ones((2, 3)))
        source = TensorBase([1, 2, 3, 4, 5, 6])
        mask = TensorBase([0, 1, 0])
        t.masked_scatter_(mask, source)
        self.assertTrue(np.array_equal(t, TensorBase([[1, 1, 1], [1, 2, 1]])))

    def test_masked_scatter_braodcasting_2(self):
        t = TensorBase(np.ones((2, 3)))
        source = TensorBase([1, 2, 3, 4, 5, 6])
        mask = TensorBase([[1], [0]])
        t.masked_scatter_(mask, source)
        self.assertTrue(np.array_equal(t, TensorBase([[1, 2, 3], [1, 1, 1]])))


class masked_fill_Tests(unittest.TestCase):
    def test_masked_fill_(self):
        t = TensorBase(np.ones((2, 3)))
        value = 2.0
        mask = TensorBase([[0, 0, 0], [1, 1, 0]])
        t.masked_fill_(mask, value)
        self.assertTrue(np.array_equal(t, TensorBase([[1.0, 1.0, 1.0], [2.0, 2.0, 1.0]])))

    def test_masked_fill_broadcasting(self):
        t = TensorBase(np.ones((2, 3)))
        value = 2
        mask = TensorBase([[1], [0]])
        t.masked_fill_(mask, value)
        self.assertTrue(np.array_equal(t, TensorBase([[2, 2, 2], [1, 1, 1]])))


class masked_select_Tests(unittest.TestCase):
    def test_masked_select(self):
        t = TensorBase(np.arange(0, 6).reshape(2, 3))
        mask = TensorBase([[0, 0, 0], [1, 1, 0]])
        self.assertTrue(np.array_equal(tensor.masked_select(t, mask), TensorBase([3, 4])))

    def test_masked_select_broadcasting_1(self):
        t = TensorBase(np.arange(0, 6).reshape(2, 3))
        mask = TensorBase([[1, 1, 0]])
        self.assertTrue(np.array_equal(tensor.masked_select(t, mask), TensorBase([0, 1, 3, 4])))

    def test_masked_select_broadcasting_2(self):
        t = TensorBase([2.0])
        mask = TensorBase([[1, 1, 0]])
        self.assertTrue(np.array_equal(tensor.masked_select(t, mask), TensorBase([2.0, 2.0])))

    def test_tensor_base_masked_select(self):
        t = TensorBase(np.arange(0, 6).reshape(2, 3))
        mask = TensorBase([[1, 1, 0]])
        self.assertTrue(np.array_equal(t.masked_select(mask), TensorBase([0, 1, 3, 4])))


class eqTests(unittest.TestCase):
    def test_eq_with_tensor(self):
        t1 = TensorBase(np.arange(5))
        t2 = TensorBase(np.arange(5)[-1::-1])
        truth_values = t1.eq(t2)
        self.assertEqual(truth_values, [False, False, True, False, False])

    def test_eq_with_number(self):
        t1 = TensorBase(np.arange(5))
        truth_values = t1.eq(1)
        self.assertEqual(truth_values, [False, True, False, False, False])

    def test_eq_in_place_with_tensor(self):
        t1 = TensorBase(np.arange(5))
        t2 = TensorBase(np.arange(5)[-1::-1])
        t1.eq_(t2)
        self.assertEqual(t1, [False, False, True, False, False])

    def test_eq_in_place_with_number(self):
        t1 = TensorBase(np.arange(5))
        t1.eq_(1)
        self.assertEqual(t1, [False, True, False, False, False])


class mm_test(unittest.TestCase):
    def test_mm_1d(self):
        t1 = TensorBase(np.array([2, 3, 4]))
        t2 = TensorBase(np.array([3, 4, 5]))
        out = t1.mm(t2)
        self.assertTrue(np.alltrue(out.data == [38]))

    def test_mm_2d(self):
        t1 = TensorBase(np.array([[1, 2], [1, 2]]))
        t2 = TensorBase(np.array([[2, 3], [2, 3]]))
        out = t1.mm(t2)
        self.assertTrue(np.alltrue(out.data == [[6, 9], [6, 9]]))

    def test_mm_3d(self):
        t1 = TensorBase(np.array([[1, 2], [2, 3], [3, 4]]))
        t2 = TensorBase(np.array([[1, 2, 3], [2, 3, 4]]))
        out = t1.mm(t2)
        self.assertTrue(np.alltrue(out.data == [[5, 8, 11], [8, 13, 18], [11, 18, 25]]))


class newTensorTests(unittest.TestCase):
    def test_encrypted_error(self):

        t1 = TensorBase(np.array([1, 1, 1]), encrypted=True)
        t2 = t1.new([1, 1, 2], encrypted=True)

        self.assertEqual(t2, NotImplemented)

    def test_return_new_float_tensor(self):

        t1 = TensorBase(np.array([1, 1, 1]))
        t2 = t1.new(np.array([1., 1., 2.]))

        self.assertTrue(t2.data.dtype == np.float64)

    def test_return_new_int_tensor(self):

        t1 = TensorBase(np.array([1, 1, 1]))
        t2 = t1.new(np.array([1, 1, 2]))

        self.assertTrue(t2.data.dtype == np.int64)


class half(unittest.TestCase):
    def half_test_1(self):
        t1 = TensorBase(np.array([2, 3, 4]))
        self.assertTrue(np.alltrue(t1.half() == np.array([2, 3, 4]).astype('float16')))

    def half_test_2(self):
        t1 = TensorBase(np.array([[1.1, 2.1], [1.11, 2.11]]))
        self.assertTrue(np.alltrue(t1.half() == np.array([[1.1, 2.1], [1.11, 2.11]]).astype('float16')))


class fmodTest(unittest.TestCase):
    def test_fmod_number(self):
        t1 = TensorBase(np.array([-3, -2, -1, 1, 2, 3]))
        self.assertTrue(np.array_equal(t1.fmod(2).data, np.array([-1, 0, -1, 1, 0, 1])))
        t2 = TensorBase(np.array([-3.5, -2.5, -1.5, 1.5, 2.5, 3.5]))
        self.assertTrue(np.array_equal(t2.fmod(2.).data, np.array([-1.5, -0.5, -1.5, 1.5, 0.5, 1.5])))

    def test_fmod_tensor(self):
        t1 = TensorBase(np.array([-3, -2, -1, 1, 2, 3]))
        divisor = np.array([2] * 6)
        self.assertTrue(np.array_equal(t1.fmod(divisor).data, np.array([-1, 0, -1, 1, 0, 1])))
        t2 = TensorBase(np.array([-3.5, -2.5, -1.5, 1.5, 2.5, 3.5]))
        divisor = np.array([2.] * 6)
        self.assertTrue(np.array_equal(t2.fmod(divisor).data, np.array([-1.5, -0.5, -1.5, 1.5, 0.5, 1.5])))


class fmod_Test(unittest.TestCase):
    def test_fmod_number(self):
        t1 = TensorBase(np.array([-3, -2, -1, 1, 2, 3]))
        t1.fmod_(2)
        self.assertTrue(np.array_equal(t1.data, np.array([-1, 0, -1, 1, 0, 1])))
        t2 = TensorBase(np.array([-3.5, -2.5, -1.5, 1.5, 2.5, 3.5]))
        t2.fmod_(2.)
        self.assertTrue(np.array_equal(t2.data, np.array([-1.5, -0.5, -1.5, 1.5, 0.5, 1.5])))

    def test_fmod_tensor(self):
        t1 = TensorBase(np.array([-3, -2, -1, 1, 2, 3]))
        divisor = np.array([2] * 6)
        t1.fmod_(divisor)
        self.assertTrue(np.array_equal(t1.data, np.array([-1, 0, -1, 1, 0, 1])))
        t2 = TensorBase(np.array([-3.5, -2.5, -1.5, 1.5, 2.5, 3.5]))
        divisor = np.array([2.] * 6)
        t2.fmod_(divisor)
        self.assertTrue(np.array_equal(t2.data, np.array([-1.5, -0.5, -1.5, 1.5, 0.5, 1.5])))


class numel_test(unittest.TestCase):
    def numel_test_int(self):
        t1_len = 3
        t1 = TensorBase(np.array([2, 3, 4]))
        self.assertEqual(t1.numel(), t1_len)

    def numel_test_float(self):
        t1_len = 3
        t1 = TensorBase(np.array([2.0, 3.0, 4.0]))
        self.assertEqual(t1.numel(), t1_len)

    def numel_test_2d(self):
        t1_len = 6
        t1 = TensorBase(np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]))
        self.assertEqual(t1.numel(), t1_len)

    def numel_test_3d(self):
        t1_len = 12
        t1 = TensorBase(np.array([[[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]]))
        self.assertEqual(t1.numel(), t1_len)

    def numel_test_str(self):
        t1_len = 4
        t1 = TensorBase(np.array(['a', 'b', 'c', 'd']))
        self.assertEqual(t1.numel(), t1_len)

    def numel_test_encrypted(self):
        t1_len = 4
        t1 = TensorBase(np.array(['a', 'b', 'c', 'd']))
        t1enc = TensorBase(t1, encrypted=True)
        self.assertEqual(t1enc.numel(), t1_len)


class nelement_test(unittest.TestCase):
    def nelement_test_int(self):
        t1_len = 3
        t1 = TensorBase(np.array([2, 3, 4]))
        self.assertEqual(t1.nelement(), t1_len)

    def nelement_test_float(self):
        t1_len = 3
        t1 = TensorBase(np.array([2.0, 3.0, 4.0]))
        self.assertEqual(t1.nelement(), t1_len)

    def nelement_test_2d(self):
        t1_len = 6
        t1 = TensorBase(np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]))
        self.assertEqual(t1.nelement(), t1_len)

    def nelement_test_3d(self):
        t1_len = 12
        t1 = TensorBase(np.array([[[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]]))
        self.assertEqual(t1.nelement(), t1_len)

    def nelement_test_str(self):
        t1_len = 4
        t1 = TensorBase(np.array(['a', 'b', 'c', 'd']))
        self.assertEqual(t1.nelement(), t1_len)

    def nelement_test_encrypted(self):
        t1_len = 4
        t1 = TensorBase(np.array(['a', 'b', 'c', 'd']))
        t1enc = TensorBase(t1, encrypted=True)
        self.assertEqual(t1enc.nelement(), t1_len)


class size_test(unittest.TestCase):
    def size_test_int(self):
        t1_size = (3,)
        t1 = TensorBase(np.array([2, 3, 4]))
        self.assertEqual(t1.size(), t1_size)

    def size_test_float(self):
        t1_size = (3,)
        t1 = TensorBase(np.array([2.0, 3.0, 4.0]))
        self.assertEqual(t1.size(), t1_size)

    def size_test_2d(self):
        t1_size = (2, 3)
        t1 = TensorBase(np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]))
        self.assertEqual(t1.size(), t1_size)

    def size_test_3d(self):
        t1_size = (2, 2, 3)
        t1 = TensorBase(np.array([[[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]]))
        self.assertEqual(t1.size(), t1_size)

    def size_test_str(self):
        t1_size = (4,)
        t1 = TensorBase(np.array(['a', 'b', 'c', 'd']))
        self.assertEqual(t1.size(), t1_size)


class lerpTests(unittest.TestCase):
    def test_lerp(self):
        t1 = TensorBase(np.array([1, 2, 3, 4]))
        t2 = TensorBase(np.array([3, 4, 5, 6]))
        weight = 0.5
        out = t1.lerp(t2, weight)
        self.assertTrue(np.array_equal(out.data, [2, 3, 4, 5]))

    def test_lerp_(self):
        t1 = TensorBase(np.array([1, 2, 3, 4]))
        t2 = TensorBase(np.array([3, 4, 5, 6]))
        weight = 0.5
        t1.lerp_(t2, weight)
        self.assertTrue(np.array_equal(t1.data, [2, 3, 4, 5]))


class RenormTests(unittest.TestCase):
    def testRenorm(self):
        t1 = TensorBase(np.array([[1, 2, 3], [4, 5, 6]]))
        t2 = t1.renorm(2, 0, 6)
        self.assertTrue(np.allclose(t2, np.array([[1.0, 2.0, 3.0], [2.735054, 3.418817, 4.102581]])))

    def testRenorm_(self):
        t = TensorBase(np.array([[1, 2, 3], [4, 5, 6]]))
        t.renorm_(2, 0, 6)
        self.assertTrue(np.allclose(t, np.array([[1.0, 2.0, 3.0], [2.735054, 3.418817, 4.102581]])))


class unfold_Test(unittest.TestCase):
    def unfold_test_small(self):
        t1 = TensorBase(np.arange(1, 5))
        t1_unfolded_actual = t1.unfold(0, 1, 1).to_numpy()
        t1_unfolded_expected = np.array([[1], [2], [3], [4]])
        self.assertTrue(np.array_equal(t1_unfolded_expected,
                                       t1_unfolded_actual))
        t1_unfolded_actual_1 = t1.unfold(0, 1, 2).to_numpy()
        t1_unfolded_expected_1 = np.array([[1], [3]])
        self.assertTrue(np.array_equal(t1_unfolded_expected_1,
                                       t1_unfolded_actual_1))
        t1_unfolded_actual_2 = t1.unfold(-1, 1, 2).to_numpy()
        t1_unfolded_expected_2 = np.array([[1], [3]])
        self.assertTrue(np.array_equal(t1_unfolded_expected_2,
                                       t1_unfolded_actual_2))

    def unfold_test_big(self):
        arr = np.array(
            [[[1, 3], [2, 4], [0, 2], [1, 4]], [[1, 3], [0, 0], [1, 2], [2, 1]], [[3, 1], [2, 2], [1, 0], [2, 1]],
             [[4, 1], [4, 1], [0, 2], [0, 4]], [[4, 4], [2, 2], [0, 1], [1, 3]]])
        t1 = TensorBase(arr)
        t1_unfolded_actual = t1.unfold(0, 1, 1).to_numpy()
        t1_unfolded_expected = np.array(
            [[[[1], [3]], [[2], [4]], [[0], [2]], [[1], [4]]], [[[1], [3]], [[0], [0]], [[1], [2]], [[2], [1]]],
             [[[3], [1]], [[2], [2]], [[1], [0]], [[2], [1]]], [[[4], [1]], [[4], [1]], [[0], [2]], [[0], [4]]],
             [[[4], [4]], [[2], [2]], [[0], [1]], [[1], [3]]]])
        self.assertTrue(np.array_equal(t1_unfolded_expected,
                                       t1_unfolded_actual))
        t1_unfolded_actual_1 = t1.unfold(2, 1, 2).to_numpy()
        t1_unfolded_expected_1 = np.array(
            [[[[1]], [[2]], [[0]], [[1]]], [[[1]], [[0]], [[1]], [[2]]], [[[3]], [[2]], [[1]], [[2]]],
             [[[4]], [[4]], [[0]], [[0]]], [[[4]], [[2]], [[0]], [[1]]]])
        self.assertTrue(np.array_equal(t1_unfolded_expected_1,
                                       t1_unfolded_actual_1))
        t1_unfolded_actual_2 = t1.unfold(-1, 1, 2).to_numpy()
        t1_unfolded_expected_2 = t1_unfolded_expected_1
        self.assertTrue(np.array_equal(t1_unfolded_expected_2,
                                       t1_unfolded_actual_2))


if __name__ == "__main__":

    unittest.main()
