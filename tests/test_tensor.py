from syft import TensorBase
import syft
import unittest
import numpy as np


# Here's our "unit tests".
class DimTests(unittest.TestCase):
    def testDimOne(self):
        t = TensorBase(np.array([1, 2, 3]))
        self.assertTrue(t.dim() == 1)


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
        t = TensorBase(np.array([1.4,2.7,6.2]))
        self.assertTrue(syft.equal(t.ceil(),[2,3,7]))


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
        t = TensorBase(np.array([1, 2, 3]))
        t *= np.array([1, 2, 3])
        self.assertTrue(syft.equal(t.data, [1, 4, 9]))

    def testScalar(self):
        t = TensorBase(np.array([2, 4, 6]))
        self.assertTrue(syft.equal(t / 2, [1, 2, 3]))


class AbsTests(unittest.TestCase):
       
    def testabs(self):
        t = TensorBase(np.array([-1,-2,3]))
        self.assertTrue(np.array_equal(t.abs(),[1,2,3]))
    def testabs_(self):
        t = TensorBase(np.array([-1,-2,3]))
        self.assertTrue(np.array_equal(t.abs_(),t.data))  

class ShapeTests(unittest.TestCase):
    def testShape(self):
        t = TensorBase(np.array([[0, 1], [0, 5]]))
        self.assertTrue(syft.equal(t.shape(), (2, 2)))


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
        self.assertEqual(1.2, t1[0].data)
        self.assertEqual(3, t1[-1].data)


class addmm(unittest.TestCase):
    def testaddmm1d(self):
        t1=TensorBase(np.array([1,2,3]))
        t2=TensorBase(np.array([2,3,4]))
        mat=TensorBase(np.array([5]))
        out=t1.addmm(t2,mat,beta=2,alpha=2)
        self.assertTrue(np.array_equal(out.data,[50]))
    
    def testaddmm2d(self):
        t1=TensorBase(np.array([[1,2],[1,2]]))
        t2=TensorBase(np.array([[1,2],[1,2]]))
        mat=TensorBase(np.array([[2,3],[3,4]]))
        out=t1.addmm(t2,mat,beta=2,alpha=2)
        self.assertTrue(np.array_equal(out.data,[[10,18],[12,20]]))
    
    def testaddmm_1d(self):
        t1=TensorBase(np.array([1,2,3]))
        t2=TensorBase(np.array([2,3,4]))
        mat=TensorBase(np.array([5]))
        t1.addmm_(t2,mat,beta=2,alpha=2)
        self.assertTrue(np.array_equal(t1.data,[50]))
    
    def testaddmm_2d(self):
        t1=TensorBase(np.array([[1,2],[1,2]]))
        t2=TensorBase(np.array([[1,2],[1,2]]))
        mat=TensorBase(np.array([[2,3],[3,4]]))
        t1.addmm_(t2,mat,beta=2,alpha=2)
        self.assertTrue(np.array_equal(t1.data,[[10,18],[12,20]]))
