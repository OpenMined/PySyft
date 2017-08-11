from syft import TensorBase
import unittest
import numpy as np

# Here's our "unit tests".
class AddTests(unittest.TestCase):
    def testSimple(self):
        t = TensorBase(np.array([1,2,3]))
        self.assertTrue(np.array_equal(t + np.array([1,2,3]), [2,4,6]))
    
    def testInplace(self):
        t = TensorBase(np.array([1,2,3]))
        t += np.array([1,2,3])
        self.assertTrue(np.array_equal(t.data, [2,4,6]))

    def testScalar(self):
        t = TensorBase(np.array([1,2,3]))
        self.assertTrue(np.array_equal(t + 2, [3, 4, 5]))

class SubTests(unittest.TestCase):
    def testSimple(self):
        t = TensorBase(np.array([1,2,3]))
        self.assertTrue(np.array_equal(t - np.array([1,2,3]), [0,0,0]))
    
    def testInplace(self):
        t = TensorBase(np.array([1,2,3]))
        t -= np.array([1,2,3])
        self.assertTrue(np.array_equal(t.data, [0,0,0]))

    def testScalar(self):
        t = TensorBase(np.array([1,2,3]))
        self.assertTrue(np.array_equal(t - 1, [0, 1, 2]))

class MultTests(unittest.TestCase):
    def testSimple(self):
        t = TensorBase(np.array([1,2,3]))
        self.assertTrue(np.array_equal(t * np.array([1,2,3]), [1,4,9]))

    def testInplace(self):
        t = TensorBase(np.array([1,2,3]))
        t *= np.array([1,2,3])
        self.assertTrue(np.array_equal(t.data, [1,4,9]))

    def testScalar(self):
        t = TensorBase(np.array([1,2,3]))
        self.assertTrue(np.array_equal(t * 2, [2, 4, 6]))

class DivTests(unittest.TestCase):
    def testSimple(self):
        t = TensorBase(np.array([2,4,8]))
        self.assertTrue(np.array_equal(t / np.array([2,2,2]), [1,2,4]))

    def testInplace(self):
        t = TensorBase(np.array([1,2,3]))
        t *= np.array([1,2,3])
        self.assertTrue(np.array_equal(t.data, [1,4,9]))

    def testScalar(self):
        t = TensorBase(np.array([2,4,6]))
        self.assertTrue(np.array_equal(t / 2, [1, 2, 3]))
    
def main():
    unittest.main()

if __name__ == '__main__':
    main()