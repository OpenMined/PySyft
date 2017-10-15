from syft import TensorBase
import numpy as np
from syft.nonlin import SigmoidActivation, SquareActivation, LinearSquareActivation
from syft.nonlin import PolyFunction, CubicActivation, LinearCubicActivation
import unittest


# Here's our "unit tests".
class ActivationTests(unittest.TestCase):
    def test_sigmoid(self):
        
        a = TensorBase(np.array([1, 2, 3]))
        sigmoid = SigmoidActivation()
        approx = sigmoid(a)
        self.assertEqual(approx[0], 0.70788285770000015)
        self.assertEqual(approx[1], 0.87170293820000011)
        self.assertEqual(approx[2], 0.96626517229999997)
        
    def test_square(self):
        
        a = TensorBase(np.array([1, 2, 3]))
        square = SquareActivation()
        approx = square(a)
        self.assertEqual(approx[0], 1)
        self.assertEqual(approx[1], 4)
        self.assertEqual(approx[2], 9)
        
    def test_square_derivative(self):
        
        a = TensorBase(np.array([1, 2, 3]))
        square = SquareActivation()
        approx = square.derivative(a)
        self.assertEqual(approx[0], 2)
        self.assertEqual(approx[1], 4)
        self.assertEqual(approx[2], 6)
        
    def test_linear_square(self):
        
        a = TensorBase(np.array([1, 2, 3]))
        linear_square = LinearSquareActivation()
        approx = linear_square(a)
        self.assertEqual(approx[0], 2)
        self.assertEqual(approx[1], 6)
        self.assertEqual(approx[2], 12)
        
    def test_linear_square_derivative(self):
        
        a = TensorBase(np.array([1, 2, 3]))
        linear_square = LinearSquareActivation()
        approx = linear_square.derivative(a)
        self.assertEqual(approx[0], 3)
        self.assertEqual(approx[1], 5)
        self.assertEqual(approx[2], 7)    
        
    def test_cubic(self):
        
        a = TensorBase(np.array([1, 2, 3]))
        cubic = CubicActivation()
        approx = cubic(a)
        self.assertEqual(approx[0], 1)
        self.assertEqual(approx[1], 8)
        self.assertEqual(approx[2], 27)
        
    def test_cubic_derivative(self):
        
        a = TensorBase(np.array([1, 2, 3]))
        cubic = CubicActivation()
        approx = cubic.derivative(a)
        self.assertEqual(approx[0], 3)
        self.assertEqual(approx[1], 12)
        self.assertEqual(approx[2], 27)
        
    def test_linear_cubic(self):
        
        a = TensorBase(np.array([1, 2, 3]))
        linear_cubic = LinearCubicActivation()
        approx = linear_cubic(a)
        self.assertEqual(approx[0], 2)
        self.assertEqual(approx[1], 10)
        self.assertEqual(approx[2], 30)
        
    def test_linear_cubic_derivative(self):
        
        a = TensorBase(np.array([1, 2, 3]))
        linear_cubic = LinearCubicActivation()
        approx = linear_cubic.derivative(a)
        self.assertEqual(approx[0], 4)
        self.assertEqual(approx[1], 13)
        self.assertEqual(approx[2], 28)  

        
class TestPolyApproximators(unittest.TestCase):
    def test_poly_approx_sigmoid(self):

        sigmoid = PolyFunction.from_approximation(lambda x: 1 / (1 + np.exp(-x)))

        a = TensorBase(np.array([.1, .2, .3, .4]))
        b = TensorBase(np.ones(4))

        siga = sigmoid(a)
        sigb = sigmoid(b)

        self.assertTrue(np.abs(siga[0] - 0.52158376423960751) < 0.0001)
        self.assertTrue(np.abs(siga[1] - 0.54311827756855013) < 0.0001)

        self.assertTrue(np.abs(sigb[0] - 0.7078828574) < 0.0001)
        self.assertTrue(np.abs(sigb[1] - 0.7078828574) < 0.0001)

    def test_poly_approx_tanh(self):

        tanh = PolyFunction.from_approximation(np.tanh)

        a = TensorBase(np.array([.1, .2, .3, .4]))
        b = TensorBase(np.ones(4))

        tana = tanh(a)
        tanb = tanh(b)

        self.assertTrue(np.abs(tana[0] - 0.05772423) < 0.0001)
        self.assertTrue(np.abs(tana[1] - 0.11527286) < 0.0001)

        self.assertTrue(np.abs(tanb[1] - 0.54896505) < 0.0001)

    def test_function_fitting(self):
        square = lambda x: x**2
        square_approx = PolyFunction.fit_function(square,2,1,np.asarray([-1,0,1]))
        self.assertTrue(np.abs(square_approx[0]-1) < 0.0001)
        self.assertTrue(np.abs(square_approx[1]) < 0.0001)
        self.assertTrue(np.abs(square_approx[2]) < 0.0001)
        
    def test_function_approx(self):
        square = lambda x: x**2
        square_approx = PolyFunction.from_approximation(square,degree=2,precision=1,num=3)
        coefs = square_approx.coefs
        self.assertTrue(np.abs(coeffs[0]-1) < 0.0001)
        self.assertTrue(np.abs(coeffs[1]) < 0.0001)
        self.assertTrue(np.abs(coeffs[2]) < 0.0001)
        
        derivative_coefs = square_approx.derivative_coefs
        self.assertTrue(np.abs(derivative_coefs[0]-2) < 0.0001)
        self.assertTrue(np.abs(derivative_coefs[1]) < 0.0001)