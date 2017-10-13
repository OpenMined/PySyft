from syft import TensorBase
import numpy as np
from syft.nonlin import sigmoid, PolyApproximator
import unittest


# Here's our "unit tests".
class NonlinTests(unittest.TestCase):
    def test_sigmoid(self):

        a = TensorBase(np.array([1, 2, 3]))
        approx = sigmoid(a)
        self.assertEqual(approx[0], 0.70788285770000015)
        self.assertEqual(approx[1], 0.87170293820000011)
        self.assertEqual(approx[2], 0.96626517229999997)


class TestPolyApproximators(unittest.TestCase):
    def test_poly_approx_sigmoid(self):

        sigmoid = PolyApproximator(lambda x: 1 / (1 + np.exp(-x))).output

        a = TensorBase(np.array([.1, .2, .3, .4]))
        b = TensorBase(np.ones(4))

        siga = sigmoid(a)
        sigb = sigmoid(b)

        self.assertTrue(np.abs(siga[0] - 0.52158376423960751) < 0.0001)
        self.assertTrue(np.abs(siga[1] - 0.54311827756855013) < 0.0001)

        self.assertTrue(np.abs(sigb[1] - 0.7078828574) < 0.0001)

    def test_poly_approx_tanh(self):

        tanh = PolyApproximator(np.tanh).output

        a = TensorBase(np.array([.1, .2, .3, .4]))
        b = TensorBase(np.ones(4))

        tana = tanh(a)
        tanb = tanh(b)

        self.assertTrue(np.abs(tana[0] - 0.05772423) < 0.0001)
        self.assertTrue(np.abs(tana[1] - 0.11527286) < 0.0001)

        self.assertTrue(np.abs(tanb[1] - 0.54896505) < 0.0001)
