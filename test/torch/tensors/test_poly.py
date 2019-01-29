import syft 
from syft.frameworks.torch.tensors.Polynomial import PolynomialTensor

import math

class TestPolynomialTensor:

    """ Test cases to ensure working of Polynomial Tensor. The tests under these ensure that the error between actual funcion values and approximations do not deviate too much"""

    def __init__(self):

        self.tensor = PolynomialTensor()
        # Maximum permissible error as calculated by EvalRelative under PolynomialTensor
        self.threshold = 0.1

    def EvalRelative_Test(self, x_true, x_pred):

        assert self.tensor.EvalRelative(15 / 7.5) == 0.5
        assert self.tensor.EvalRelative(15 / 1.5) == 0.9
        assert self.tensor.EvalRelative(15 / 15) == 0

    def SigmoidTest(self):

        test_range = 4
        for i in range(-test_range, test_range, 1):

            assert (
                self.tensor.EvalRelative(self.tensor.exp(i / 10), math.exp(i / 10))
                < self.threshold
            )

    def ExpTest(self):

        test_range = 10
        for i in range(-test_range, test_range, 1):

            assert (
                self.tensor.EvalRelative(self.tensor.exp(i / 10), math.exp(i / 10))
                < self.threshold
            )

    def tanhTest(self):

        test_range = 9
        for i in range(-test_range, test_range, 1):

            assert (
                self.tensor.EvalRelative(self.tensor.tanh(i / 10), math.tanh(i / 10))
                < self.threshold
            )

    def LogTest(self):

        test_range = 4
        for i in range(1, test_range, 1):

            assert (
                self.tensor.EvalRelative(self.tensor.log(i / 10), math.log(i / 10))
                < self.threshold
            )
