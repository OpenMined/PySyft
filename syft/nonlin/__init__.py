from .polyfunction import PolyFunction
from .activations import Activation, PolynomialActivation, SquareActivation
from .activations import LinearSquareActivation, CubicActivation
from .activations import SigmoidActivation, LinearCubicActivation


s = str(Activation)
s += str(PolynomialActivation)
s += str(SquareActivation)
s += str(CubicActivation)
s += str(LinearSquareActivation)
s += str(LinearCubicActivation)
s += str(SigmoidActivation)