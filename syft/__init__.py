from syft.syft import FloatTensor
from syft.nn import Linear, Sigmoid, Model, MSELoss
from syft import controller
from syft import nn
import imp

def reload():
    imp.reload(syft)
    imp.reload(nn)
    imp.reload(controller)
