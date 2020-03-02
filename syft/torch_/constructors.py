from syft.torch_.native_parameter import ParameterConstructor
from syft.torch_.native_tensor import TensorConstructor

import torch as _th

Tensor = TensorConstructor(_th.Tensor)
tensor = Tensor

Parameter = ParameterConstructor(_th.nn.Parameter)
parameter = Parameter