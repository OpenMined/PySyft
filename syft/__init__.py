from syft import he
from syft import nn

from syft.tensor import equal, TensorBase
from syft.math import cumprod, cumsum, ceil, dot, matmul, addmm, addcmul
from syft.math import addcdiv, addmv, addbmm, baddbmm, transpose

s = str(he)
s += str(nn)

s += str(equal) + str(TensorBase) + str(cumprod) + str(cumsum) + str(ceil)
s += str(dot) + str(matmul) + str(addmm) + str(addcmul) + str(addcdiv)
s += str(addmv) + str(addbmm) + str(baddbmm)
s += str(transpose)
