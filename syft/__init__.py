from syft import he
from syft import nn
from syft import test
from syft import mpc

from syft.tensor import equal, TensorBase
from syft.math import cumprod, cumsum, ceil, dot, matmul, addmm, addcmul
from syft.math import addcdiv, addmv, bmm, addbmm, baddbmm, transpose
from syft.math import unsqueeze, zeros, ones, rand, randn, mm, fmod, diag, lerp, renorm, numel

s = str(he)
s += str(nn)
s += str(test)
s += str(mpc)

s += str(equal) + str(TensorBase) + str(cumprod) + str(cumsum) + str(ceil)
s += str(dot) + str(matmul) + str(addmm) + str(addcmul) + str(addcdiv)
s += str(addmv) + str(bmm) + str(addbmm) + str(baddbmm)
s += str(transpose) + str(rand) + str(randn) + str(ones) + str(zeros)
s += str(unsqueeze)
s += str(mm) + str(diag)

s += str(fmod)
s += str(lerp)
s += str(numel)
s += str(renorm)
