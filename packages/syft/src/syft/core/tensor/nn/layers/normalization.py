# third party
import numpy as np

# relative
from ...autodp.phi_tensor import PhiTensor
from ...lazy_repeat_array import lazyrepeatarray as lra
from .base import Layer


class BatchNorm(Layer):
    def __init__(self, epsilon=1e-6, momentum=0.9, axis=0):
        self.epsilon = epsilon
        self.momentum = momentum
        self.axis = axis

        self.beta, self.dbeta = None, None
        self.gamma, self.dgamma = None, None
        self.cache = None

    def connect_to(self, prev_layer):
        n_in = prev_layer.out_shape[-1]
        self.beta = np.zeros((n_in,))
        self.gamma = np.ones((n_in,))

    def forward(self, input: PhiTensor, *args, **kwargs):
        # N, D = x.shape
        self.out_shape = input.shape

        # step1: calculate the mean
        xmu = input - input.mean(axis=0)
        # step3:
        var = xmu.std(axis=0)
        sqrtvar = (var + self.epsilon).sqrt()
        ivar = 1.0 / sqrtvar
        # step5: normalization->x^
        xhat = xmu * ivar

        # step6: scale and shift
        gammax = xhat * self.gamma
        out = gammax + self.beta

        self.cache = (xhat, xmu, ivar, sqrtvar, var)
        return out

    def backward(self, pre_grad, *args, **kwargs):
        """
        If you get stuck, here's a resource:
        https://kratzert.github.io/2016/02/12/understanding-the-
        gradient-flow-through-the-batch-normalization-layer.html

        Note:
            - I removed the np.ones() at a few places where I
               thought it wasn't making a difference
            - I occasionally have kernel crashes on my 8GB machine
            when running this. Perhaps too many large temp vars?
            could also be due to too many large numbers.
        """

        xhat, xmu, ivar, sqrtvar, var = self.cache

        N, D, x, y = pre_grad.shape
        #         print(f"input shape of (N,D,x,y) = {(N, D, x, y)}")

        # step6
        self.dbeta = pre_grad.sum(axis=0)
        dgammax = pre_grad
        self.dgamma = (dgammax * xhat).sum(axis=0)
        dxhat = dgammax * self.gamma
        #         print(f"step 6: shaep of dbeta = {self.dbeta.shape}")
        #         print(f"step 6: shaep of dgamma = {self.dgamma.shape}")
        #         print(f"step 6: shaep of dxhat = {dxhat.shape}")

        # step5
        divar = (dxhat * xmu).sum(axis=0)
        dxmu1 = dxhat * ivar

        # step4
        dsqrtvar = -1.0 / (sqrtvar * sqrtvar) * divar
        #         print(f"step 4: shaep of dsqrtvar = {dsqrtvar.shape}")
        inv_var_eps_sqrt = 1.0 / (var + self.epsilon).sqrt()

        #         print(f"var + eps shape:", inv_var_eps_sqrt.shape)
        dvar = dsqrtvar * 0.5 * inv_var_eps_sqrt
        #         print(f"dvar shape:", dvar.shape)


        # step3
        dxmu2 = xmu * dvar * (2/N)

        # step2,
        dx1 = (dxmu1 + dxmu2)

        #         # step1,
        dmu = (dxmu1 + dxmu2).sum(axis=0) * -1
        dx2 = dmu * (1/N)

        # step0 done!
        dx = dx1 + dx2

        return dx

    @property
    def params(self):
        return self.beta, self.gamma

    @property
    def grades(self):
        return self.dbeta, self.dgamma
