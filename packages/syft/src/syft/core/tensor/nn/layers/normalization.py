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

        ivar = PhiTensor(
            child=sqrtvar.child ** -1,
            data_subjects=sqrtvar.data_subjects,
            min_vals=lra(data=1/sqrtvar.min_vals.data,shape=sqrtvar.shape),
            max_vals=lra(data=1/sqrtvar.max_vals.data,shape=sqrtvar.shape)
        )

        # step5: normalization->x^
        xhat = xmu * ivar

        # step6: scale and shift
        gammax = xhat * self.gamma
        out = gammax + self.beta

        self.cache = (xhat, xmu, ivar, sqrtvar, var)

        return out

    def backward(self, pre_grad, *args, **kwargs):
        xhat, xmu, ivar, sqrtvar, var = self.cache

        N, D = pre_grad.shape

        # step6
        self.dbeta = np.sum(pre_grad, axis=0)
        dgammax = pre_grad
        self.dgamma = np.sum(dgammax * xhat, axis=0)
        dxhat = dgammax * self.gamma

        # step5
        divar = np.sum(dxhat * xmu, axis=0)
        dxmu1 = dxhat * ivar

        # step4
        dsqrtvar = -1. / (sqrtvar ** 2) * divar
        dvar = 0.5 * 1. / np.sqrt(var + self.epsilon) * dsqrtvar

        # step3
        dsq = 1. / N * np.ones((N, D)) * dvar
        dxmu2 = 2 * xmu * dsq

        # step2,
        dx1 = (dxmu1 + dxmu2)

        # step1,
        dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)
        dx2 = 1. / N * np.ones((N, D)) * dmu

        # step0 done!
        dx = dx1 + dx2

        return dx

    @property
    def params(self):
        return self.beta, self.gamma

    @property
    def grades(self):
        return self.dbeta, self.dgamma
