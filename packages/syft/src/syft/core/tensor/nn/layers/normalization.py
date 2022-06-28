# stdlib
from typing import Dict
from typing import Tuple
from typing import Union

# third party
import numpy as np

# relative
from .. import activations
from ....common.serde.serializable import serializable
from ...autodp.phi_tensor import GammaTensor
from ...autodp.phi_tensor import PhiTensor
from .base import Layer


@serializable(recursive_serde=True)
class BatchNorm(Layer):
    __attr_allowlist__ = (
        "epsilon",
        "momentum",
        "axis",
        "activation",
        "beta",
        "dbeta",
        "gamma",
        "dgamma",
        "cache",
        "input_shape",
        "out_shape",
    )

    def __init__(
        self,
        epsilon: float = 1e-6,
        momentum: float = 0.9,
        axis: int = 0,
        activation: activations.Activation = None,
    ):
        self.epsilon = epsilon
        self.momentum = momentum
        self.axis = axis
        self.activation = activations.get(activation)

        self.beta, self.dbeta = None, None
        self.gamma, self.dgamma = None, None
        self.cache = None
        self.input_shape = None
        self.out_shape = None

    def connect_to(self, prev_layer):
        n_in = prev_layer.out_shape[-1]
        self.beta = np.zeros((n_in,))
        self.gamma = np.ones((n_in,))
        self.out_shape = prev_layer.out_shape

    def forward(self, input: PhiTensor, *args, **kwargs):
        # N, D = x.shape

        # step1: calculate the mean
        xmu = input - input.mean(axis=0)
        # step2:
        # sq = xmu ** 2
        # var = 1. / N * np.sum(sq, axis=0)
        var = xmu.std(axis=0)

        # step3:
        sqrtvar = (var + self.epsilon).sqrt()
        ivar = 1.0 / sqrtvar

        # step4: normalization->x^
        xhat = xmu * ivar

        # step5: scale and shift
        gammax = xhat * self.gamma
        out = gammax + self.beta

        out = self.activation.forward(out) if self.activation is not None else out

        self.cache = (xhat, xmu, ivar, sqrtvar, var)

        return out

    def backward(
        self, pre_grad: Union[PhiTensor, GammaTensor], *args: Tuple, **kwargs: Dict
    ):
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
        pre_grad = (
            (pre_grad * self.activation.derivative())
            if self.activation is not None
            else pre_grad
        )

        # step6
        self.dbeta = pre_grad.sum(axis=0)
        dgammax = pre_grad
        self.dgamma = (dgammax * xhat).sum(axis=0)
        dxhat = dgammax * self.gamma

        # step5
        divar = (dxhat * xmu).sum(axis=0)
        dxmu1 = dxhat * ivar

        # step4
        dsqrtvar = -1.0 / (sqrtvar * sqrtvar) * divar
        inv_var_eps_sqrt = 1.0 / (var + self.epsilon).sqrt()
        dvar = dsqrtvar * 0.5 * inv_var_eps_sqrt

        # step3
        dsq = (pre_grad.ones_like()) * dvar * (1.0 / N)
        dxmu2 = xmu * dsq * 2

        # step2,
        dx1 = dxmu1 + dxmu2

        # step1,
        dmu = (dxmu1 + dxmu2).sum(axis=0) * -1
        dx2 = ((1 / N) * pre_grad.ones_like()) * dmu

        # step0 done!
        dx = dx1 + dx2

        return dx

    @property
    def params(self):
        return self.beta, self.gamma

    @property
    def grades(self):
        return self.dbeta, self.dgamma
