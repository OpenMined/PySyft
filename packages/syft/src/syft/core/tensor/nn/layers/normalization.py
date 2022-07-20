# stdlib
from typing import Dict
from typing import Optional
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
        activation: Optional[str] = None,
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

        N, C, H, W = prev_layer.out_shape
        self.beta = np.zeros((1, C, 1, 1))
        self.gamma = np.zeros((1, C, 1, 1))
        self.input_shape = prev_layer.out_shape
        self.out_shape = prev_layer.out_shape

    def forward(self, input: Union[PhiTensor, GammaTensor], *args, **kwargs):
        # N, D = x.shape

        # step1: calculate the mean
        # mean calc along N, H, W axis
        # Reference: https://arxiv.org/pdf/1803.08494.pdf
        mean = input.mean(axis=(0, 2, 3), keepdims=True)

        xmu = input - mean

        # step2:
        # sq = xmu ** 2
        # var = 1. / N * np.sum(sq, axis=(N, H, W))
        # std calc along N, H, W axis
        # Reference: https://arxiv.org/pdf/1803.08494.pdf
        var = xmu.std(axis=(0, 2, 3), keepdims=True)

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

        # mean, std, sum is calculated along N, H, W axis
        # Reference: https://arxiv.org/pdf/1803.08494.pdf

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
            self.activation.derivative(pre_grad)
            if self.activation is not None
            else pre_grad
        )

        # step6
        # sum is calculated along N, H, W axis
        self.dbeta = pre_grad.sum(axis=(0, 2, 3), keepdims=True)
        dgammax = pre_grad

        # sum is calculated along N, H, W axis
        self.dgamma = (dgammax * xhat).sum(axis=(0, 2, 3), keepdims=True)
        dxhat = dgammax * self.gamma

        # step5
        # var is calculated along N, H, W axis
        divar = (dxhat * xmu).sum(axis=(0, 2, 3), keepdims=True)
        dxmu1 = dxhat * ivar

        # step4
        dsqrtvar = -1.0 / (sqrtvar * sqrtvar) * divar
        inv_var_eps_sqrt = 1.0 / (var + self.epsilon).sqrt()
        dvar = dsqrtvar * 0.5 * inv_var_eps_sqrt

        # step3
        dsq = (pre_grad.ones_like()) * dvar * (1.0 / N)
        dxmu2 = xmu * dsq * 2

        # step2
        dx1 = dxmu1 + dxmu2

        # step1
        # sum is calculated along N, H, W axis
        dmu = (dxmu1 + dxmu2).sum(axis=(0, 2, 3), keepdims=True) * -1
        dx2 = (pre_grad.ones_like()) * dmu * (1 / N)

        # step0 done!
        dx = dx1 + dx2

        return dx

    @property
    def params(self):
        return self.beta, self.gamma

    @params.setter
    def params(self, new_params):
        if len(new_params) != 2:
            raise ValueError(
                f"Expected Two values Update Params has length{len(new_params)}"
            )
        self.beta, self.gamma = new_params

    @property
    def grads(self):
        return self.dbeta, self.dgamma
