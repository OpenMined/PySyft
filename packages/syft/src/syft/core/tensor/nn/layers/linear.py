# third party
import numpy as np

# relative
from ...autodp.phi_tensor import PhiTensor
from ..initializations import XavierInitialization
from .base import Layer


class Linear(Layer):
    def __init__(self, n_out, n_in=None):
        self.n_out = n_out
        self.n_in = n_in
        self.out_shape = (None, n_out)

        self.W = None
        self.b = None
        self.dW = None
        self.db = None
        self.last_input = None
        self.init = XavierInitialization()

    def connect_to(self, prev_layer=None):
        if prev_layer is None:
            assert self.n_in is not None
            n_in = self.n_in
        else:
            assert len(prev_layer.out_shape) == 2
            n_in = prev_layer.out_shape[-1]

        self.W = self.init((n_in, self.n_out))
        self.b = np.zeros((self.n_out,))

    def forward(self, input: PhiTensor, *args, **kwargs):
        self.last_input = input
        return input.dot(self.W) + self.b

    def backward(self, pre_grad, *args, **kwargs):
        self.dW = np.dot(self.last_input.T, pre_grad)
        self.db = np.mean(pre_grad, axis=0)
        if not self.first_layer:
            return np.dot(pre_grad, self.W.T)

    @property
    def params(self):
        return self.W, self.b

    @property
    def grads(self):
        return self.dW, self.db