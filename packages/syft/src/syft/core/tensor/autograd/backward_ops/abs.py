# stdlib
import uuid

# third party
import numpy as np

# syft relative
from ...passthrough import is_acceptable_simple_type
from ..tensor import AutogradTensor
from .op import Op


class AbsOp(Op):
    def forward(self, x: AutogradTensor):
        self.x = x

        return AutogradTensor(x.child.__abs__(), requires_grad=x.requires_grad)

    def _backward(self, grad, backprop_id):

        if self.x.requires_grad:

            _grad = self.x.child > 0  # returns 0s and 1s
            _grad = (_grad * 2) - 1  # returns -1s and 1s

            grad = _grad * grad.child

            self.x.add_grad(AutogradTensor(grad))

            if self.x.grad_fn:
                self.x.backward(backprop_id=backprop_id)
