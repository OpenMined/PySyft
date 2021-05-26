# stdlib
import uuid

# third party
import numpy as np

# syft relative
from ...passthrough import is_acceptable_simple_type
from ..tensor import AutogradTensor
from .op import Op


class PowOp(Op):
    def forward(self, x: AutogradTensor, y: AutogradTensor) -> AutogradTensor:
        self.x = x
        self.y = y

        requires_grad = x.requires_grad

        if is_acceptable_simple_type(y):
            return AutogradTensor(x.child ** y, requires_grad=requires_grad)

        requires_grad = requires_grad or y.requires_grad
        return AutogradTensor(x.child ** y.child, requires_grad=requires_grad)

    def _backward(self, grad: AutogradTensor, backprop_id: uuid.uuid4):

        y_is_simple = is_acceptable_simple_type(self.y)

        if self.x.requires_grad:
            if y_is_simple:
                y_form = self.y
            else:
                y_form = self.y.child

            self.x.add_grad(
                AutogradTensor(
                    grad.child * y_form * (self.x.child ** (y_form - 1)), False
                )
            )

            if self.x.grad_fn:
                self.x.backward(backprop_id=backprop_id)

        if not y_is_simple and self.y.requires_grad:
            self.y.add_grad(
                AutogradTensor(
                    np.log(self.x.child) * grad.child * self.x.child ** self.y.child,
                    False,
                )
            )
            if self.y.grad_fn:
                self.y.backward(backprop_id=backprop_id)
