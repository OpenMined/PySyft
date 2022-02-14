# stdlib
import uuid

# third party
import numpy as np

# relative
from ...passthrough import is_acceptable_simple_type  # type: ignore
from ..tensor import AutogradTensor
from .op import Op


class PowOp(Op):
    def forward(self, x: AutogradTensor, y: AutogradTensor) -> AutogradTensor:  # type: ignore
        self.x = x
        self.y = y

        requires_grad = x.requires_grad

        if is_acceptable_simple_type(y):
            return AutogradTensor(x.child**y, requires_grad=requires_grad)

        requires_grad = requires_grad or y.requires_grad
        return AutogradTensor(x.child**y.child, requires_grad=requires_grad)

    def _backward(self, grad: AutogradTensor, backprop_id: uuid.UUID) -> None:

        y_is_simple = is_acceptable_simple_type(self.y)

        if self.x.requires_grad:
            y_form = self.y

            # ignoring type b/c method hasn't been implemented yet
            self.x.add_grad(grad * y_form * (self.x ** (y_form - 1)))  # type: ignore

            if self.x.grad_fn:
                self.x.backward(backprop_id=backprop_id)

        if not y_is_simple and self.y.requires_grad:

            self.y.add_grad(np.log(self.x) * grad * self.x**self.y)

            if self.y.grad_fn:
                self.y.backward(backprop_id=backprop_id)
