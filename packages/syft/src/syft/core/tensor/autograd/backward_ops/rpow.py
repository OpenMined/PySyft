# stdlib
import uuid

# third party
import numpy as np

# relative
from ...passthrough import is_acceptable_simple_type  # type: ignore
from ..tensor import AutogradTensor
from .op import Op


class RPowOp(Op):
    def forward(self, x: AutogradTensor, y: AutogradTensor) -> AutogradTensor:  # type: ignore
        self.x = x
        self.y = y

        requires_grad = x.requires_grad

        if is_acceptable_simple_type(y):
            return AutogradTensor(y**x.child, requires_grad=requires_grad)

        requires_grad = requires_grad or y.requires_grad
        return AutogradTensor(y.child**x.child, requires_grad=requires_grad)

    def _backward(self, grad: AutogradTensor, backprop_id: uuid.UUID) -> None:

        y_is_simple = is_acceptable_simple_type(self.y)

        if self.x.requires_grad:

            y_form = self.y

            self.x.add_grad(np.log(y_form) * grad * y_form**self.x)

            if self.x.grad_fn:
                self.x.backward(backprop_id=backprop_id)

        if not y_is_simple and self.y.requires_grad:
            # ignore type error b/c method hasn't been implemented yet
            self.y.add_grad(grad * self.x * self.y ** (self.x - 1))  # type: ignore
            if self.y.grad_fn:
                self.y.backward(backprop_id=backprop_id)
