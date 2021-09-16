# stdlib
from uuid import UUID

# third party
from numpy import ndarray

# relative
from .....core.common.serde.serializable import serializable
from ...passthrough import is_acceptable_simple_type  # type: ignore
from ..tensor import AutogradTensor
from .op import Op


@serializable(recursive_serde=True)
class MulOp(Op):
    """Multiplication operation with 2 tensors"""

    __attr_allowlist__ = [
        "x",
        "y",
    ]

    def forward(self, x: AutogradTensor, y: AutogradTensor) -> AutogradTensor:  # type: ignore
        self.x = x
        self.y = y

        requires_grad = x.requires_grad

        if is_acceptable_simple_type(y):
            return AutogradTensor(x.child * y, requires_grad=requires_grad)

        # print(y)
        requires_grad = requires_grad or y.requires_grad

        # print()
        # print()
        # print("mul._backward")
        # print(x.child)
        # print()
        # print(y.child)
        if is_acceptable_simple_type(y.child):
            return AutogradTensor(x.child * y.child, requires_grad=requires_grad)

        return AutogradTensor(y.child * x.child, requires_grad=requires_grad)

    def _backward(self, grad: ndarray, backprop_id: UUID) -> None:

        y_is_simple = is_acceptable_simple_type(self.y)

        if self.x.requires_grad:

            if y_is_simple:
                self.x.add_grad(grad * self.y)
            else:
                temp = self.y * grad
                self.x.add_grad(temp)

            if self.x.grad_fn:
                self.x.backward(backprop_id=backprop_id)

        # if y_is_simple then it's definitely not an autograd tensor (doesn't need to be
        # backpropagated into. also if it doesn't .requires_grad
        if not y_is_simple and self.y.requires_grad:
            self.y.add_grad(self.x * grad)
            if self.y.grad_fn:
                self.y.backward(backprop_id=backprop_id)
