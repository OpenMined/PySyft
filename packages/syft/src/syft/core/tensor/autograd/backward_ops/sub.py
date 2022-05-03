# stdlib
from uuid import UUID

# third party
import numpy as np

# relative
from ...passthrough import is_acceptable_simple_type  # type: ignore
from ..tensor import AutogradTensor  # type: ignore
from .op import Op  # type: ignore


class SubOp(Op):
    """Substraction operation with 2 tensors"""

    def forward(self, x: AutogradTensor, y: AutogradTensor) -> AutogradTensor:  # type: ignore
        self.x = x
        self.y = y

        requires_grad = x.requires_grad

        if is_acceptable_simple_type(y):
            return AutogradTensor(x.child - y, requires_grad=requires_grad)

        requires_grad = requires_grad or y.requires_grad
        # print()
        # print(x)
        # print(y)

        return AutogradTensor(x.child - y.child, requires_grad=requires_grad)

    def _backward(self, grad: np.ndarray, backprop_id: UUID) -> None:

        if self.x.requires_grad:
            # as we have matrix operation one of the parameters can
            # have partial shape in such scenarion we need to sum
            # gradient values by missed axis
            if self.x.shape != grad.shape:
                print("shapes don't match")
                axis = np.argmax(np.abs(np.array(self.x.shape) - np.array(grad.shape)))
                self.x.add_grad(grad.sum(axis=axis, keepdims=True))
            else:

                self.x.add_grad(grad)

            if self.x.grad_fn:
                self.x.backward(backprop_id=backprop_id)

        if self.y.requires_grad:
            if self.y.shape != grad.shape:
                print("shapes don't match")

                axis = np.argmax(np.abs(np.array(self.y.shape) - np.array(grad.shape)))
                self.y.add_grad(-(grad.sum(axis=axis, keepdims=True)))
            else:
                self.y.add_grad(-grad)

            if self.y.grad_fn:
                self.y.backward(backprop_id=backprop_id)
