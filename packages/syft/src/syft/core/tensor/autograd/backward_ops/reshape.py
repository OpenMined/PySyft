# syft relative
# stdlib
from uuid import UUID

# third party
from numpy import ndarray

# relative
from ..tensor import AutogradTensor
from .op import Op


class ReshapeOp(Op):
    """Multiplication operation with 2 tensors"""

    def forward(self, x: AutogradTensor, *shape: tuple) -> AutogradTensor:
        self.x = x
        self.shape = shape
        self.backward_shape = self.x.shape

        return AutogradTensor(x.child.reshape(*shape), requires_grad=x.requires_grad)

    def _backward(self, grad: ndarray, backprop_id: UUID) -> None:

        if self.x.requires_grad:
            self.x.add_grad(grad.reshape(*self.backward_shape))

            if self.x.grad_fn:
                self.x.backward(backprop_id=backprop_id)
