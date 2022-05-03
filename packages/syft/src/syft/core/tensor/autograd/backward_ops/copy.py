# stdlib
from uuid import UUID

# third party
from numpy import ndarray

# relative
from ..tensor import AutogradTensor
from .op import Op


class CopyOp(Op):
    """Copy a tensor"""

    def forward(self, x: AutogradTensor) -> AutogradTensor:  # type: ignore
        self.x = x

        return AutogradTensor(x.child.copy(), requires_grad=x.requires_grad)

    def _backward(self, grad: ndarray, backprop_id: UUID) -> None:

        if self.x.requires_grad:

            self.x.add_grad(grad.copy())

            if self.x.grad_fn:
                self.x.backward(backprop_id=backprop_id)
