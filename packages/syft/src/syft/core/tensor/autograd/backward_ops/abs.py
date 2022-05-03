# stdlib
from uuid import UUID

# third party
from numpy import ndarray

# relative
from ..tensor import AutogradTensor
from .op import Op


class AbsOp(Op):
    def forward(self, x: AutogradTensor) -> AutogradTensor:  # type: ignore
        self.x = x

        return AutogradTensor(x.child.__abs__(), requires_grad=x.requires_grad)

    def _backward(self, grad: ndarray, backprop_id: UUID) -> None:

        if self.x.requires_grad:
            _grad = int(self.x > 0)  # returns 0s and 1s
            _grad = int((_grad * 2) - 1)  # returns -1s and 1s

            grad = _grad * grad

            self.x.add_grad(grad)

            if self.x.grad_fn:
                self.x.backward(backprop_id=backprop_id)
