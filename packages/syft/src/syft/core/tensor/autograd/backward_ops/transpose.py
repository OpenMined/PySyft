# stdlib
from typing import Tuple
from uuid import UUID

# third party
from numpy import ndarray

# relative
from ..tensor import AutogradTensor
from .op import Op


class TransposeOp(Op):
    """Repeat operation across a dimension"""

    def forward(self, x: AutogradTensor, *dims: Tuple[int]) -> AutogradTensor:  # type: ignore
        self.x = x
        self.dims = dims

        reverse_t_dims = {}
        for i, d in enumerate(self.dims):
            reverse_t_dims[d] = i

        l_dims = sorted(
            [(x[0], x[1]) for x in reverse_t_dims.items()], key=lambda x: x[0]
        )
        self.reverse_dims = [x[1] for x in l_dims]

        return AutogradTensor(x.child.transpose(*dims), requires_grad=x.requires_grad)

    def _backward(self, grad: ndarray, backprop_id: UUID) -> None:

        if self.x.requires_grad:

            grad = grad.transpose(*self.reverse_dims)

            self.x.add_grad(grad)

            if self.x.grad_fn:
                self.x.backward(backprop_id=backprop_id)
