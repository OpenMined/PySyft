# stdlib
from uuid import UUID

# third party
import numpy as np

# relative
from .....core.common.serde.serializable import serializable
from ..tensor import AutogradTensor
from .op import Op


@serializable(recursive_serde=True)
class SumOp(Op):
    """Sum operation across a dimension"""

    __attr_allowlist__ = ["x", "axis", "dim_at_axis", "backward_shape"]

    def forward(self, x: AutogradTensor, axis: int) -> AutogradTensor:  # type: ignore
        self.x = x
        self.axis = axis
        if axis is not None:
            # obj.sum() can be called without dims
            self.dim_at_axis = self.x.shape[self.axis]
        else:
            self.dim_at_axis = None  # type: ignore
        self.backward_shape = self.x.shape

        result = x.child.sum(axis=axis)

        if result.shape == ():
            result = result.reshape(1)

        return AutogradTensor(result, requires_grad=x.requires_grad)

    def _backward(self, grad: np.ndarray, backprop_id: UUID) -> None:

        if self.x.requires_grad:

            if self.axis is not None:

                grad = np.expand_dims(grad, self.axis)
                grad = grad.repeat(self.dim_at_axis, axis=self.axis)

            else:
                n_times = np.prod(self.backward_shape)  # type: ignore
                grad = grad.repeat(n_times, axis=0).reshape(self.backward_shape)

            self.x.add_grad(grad)

            if self.x.grad_fn:
                self.x.backward(backprop_id=backprop_id)
