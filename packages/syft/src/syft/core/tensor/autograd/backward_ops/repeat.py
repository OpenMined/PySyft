# stdlib
from typing import Optional
from uuid import UUID

# third party
from numpy import ndarray

# relative
from ....common.serde.serializable import serializable
from ..tensor import AutogradTensor
from .op import Op


@serializable(recursive_serde=True)
class RepeatOp(Op):
    """Repeat operation across a dimension"""

    __attr_allowlist__ = ["x", "axis", "repeats", "input_shape", "output_shape"]

    def forward(  # type: ignore
        self, x: AutogradTensor, repeats: int, axis: Optional[int] = None
    ) -> AutogradTensor:
        self.x = x
        self.repeats = repeats
        self.axis = axis

        self.input_shape = self.x.shape

        output = x.child.repeat(repeats=repeats, axis=axis)

        self.output_shape = output.shape

        return AutogradTensor(output, requires_grad=x.requires_grad)

    def _backward(self, grad: ndarray, backprop_id: UUID) -> None:

        if self.x.requires_grad:

            axis = self.axis
            if axis is None:
                axis = len(self.input_shape) - 1

            intermediate_shape = list(self.input_shape)
            intermediate_shape.insert(axis + 1, -1)

            if self.x.shape[self.axis] == 1:  # type: ignore
                grad = grad.sum(axis=axis)
            else:
                grad = grad.reshape(*intermediate_shape)
                grad = grad.sum(axis=axis + 1)

            self.x.add_grad(grad)

            if self.x.grad_fn:
                self.x.backward(backprop_id=backprop_id)
