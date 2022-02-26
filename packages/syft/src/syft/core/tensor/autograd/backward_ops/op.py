# stdlib
from abc import abstractmethod
from typing import Any
from typing import Optional
from uuid import UUID

# third party
from numpy import ndarray

# relative
from ..tensor import AutogradTensor


class Op:
    def __init__(self) -> None:
        self.backprop_id: Optional[UUID] = None

    @abstractmethod
    def forward(self, *args: AutogradTensor, **kwargs: AutogradTensor) -> Any:
        raise NotImplementedError

    def _backward(self, grad: ndarray, backprop_id: UUID) -> Any:
        raise NotImplementedError

    def backward(self, grad: ndarray, backprop_id: UUID) -> Any:

        self.backprop_id = backprop_id

        for t in self.parent_tensors:
            t.backprop_id = backprop_id

        return self._backward(grad=grad, backprop_id=backprop_id)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:

        self.parent_tensors = list()

        for arg in args:
            if isinstance(arg, AutogradTensor):
                arg.ops.append(self)
                self.parent_tensors.append(arg)

        for _, arg in kwargs.items():
            if isinstance(arg, AutogradTensor):
                arg.ops.append(self)
                self.parent_tensors.append(arg)

        self.out = self.forward(*args, **kwargs)
        self.out._grad_fn = self

        return self.out
