# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Dict as TypeDict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union
import uuid

# third party
import numpy as np

# relative
from .. import autograd
from ....lib.python.collections.collections import DefaultDict
from ....lib.python.collections.collections import SerializableCounter
from ...common.serde.serializable import serializable
from ..ancestors import AutogradTensorAncestor
from ..ancestors import PhiTensorAncestor
from ..passthrough import AcceptableSimpleType  # type: ignore
from ..passthrough import PassthroughTensor  # type: ignore
from ..passthrough import is_acceptable_simple_type  # type: ignore


@serializable(recursive_serde=True)
class AutogradTensor(PassthroughTensor, PhiTensorAncestor):

    __attr_allowlist__ = [
        "child",
        "requires_grad",
        "_grad",
        "_grad_fn",
        "ops",
        "backprop_id",
        "n_backwards",
    ]

    def __init__(
        self,
        child: Union[Type[AutogradTensor], AcceptableSimpleType],
        requires_grad: bool = False,
    ) -> None:
        super().__init__(child)

        # whether to run backpropagation or not
        self.requires_grad = requires_grad

        # tensor gradient
        self._grad: TypeDict = DefaultDict(lambda: None)

        # operation used to create this tensor (if any)
        self._grad_fn: Optional[Type[autograd.backward_ops.Op]] = None

        # list of ops which use this tensor
        self.ops: List = list()

        self.backprop_id: Optional[uuid.UUID] = None

        self.n_backwards: SerializableCounter = (
            SerializableCounter()
        )  # may have to add [uuid.UUID] for type annotation

    @property
    def grad(self) -> Optional[np.ndarray]:
        if self.backprop_id not in self._grad:
            return None
        return self._grad[self.backprop_id]

    @property
    def grad_fn(
        self,
    ) -> Optional[Type[autograd.backward_ops.Op]]:
        if not self.requires_grad:
            raise Exception("This tensor is not backpropagated")
        return self._grad_fn

    # Autograd Tensor Operations
    """ Note: Ignoring return type incompatibilities since AutogradTensorAncestor doesn't inherit from
    PassThroughTensor"""

    def __abs__(self) -> AutogradTensorAncestor:
        op = autograd.backward_ops.AbsOp()
        return op(self)

    def __add__(self, other: AutogradTensor) -> AutogradTensorAncestor:  # type: ignore
        op = autograd.backward_ops.AddOp()
        return op(self, other)

    def __sub__(self, other: AutogradTensor) -> AutogradTensorAncestor:  # type: ignore
        op = autograd.backward_ops.SubOp()
        return op(self, other)

    def __mul__(self, other: AutogradTensor) -> AutogradTensorAncestor:  # type: ignore
        op = autograd.backward_ops.MulOp()
        return op(self, other)

    def __rmul__(self, other: AutogradTensor) -> AutogradTensorAncestor:  # type: ignore
        op = autograd.backward_ops.MulOp()
        return op(self, other)

    def __truediv__(self, other: AutogradTensor) -> AutogradTensorAncestor:  # type: ignore
        if is_acceptable_simple_type(other):
            # Ignoring type annotation error because only int, floats, np.ndarrays will be parsed
            return self * (1 / other)  # type: ignore
        return NotImplemented

    def __pow__(self, other: Any) -> AutogradTensorAncestor:  # type: ignore
        op = autograd.backward_ops.PowOp()
        return op(self, other)

    def __rpow__(self, other: Any) -> AutogradTensorAncestor:  # type: ignore
        op = autograd.backward_ops.RPowOp()
        return op(self, other)

    def reshape(self, *shape: Tuple[int]) -> AutogradTensorAncestor:  # type: ignore
        op = autograd.backward_ops.ReshapeOp()
        return op(self, *shape)

    def repeat(self, *args: Tuple[Any, ...], **kwargs: Any) -> AutogradTensorAncestor:  # type: ignore
        op = autograd.backward_ops.RepeatOp()
        return op(self, *args, **kwargs)

    def copy(self) -> AutogradTensorAncestor:  # type: ignore
        op = autograd.backward_ops.CopyOp()
        return op(self)

    def sum(self, *args: int, **kwargs: int) -> AutogradTensorAncestor:  # type: ignore
        op = autograd.backward_ops.SumOp()
        return op(self, *args, **kwargs)

    def transpose(self, *dims: tuple) -> AutogradTensorAncestor:  # type: ignore
        op = autograd.backward_ops.TransposeOp()
        return op(self, *dims)

    # End Autograd Tensor Operations

    def add_grad(self, grad: np.ndarray) -> None:

        # print("Adding grad:" + str(type(grad)) + " w/ backprop_id:" + str(self.backprop_id))

        if self._grad[self.backprop_id] is None:
            self._grad[self.backprop_id] = grad
        else:

            self._grad[self.backprop_id] = self._grad[self.backprop_id] + grad

    def backward(
        self,
        grad: Optional[np.ndarray] = None,
        backprop_id: Optional[uuid.UUID] = None,
    ) -> bool:

        if backprop_id is None:
            backprop_id = uuid.uuid4()

        self.n_backwards[backprop_id] += 1

        self.backprop_id = backprop_id

        if not self.grad_fn:
            return False

        if grad is None and self._grad[self.backprop_id] is None:
            # in case if this is last loss tensor
            grad = np.ones(self.shape)
            # grad = self.__class__(grad, requires_grad=False)
            # this more or less ensures it has the right tensor chain
            # grad = (self * 0) + 1

        elif self.grad is not None:
            grad = self._grad[self.backprop_id]

        if not self.requires_grad:
            raise Exception("This tensor is not backpropagated")

        # if all gradients are accounted for - backprop
        if self.n_backwards[backprop_id] >= len(self.ops):

            self.grad_fn.backward(grad, backprop_id=backprop_id)  # type: ignore

        # if some gradietns appear to be missing - parse forward in
        # the graph to double check
        else:

            # investigate whether any of the missing ops are actually
            # going to get used.
            found_id = False

            n_direct_ops = 0
            for op in self.ops:
                if op.backprop_id is not None and op.backprop_id == backprop_id:
                    n_direct_ops += 1

            # if the number of operations we know will be backpropagating gradients to us
            # exceeds the number of times we've been backpropgated into - then we know
            # we need to wait.
            if n_direct_ops > self.n_backwards[backprop_id]:
                found_id = True

            else:

                for op in self.ops:
                    if op.backprop_id is None:
                        if op.out.find_backprop_id(self.backprop_id):
                            found_id = True
                            break

            if found_id:
                "do nothing - we're going to get another gradient"
            else:
                # backprop anyway - we've got all the grads we're gonna get
                self.grad_fn.backward(grad, backprop_id=backprop_id)  # type: ignore

        return True

    def find_backprop_id(self, backprop_id: Optional[uuid.UUID]) -> bool:
        found_id = False

        for op in self.ops:
            if op.backprop_id is not None and op.backprop_id == backprop_id:
                return True

            if op.out.find_backprop_id(self.backprop_id):
                found_id = True
                break

        return found_id
