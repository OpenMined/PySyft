# future
from __future__ import annotations

# stdlib
from collections import Counter
from collections import defaultdict
from typing import Optional
import uuid

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
import numpy as np

# syft relative
from .. import autograd
from ....core.common.serde.serializable import Serializable
from ....lib.util import full_name_with_name
from ....proto.core.tensor.tensor_pb2 import Tensor as Tensor_PB
from ...common.serde.deserialize import _deserialize as deserialize
from ...common.serde.serializable import bind_protobuf
from ...common.serde.serialize import _serialize as serialize
from ..ancestors import AutogradTensorAncestor
from ..ancestors import PhiTensorAncestor
from ..passthrough import PassthroughTensor
from ..passthrough import is_acceptable_simple_type
from ..util import implements


@bind_protobuf
class AutogradTensor(PassthroughTensor, PhiTensorAncestor, Serializable):
    def __init__(self, child, requires_grad=False):
        super().__init__(child)

        # whether to run backpropagation or not
        self.requires_grad = requires_grad

        # tensor gradient
        self._grad = defaultdict(lambda: None)

        # operation used to create this tensor (if any)
        self._grad_fn = None

        # list of ops which use this tensor
        self.ops = list()

        self.backprop_id: Optional[uuid.uuid4] = None

        self.n_backwards = Counter()

    @property
    def grad(self):
        if self.backprop_id not in self._grad:
            return None
        return self._grad[self.backprop_id]

    @property
    def grad_fn(self):
        if not self.requires_grad:
            raise Exception("This tensor is not backpropagated")
        return self._grad_fn

    # Autograd Tensor Operations
    def __abs__(self) -> AutogradTensorAncestor:
        op = autograd.backward_ops.AbsOp()
        return op(self)

    def __add__(self, other) -> AutogradTensorAncestor:
        op = autograd.backward_ops.AddOp()
        return op(self, other)

    def __sub__(self, other) -> AutogradTensorAncestor:
        op = autograd.backward_ops.SubOp()
        return op(self, other)

    def __mul__(self, other) -> AutogradTensorAncestor:
        op = autograd.backward_ops.MulOp()
        return op(self, other)

    def __rmul__(self, other) -> AutogradTensorAncestor:
        op = autograd.backward_ops.MulOp()
        return op(self, other)

    def __truediv__(self, other) -> AutogradTensorAncestor:
        if is_acceptable_simple_type(other):
            return self * (1 / other)
        return NotImplemented

    def __pow__(self, other) -> AutogradTensorAncestor:
        op = autograd.backward_ops.PowOp()
        return op(self, other)

    def __rpow__(self, other) -> AutogradTensorAncestor:
        op = autograd.backward_ops.RPowOp()
        return op(self, other)

    def reshape(self, *shape) -> AutogradTensorAncestor:
        op = autograd.backward_ops.ReshapeOp()
        return op(self, *shape)

    def repeat(self, *args, **kwargs) -> AutogradTensorAncestor:
        op = autograd.backward_ops.RepeatOp()
        return op(self, *args, **kwargs)

    def copy(self) -> AutogradTensorAncestor:
        op = autograd.backward_ops.CopyOp()
        return op(self)

    def sum(self, *args, **kwargs) -> AutogradTensorAncestor:
        op = autograd.backward_ops.SumOp()
        return op(self, *args, **kwargs)

    def transpose(self, *dims) -> AutogradTensorAncestor:
        op = autograd.backward_ops.TransposeOp()
        return op(self, *dims)

    # End Autograd Tensor Operations

    def add_grad(self, grad):

        # print("Adding grad:" + str(type(grad)) + " w/ backprop_id:" + str(self.backprop_id))

        if self._grad[self.backprop_id] is None:
            self._grad[self.backprop_id] = grad
        else:

            self._grad[self.backprop_id] = self._grad[self.backprop_id] + grad

    def backward(self, grad=None, backprop_id: Optional[uuid.uuid4] = None):

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

            self.grad_fn.backward(grad, backprop_id=backprop_id)

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
                self.grad_fn.backward(grad, backprop_id=backprop_id)

        return True

    def find_backprop_id(self, backprop_id):
        found_id = False

        for op in self.ops:
            if op.backprop_id is not None and op.backprop_id == backprop_id:
                return True

            if op.out.find_backprop_id(self.backprop_id):
                found_id = True
                break

        return found_id

    def _object2proto(self) -> Tensor_PB:
        arrays = []
        tensors = []
        if isinstance(self.child, np.ndarray):
            use_tensors = False
            arrays = [serialize(self.child)]
        else:
            use_tensors = True
            tensors = [serialize(self.child)]

        return Tensor_PB(
            obj_type=full_name_with_name(klass=type(self)),
            use_tensors=use_tensors,
            arrays=arrays,
            tensors=tensors,
            requires_grad=self.requires_grad,
        )

    @staticmethod
    def _proto2object(proto: Tensor_PB) -> AutogradTensor:
        use_tensors = proto.use_tensors
        child = []
        if use_tensors:
            child = [deserialize(tensor) for tensor in proto.tensors]
        else:
            child = [deserialize(array) for array in proto.arrays]

        child = child[0]
        return AutogradTensor(child, requires_grad=proto.requires_grad)

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Tensor_PB
