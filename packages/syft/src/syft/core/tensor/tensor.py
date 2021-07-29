# future
from __future__ import annotations

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
import numpy as np
import torch as th

# relative
# syft relative
from ...core.common.serde.serializable import Serializable
from ...lib.util import full_name_with_name
from ...proto.core.tensor.tensor_pb2 import Tensor as Tensor_PB
from ..common.serde.deserialize import _deserialize as deserialize
from ..common.serde.serializable import bind_protobuf
from ..common.serde.serialize import _serialize as serialize
from .ancestors import AutogradTensorAncestor
from .ancestors import PhiTensorAncestor
from .passthrough import PassthroughTensor


@bind_protobuf
class Tensor(
    PassthroughTensor, AutogradTensorAncestor, PhiTensorAncestor, Serializable
):
    def __init__(self, child) -> None:
        """data must be a list of numpy array"""

        if isinstance(child, list):
            child = np.array(child)

        if isinstance(child, th.Tensor):
            child = child.numpy()

        if not isinstance(child, PassthroughTensor) and not isinstance(
            child, np.ndarray
        ):
            raise Exception("Data must be list or nd.array")

        super().__init__(child=child)

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
        )

    @staticmethod
    def _proto2object(proto: Tensor_PB) -> Tensor:
        use_tensors = proto.use_tensors
        child = []
        if use_tensors:
            child = [deserialize(tensor) for tensor in proto.tensors]
        else:
            child = [deserialize(array) for array in proto.arrays]

        child = child[0]
        return Tensor(child)

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Tensor_PB
