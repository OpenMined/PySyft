# future
from __future__ import annotations

# stdlib
from typing import Type
from typing import Union
from typing import Any

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
import numpy as np
from numpy.typing import ArrayLike
import torch as th

# relative
from ...core.common.serde.serializable import Serializable
from ...proto.core.tensor.tensor_pb2 import Tensor as Tensor_PB
from ..common.serde.deserialize import _deserialize as deserialize
from ..common.serde.serializable import bind_protobuf
from ..common.serde.serialize import _serialize as serialize
from .fixed_precision_tensor_ancestor import FixedPrecisionTensorAncestor
from .passthrough import PassthroughTensor  # type: ignore
from .smpc.mpc_tensor_ancestor import MPCTensorAncestor


@bind_protobuf
class Tensor(
    PassthroughTensor,
    MPCTensorAncestor,
    FixedPrecisionTensorAncestor,
    Serializable,
):

    def __init__(
        self,
        child: Union[Type[PassthroughTensor], Type[AutogradTensorAncestor], ArrayLike],
    ) -> None:
        """data must be a list of numpy array
        Unsure if np.typing.ArrayLike alone works
        """

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
        res = Tensor(child)
        return res

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Tensor_PB
