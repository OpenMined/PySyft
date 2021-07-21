# future
from __future__ import annotations

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
import numpy as np
import torch as th

# relative
# syft relative
from ...core.common.serde.recursive import RecursiveSerde
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
    PassthroughTensor, AutogradTensorAncestor, PhiTensorAncestor, RecursiveSerde
):

    __attr_allowlist__ = ['child']

    def __init__(self, child):
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
