# stdlib
from typing import Any
from typing import Dict
from typing import List

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
import numpy as np
import torch as th

# relative
from .specialized_compressor import SpecializedCompressor
from .util import registered_compressors
from ...core.common.serde.serializable import Serializable
from ..common.serde.deserialize import _deserialize as deserialize
from ..common.serde.serializable import bind_protobuf
from ..common.serde.serialize import _serialize as serialize
from ...proto.core.tensor.tensor_pb2 import Tensor as Tensor_PB

@bind_protobuf
class CompressedTensor(Serializable):
    """
    [Experimental: high-performace duet channel] compressed tensor stores the compression algorithms applied
    HpDT TODO: Should be moved into core -> tensor?
    HpDT TODO: Find better alternative for this.
    """

    def __init__(self, child: th.Tensor, compressors: List[SpecializedCompressor] = []):
        self.child = child
        self.compressors = compressors

    def compress_more(self, compressor):
        if compressor.is_eligible(self.child):
            self.child = compressor.compress(self.child)
            self.compressors.append(compressor)

    def decompress(self) -> th.Tensor:
        compressors = self.compressors
        decommpressed = self.child
        while compressors:
            compressor = compressors.pop()
            decommpressed = compressor.decompress(decommpressed)
        return decommpressed

    def encode_compressors(self) -> np.ndarray:
        encoded_compressors = list()
        for compressor in self.compressors:
            encoded_compressors.append(registered_compressors[compressor])
        return np.array(encoded_compressors)

    def decode_compressors(self, encoded):
        self.compressors = []
        inv_registered_compressors = {v: k for k, v in registered_compressors.items()}
        for encoded_compressor in encoded:
            self.compressors.append(inv_registered_compressors[encoded_compressor])

    def __repr__(self):
        return "CompressedTensor(%r, %r)" % (
            self.child,
            self.compressors,
        )

    def _object2proto(self) -> Tensor_PB:
        use_tensors = True
        arrays = [serialize(self.encode_compressors())]
        tensors = [serialize(self.child)]

        return Tensor_PB(
            use_tensors=use_tensors,
            arrays=arrays,
            tensors=tensors,
        )

    @staticmethod
    def _proto2object(proto: Tensor_PB):
        use_tensors = proto.use_tensors
        child = [deserialize(tensor) for tensor in proto.tensors]
        child = child[0]
        res = CompressedTensor(child, [])

        encoded_compressors = [deserialize(array) for array in proto.arrays]
        res.decode_compressors(encoded_compressors)

        return res

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Tensor_PB