# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
import numpy as np
import torch as th

# relative
from ...proto.lib.torch.device_pb2 import Device as Device_PB
from ...proto.lib.torch.tensor_pb2 import TensorProto as TorchTensor_PB
from ...lib.torch.tensor_util import tensor_deserializer
from ...lib.torch.tensor_util import tensor_serializer
from .specialized_compressor import SpecializedCompressor
from .util import registered_compressors
from ...logger import warning
from ...core.common.serde.serializable import Serializable
from ..common.serde.deserialize import _deserialize as deserialize
from ..common.serde.serializable import bind_protobuf
from ..common.serde.serialize import _serialize as serialize
from ...proto.core.tensor.tensor_pb2 import Tensor as Tensor_PB
from ..tensor import Tensor
from ...util import get_fully_qualified_name

@bind_protobuf
class CompressedTensor(th.Tensor, Serializable):
    """
    [Experimental: high-performace duet channel] compressed tensor stores the compression algorithms applied
    HpDT TODO: Should be moved into core -> tensor?
    HpDT TODO: Find better alternative for this.
    """
    @staticmethod
    def __new__(cls, child, compressors=[], *args, **kwargs):
        if 'core.tensor' in str(type(child)):
            child = th.Tensor(child.child)
        return super().__new__(cls, child, *args, **kwargs)

    def clone(self, *args, **kwargs):
        return CompressedTensor(self.child, self.compressors)

    def to(self, *args, **kwargs):
        new_obj = CompressedTensor([], self.compressors)
        tempTensor=super().to(*args, **kwargs)
        new_obj.data=tempTensor.data
        new_obj.requires_grad=tempTensor.requires_grad
        new_obj.refresh_child()
        return(new_obj)

    def __init__(self, child: th.Tensor, compressors: List[SpecializedCompressor] = []):
        self.use_tensors = True
        if 'core.tensor' in str(type(child)):
            child = th.Tensor(child.child)
            self.use_tensors = False
        th.Tensor.__init__(child)
        self.child = child
        self.requires_grad = child.requires_grad
        self.compressors = compressors
        if self.requires_grad:
            self.grad = child.grad
            self.compressed_grad = child.grad
        self.refresh_super_tensor()

    def compress_more(self, compressor):
        if getattr(compressor, "grad_compessor", False):
            if compressor.is_eligible(self.compressed_grad):
                if type(compressor) == type:
                    compressor = compressor()
                self.compressed_grad = compressor.compress(self.compressed_grad)
                self.compressors.append(type(compressor))
        else:
            if compressor.is_eligible(self.child):
                if type(compressor) == type:
                    compressor = compressor()
                self.child = compressor.compress(self.child)
                self.compressors.append(type(compressor))

    def decompress(self) -> th.Tensor:
        compressors = self.compressors.copy()
        decompressed = self.child
        print('decompressing', self.child, self.compressors)
        if self.requires_grad:
            decompressed_grad = self.compressed_grad
        while compressors:
            compressor = compressors.pop()
            if getattr(CompressedTensor, "grad_compessor", False):
                decompressed_grad = compressor.decompress(decompressed_grad)
            else:
                decompressed = compressor.decompress(decompressed)
        decompressed.requires_grad = self.requires_grad
        if self.requires_grad:
            decompressed.grad = self.grad
        return decompressed

    def encode_compressors(self) -> np.ndarray:
        encoded_compressors = list()
        for compressor in self.compressors:
            encoded_compressors.append(registered_compressors[compressor])
        return np.array(encoded_compressors)

    def decode_and_attach_compressors(self, encoded):
        self.compressors = self.decode_compressors(encoded)

    def decode_compressors(self, encoded):
        compressors = []
        inv_registered_compressors = {v: k for k, v in registered_compressors.items()}
        for encoded_compressor in encoded:
            compressors.append(inv_registered_compressors[encoded_compressor])
        return compressors

    def refresh_child(self):
        compressed = self.data
        for compressor in self.compressors:
            compressed = compressor().compress(compressed)
        self.child = compressed

    def refresh_super_tensor(self):
        decompressed = self.decompress()
        self.data = decompressed
        self.grad = decompressed.grad

    def __repr__(self):
        return "CompressedTensor(%r, %r)" % (
            self.child,
            self.compressors,
        )

    def _object2proto(self) -> Tensor_PB:
        use_tensors = self.use_tensors
        arrays = [serialize(self.encode_compressors())]

        self.child.requires_grad = self.requires_grad
        if self.requires_grad:
            self.child.grad = self.compressed_grad
        tensors = [torchTensor_object2proto(self.child)]
        return Tensor_PB(
            obj_type=get_fully_qualified_name(obj=self),
            use_tensors=use_tensors,
            arrays=arrays,
            tensors=tensors,
            requires_grad=self.requires_grad,
        )

    @staticmethod
    def _proto2object(proto: Tensor_PB, return_compressed=False):
        child = [torchTensor_proto2object(tensor) for tensor in proto.tensors]
        child = child[0]

        res = CompressedTensor(child, [])

        encoded_compressors = [deserialize(array) for array in proto.arrays]
        res.decode_and_attach_compressors(encoded_compressors[0])
        res.refresh_super_tensor()
        res.use_tensors = proto.use_tensors

        if not return_compressed:
            if res.use_tensors:
                return th.Tensor(res)
            return Tensor(res)
        return res

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Tensor_PB

def torchTensor_object2proto(obj: object) -> Tensor_PB:
    proto = TorchTensor_PB()
    proto.tensor = tensor_serializer(obj)

    proto.requires_grad = getattr(obj, "requires_grad", False)
    proto.device.CopyFrom(
        Device_PB(
            type=obj.device.type,  # type: ignore
            index=obj.device.index,  # type: ignore
        )
    )

    if proto.requires_grad:
        grad = getattr(obj, "grad", None)
        if grad is not None:
            proto.grad = tensor_serializer(grad)

    return proto

def torchTensor_proto2object(proto: Tensor_PB) -> th.Tensor:
    tensor = tensor_deserializer(proto.tensor)
    if proto.requires_grad:
        tensor.grad = tensor_deserializer(proto.grad)

    tensor.requires_grad_(proto.requires_grad)

    if proto.device.type == "cuda" and th.cuda.is_available():
        cuda_index = proto.device.index
        if th.cuda.device_count() < (cuda_index + 1):
            cuda_index = th.cuda.device_count() - 1
            warning(
                f"The requested CUDA index {proto.device.index} is invalid."
                + f"Falling back to GPU index {cuda_index}.",
                print=True,
            )
        return tensor.cuda(cuda_index)

    if proto.device.type == "cuda" and not th.cuda.is_available():
        warning("Cannot find any CUDA devices, falling back to CPU.", print=True)

    return tensor