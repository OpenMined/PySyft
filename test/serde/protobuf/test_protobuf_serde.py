import torch

import syft
from syft.serde import protobuf
from syft.serde.torch.serde import TORCH_STR_DTYPE

from test.serde.serde_helpers import *


dtypes = [
    "uint8",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "float64",
    "bool",
    "bfloat16",
]
quantized_dtypes = ["qint8", "quint8", "qint32"]
complex_types = []  # not yet implemented in PyTorch


@pytest.mark.parametrize("str_dtype", dtypes)
def test_protobuf_serde_tensor_roundtrip(str_dtype):
    """Checks that tensors passed through serialization-deserialization stay same"""

    def compare(roundtrip, original):
        assert type(roundtrip) == torch.Tensor
        assert roundtrip.dtype == original.dtype

        # PyTorch doesn't implement equality checking for bfloat16, so convert to float
        if original.dtype == torch.bfloat16:
            roundtrip = roundtrip.float()
            original = original.float()

        # PyTorch doesn't implement equality checking for float16, so use numpy
        assert numpy.array_equal(roundtrip.data.numpy(), original.data.numpy())
        return True

    serde_worker = syft.hook.local_worker
    original_framework = serde_worker.framework
    serde_worker.framework = None

    tensor = torch.rand([10, 10]) * 16
    tensor = tensor.to(TORCH_STR_DTYPE[str_dtype])

    protobuf_tensor = protobuf.serde._bufferize(serde_worker, tensor)
    roundtrip_tensor = protobuf.serde._unbufferize(serde_worker, protobuf_tensor)

    serde_worker.framework = original_framework

    assert compare(roundtrip_tensor, tensor) is True


# quantized types can't be created by conversion with `tensor.to()`
@pytest.mark.parametrize("str_dtype", quantized_dtypes)
def test_protobuf_serde_tensor_roundtrip_quantized(str_dtype):
    """Checks that tensors passed through serialization-deserialization stay same"""

    def compare(roundtrip, original):
        assert type(roundtrip) == torch.Tensor
        assert roundtrip.dtype == original.dtype
        roundtrip_np = roundtrip.dequantize().numpy()
        original_np = original.dequantize().numpy()
        # PyTorch does implement equality checking for float tensors, but
        # quantized tensors may not be exactly the same after a round trip
        # plus dequantizing so use numpy close checking with a tolerance
        assert numpy.allclose(roundtrip_np, original_np, atol=2 / original.q_scale())
        return True

    serde_worker = syft.hook.local_worker
    original_framework = serde_worker.framework
    serde_worker.framework = None

    tensor = torch.rand([10, 10]) * 16
    tensor = torch.quantize_per_tensor(tensor, 0.1, 10, TORCH_STR_DTYPE[str_dtype])

    protobuf_tensor = protobuf.serde._bufferize(serde_worker, tensor)
    roundtrip_tensor = protobuf.serde._unbufferize(serde_worker, protobuf_tensor)

    serde_worker.framework = original_framework

    assert compare(roundtrip_tensor, tensor) is True
